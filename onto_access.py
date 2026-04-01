import atexit
import contextlib
import fcntl
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from enum import Enum
from pathlib import Path

import owlready2

# DH: We have removed the dependency on rdflib. It was only needed to
# satisfy a return-type hint on method getGraph(). We removed the hint,
# and so we comment-out the import of rdflib.  Eventually we will remove
# this commented-out import altogether.
#import rdflib
# JD. this is till conditionally imported at points below 
# (can review alternative solutions at some point)


from owlready2 import sync_reasoner, sync_reasoner_pellet
from log_utils import warn, warning, success, info, step, error


# owlready2 World persistence cache
# ---------------------------------
# caches the parsed RDF quadstore to a SQLite file on disk so that repeated pipeline runs 
# against the same ontology skip the expensive OWL/RDF parse; uses a build-once gate with 
# copy-on-open strategy to avoid SQLite write-lock contention between concurrent processes
# ---
# default cache location: ~/.cache/logmap-llm/owlready2/
# process-private copies: /tmp/logmap-llm-owlcache-*/
# ---

# NOTE: we're probably using a more complex solution than is neccesarily required here.
# while this solution (owlready2 + a cache with readwrite locking) was originally optimal
# before making changes to LogMap for property detection, it should be possible to expose
# Java methods that we can use instead of this solution. TODO: this should be revisited.

_TEMP_DIR_PREFIX = 'logmap-llm-owlcache-'


# global registry of tmp directories this process creates for cleanup
# TODO: look into whether this is a potential cause of disk util
_temp_dirs_to_cleanup: list[str] = []


def _register_temp_cleanup(temp_dir: str) -> None:
    _temp_dirs_to_cleanup.append(temp_dir)


def _cleanup_temp_dirs() -> None:
    for d in _temp_dirs_to_cleanup:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def _sanitise_filename(name: str, max_len: int = 80) -> str:
    sanitised = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return sanitised[:max_len]


def _canonical_cache_path(onto_filepath: str, cache_dir: str) -> Path:
    """
    obtain a deterministic cache path for an ontology file

    returns a Path like:
        <cache_dir>/<sanitised_name>_<short_hash>.sqlite3

    short hash ensures collision safety when different directories contain 
    identically-named ontology files

    TODO: make configurable cache paths via config
    """
    resolved = str(Path(onto_filepath).resolve())
    short_hash = hashlib.sha256(resolved.encode()).hexdigest()[:12]
    base_name = Path(onto_filepath).name
    sanitised = _sanitise_filename(base_name)
    cache_filename = f'{sanitised}_{short_hash}.sqlite3'
    return Path(cache_dir) / cache_filename


def _is_cache_valid(onto_filepath: str, cache_path: Path) -> bool:
    """
    check whether a canonical cache file is valid (exists, non-empty, newer than src onto file)
    """
    if not cache_path.exists():
        return False
    if cache_path.stat().st_size == 0:
        return False
    source_mtime = Path(onto_filepath).stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime
    return cache_mtime >= source_mtime


# TODO: move to appropriate helper location (this is duplicated elsewhere)
def _format_age(seconds: float) -> str:
    """format a duration in seconds as a human-readable age string"""
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.0f}m'
    # else:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f'{hours}h {minutes}m'


def _build_or_wait_for_cache(onto_filepath: str, cache_dir: str) -> Path:
    """
    build-once gate 
    ensures a valid 'default' cache exists

    uses an exclusive file lock so that exactly one process builds the cache
    other concurrent processes block until it is ready, then verify the result and proceed
    returns the Path to the default cache file
    """
    cache_path = _canonical_cache_path(onto_filepath, cache_dir)
    lock_path = cache_path.with_suffix('.sqlite3.lock')
    os.makedirs(cache_dir, exist_ok=True)

    onto_display = Path(onto_filepath).name

    # cache already valid, no lock needed (happy path)
    if _is_cache_valid(onto_filepath, cache_path):
        age = time.time() - cache_path.stat().st_mtime
        success(f'Using cached quadstore for {onto_display} (cache age: {_format_age(age)})')
        return cache_path

    # need to acquire lock and potentially build
    lock_fd = open(lock_path, 'w')
    try:
        info(f'Acquiring cache lock for {onto_display} ...')
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # check after acquiring the lock (another process may have built the cache while we were waiting)
        if _is_cache_valid(onto_filepath, cache_path):
            age = time.time() - cache_path.stat().st_mtime
            success(f'Using cached quadstore for {onto_display} (built by another process, age: {_format_age(age)})')
            return cache_path

        # (we're building) parse into a temp file in the same directory 
        # NOTE: we use the same filesystem to atomically rename
        step(f'Building owlready2 cache for {onto_display} ...')
        build_start = time.time()

        temp_fd, temp_build_path = tempfile.mkstemp(suffix='.sqlite3.tmp', dir=cache_dir)
        os.close(temp_fd)  # owlready2 opens the file itself

        try:
            build_world = owlready2.World(filename=temp_build_path)
            build_world.get_ontology(str(onto_filepath)).load()
            build_world.save()
            
            # close the world's SQLite connection before renaming
            build_world.close()

            # atomic rename (either completes or fails)
            os.replace(temp_build_path, str(cache_path))

            elapsed = time.time() - build_start
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            success(f'Cache built for {onto_display} in {elapsed:.1f}s ({size_mb:.0f} MB)')
        
        except Exception:
            # cleanup on failure
            with contextlib.suppress(OSError):
                os.unlink(temp_build_path)
            raise

        return cache_path

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _copy_to_private_temp(cache_path: Path) -> str:
    """
    copy a default cache file to a process-private temp directory
    returns the path to the priv copy. registers tmp dir for cleanup
    """
    temp_dir = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
    _register_temp_cleanup(temp_dir)
    private_path = os.path.join(temp_dir, cache_path.name)
    shutil.copy2(str(cache_path), private_path)
    info(f'Working copy: {private_path}')
    return private_path


def _get_cached_world(onto_filepath: str, cache_dir: str) -> owlready2.World:
    """
    obtain an owlready2 world backed by a cached quadstore, handles:
    1. build-or-wait for the default cache
    2. copy to a private tmp file
    3. open private copy
    """
    canonical = _build_or_wait_for_cache(onto_filepath, cache_dir)
    private_path = _copy_to_private_temp(canonical)
    world = owlready2.World(filename=private_path)
    return world


class Reasoner(Enum):
    HERMIT = 0  # Not really adding the right set of entailments
    PELLET = 1  # Slow for large ontologies
    STRUCTURAL = 2  # Basic domain/range propagation
    NONE = 3  # No reasoning


class AnnotationURIs:
    """Manages the most common ontology annotations."""

    def __init__(self) -> None:
        self.mainLabelURIs = set()
        self.synonymLabelURIs = set()
        self.lexicalAnnotationURIs = set()

        # Main labels
        self.mainLabelURIs.add("http://www.w3.org/2000/01/rdf-schema#label")
        self.mainLabelURIs.add("http://www.w3.org/2004/02/skos/core#prefLabel")
        self.mainLabelURIs.add("http://purl.obolibrary.org/obo/IAO_0000111")
        self.mainLabelURIs.add("http://purl.obolibrary.org/obo/IAO_0000589")

        # synonyms or alternative names
        self.synonymLabelURIs.add("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym")
        self.synonymLabelURIs.add("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
        self.synonymLabelURIs.add("http://www.geneontology.org/formats/oboInOWL#hasExactSynonym")
        self.synonymLabelURIs.add("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym")
        self.synonymLabelURIs.add("http://purl.bioontology.org/ontology/SYN#synonym")
        self.synonymLabelURIs.add("http://scai.fraunhofer.de/CSEO#Synonym")
        self.synonymLabelURIs.add("http://purl.obolibrary.org/obo/synonym")
        self.synonymLabelURIs.add("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#FULL_SYN")
        self.synonymLabelURIs.add("http://www.ebi.ac.uk/efo/alternative_term")
        self.synonymLabelURIs.add("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#Synonym")
        self.synonymLabelURIs.add("http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#Synonym")
        self.synonymLabelURIs.add("http://www.geneontology.org/formats/oboInOwl#hasDefinition")
        self.synonymLabelURIs.add("http://bioontology.org/projects/ontologies/birnlex#preferred_label")
        self.synonymLabelURIs.add("http://bioontology.org/projects/ontologies/birnlex#synonyms")
        self.synonymLabelURIs.add("http://www.w3.org/2004/02/skos/core#altLabel")
        self.synonymLabelURIs.add("https://cfpub.epa.gov/ecotox#latinName")
        self.synonymLabelURIs.add("https://cfpub.epa.gov/ecotox#commonName")
        self.synonymLabelURIs.add("https://www.ncbi.nlm.nih.gov/taxonomy#scientific_name")
        self.synonymLabelURIs.add("https://www.ncbi.nlm.nih.gov/taxonomy#synonym")
        self.synonymLabelURIs.add("https://www.ncbi.nlm.nih.gov/taxonomy#equivalent_name")
        self.synonymLabelURIs.add("https://www.ncbi.nlm.nih.gov/taxonomy#genbank_synonym")
        self.synonymLabelURIs.add("https://www.ncbi.nlm.nih.gov/taxonomy#common_name")

        # Alternative term
        self.synonymLabelURIs.add("http://purl.obolibrary.org/obo/IAO_0000118")
        # Mouse anatomy
        # Lexically rich interesting
        self.lexicalAnnotationURIs.update(self.mainLabelURIs)
        self.lexicalAnnotationURIs.update(self.synonymLabelURIs)
        self.lexicalAnnotationURIs.add("http://www.w3.org/2000/01/rdf-schema#comment")
        self.lexicalAnnotationURIs.add("http://www.geneontology.org/formats/oboInOwl#hasDbXref")
        self.lexicalAnnotationURIs.add("http://purl.org/dc/elements/1.1/description")
        self.lexicalAnnotationURIs.add("http://purl.org/dc/terms/description")
        self.lexicalAnnotationURIs.add("http://purl.org/dc/elements/1.1/title")
        self.lexicalAnnotationURIs.add("http://purl.org/dc/terms/title")

        # Definition
        self.lexicalAnnotationURIs.add("http://purl.obolibrary.org/obo/IAO_0000115")
        # Elucidation
        self.lexicalAnnotationURIs.add("http://purl.obolibrary.org/obo/IAO_0000600")
        # has associated axiomm fol
        self.lexicalAnnotationURIs.add("http://purl.obolibrary.org/obo/IAO_0000602")
        # has associated axiomm nl
        self.lexicalAnnotationURIs.add("http://purl.obolibrary.org/obo/IAO_0000601")
        self.lexicalAnnotationURIs.add("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace")

    def get_annotation_uris_for_preferred_labels(self) -> set:
        return self.mainLabelURIs

    def get_annotation_uris_for_synonyms(self) -> set:
        return self.synonymLabelURIs

    def get_annotation_uris_for_lexical_annotations(self) -> set:
        return self.lexicalAnnotationURIs


class OntologyAccess:
    def __init__(self, urionto: str, annotate_on_init: bool = True,
                 cache_dir: str | None = None) -> None:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
        info(f"Ontology instanciated from {urionto}.")
        self.urionto = str(urionto)
        # TODO: check this claim (I can't quite recall whether this was ths issue)
        # JD: each `OntologyAccess`` instance is assigned its own isolated 
        # World (using an independent in-memory SQLite quadstore; this 
        # prevents shared-state issues (i.e., global entity cache issues,
        # and importantly: rdflib store lock contention) that caused 
        # intermittent problems previously when multiple `OntologyAccess`
        # objects were created in the same process using a shared default_world.
        #
        # When cache_dir is provided, the World is backed by a
        # process-private copy of a persistent SQLite quadstore.
        # This skips the expensive OWL/RDF parse on subsequent runs.
        if cache_dir is not None:
            self.world = _get_cached_world(self.urionto, cache_dir)
        else:
            self.world = owlready2.World()
        if annotate_on_init:
            self.load_ontology()
            self.indexAnnotations()

    def get_ontology_iri(self) -> str:
        return self.urionto

    def load_ontology(self, reasoner: Reasoner = Reasoner.NONE, memory_java: str = "10240") -> None:
        step("Loading ontology.")
        self.onto:owlready2.Ontology = self.world.get_ontology(self.urionto).load()
        owlready2.JAVA_MEMORY = memory_java
        info(f"JVM mem: {memory_java}")
        # DH: If we set log level here, then the 2nd call we make
        # to this function, to load the 2nd (target) ontology, writes
        # a lot of '* Owlready2 * ...' messages to the console, which
        # we don't want.  The LogMap-LLM user gets a much better
        # experience if we do NOT set the owlready2 log level at all.
        # TODO: trigger with 'verbose flag'
        # owlready2.set_log_level(9)

        step("Ready for reasoning.")

        if reasoner == Reasoner.PELLET:
            try:
                with self.onto:  # it does add inferences to ontology
                    # Is this wrt data assertions? Check if necessary
                    # infer_property_values = True, infer_data_property_values = True
                    logging.info("Classifying ontology with Pellet...")
                    sync_reasoner_pellet(x=self.world)  # it does add inferences to ontology
                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are %d unsatisfiable classes.", unsat)
            except Exception as e:
                logging.error("Classifying with Pellet failed: %s", e)
                raise e

        elif reasoner == Reasoner.HERMIT:
            try:
                with self.onto:  # it does add inferences to ontology
                    logging.info("Classifying ontology with HermiT...")
                    sync_reasoner(x=self.world)  # HermiT doe snot work very well....
                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are %d unsatisfiable classes.", unsat)
            except owlready2.OwlReadyOntologyParsingError:
                logging.info("Classifying with HermiT failed.")

        self.graph = self.world.as_rdflib_graph()

        step("Preparing CLS indicies.")

        # Build IRI-keyed indices for O(1) lookups.
        # The previous linear-scan approach (iterating over all classes
        # per lookup) is prohibitively slow on large ontologies such as
        # SNOMED CT (~350k classes). With ~5000 mappings, that produced
        # ~1.75 billion string comparisons on the source side alone.
        self._uri_to_class = {cls.iri: cls for cls in self.onto.classes()}
        self._uri_to_property = {}
        for prop in self.onto.properties():
            self._uri_to_property[prop.iri] = prop
        self._uri_to_individual = {ind.iri: ind for ind in self.onto.individuals()}
        self._uri_to_entity = dict(self._uri_to_class)
        self._uri_to_entity.update(self._uri_to_property)
        self._uri_to_entity.update(self._uri_to_individual)

        # DH: This used to be a logging.info() statement. But for
        # LogMap-LLM we have promoted it to a print() statement, so
        # the user can always see how big the ontology is that is
        # being processed and prepared from prompt building.
        success(f"There are {len(self.graph)} triples in the ontology")
        success(f"Indexed {len(self._uri_to_class)} classes, "
                f"{len(self._uri_to_property)} properties, "
                f"{len(self._uri_to_individual)} individuals")

    def getOntology(self) -> owlready2.Ontology:
        return self.onto

    def getClassByURI(self, uri: str) -> owlready2.EntityClass:
        return self._uri_to_class.get(uri)

    def getClassByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.name.lower() == name.lower():
                return cls
        return None

    def getEntityByURI(self, uri: str) -> owlready2.EntityClass:
        return self._uri_to_entity.get(uri)

    def getEntityByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.name.lower() == name.lower():
                return cls
        for prop in list(self.getOntology().properties()):
            if prop.name.lower() == name.lower():
                return prop
        return None


    # property-specific lookups

    def getPropertyByURI(self, uri: str):
        return self._uri_to_property.get(uri)


    def isProperty(self, uri: str) -> bool:
        return uri in self._uri_to_property


    def getDomainClasses(self, prop) -> set:
        domain_classes = set()
        for cls in prop.domain:
            if isinstance(cls, owlready2.ThingClass):
                domain_classes.add(cls)
        return domain_classes


    def getRangeClasses(self, prop) -> set:
        """
        this method returns only named classes for object properties
        use getRangeNames() for a string representation that works for both obj+data prps
        """
        range_classes = set()
        for cls in prop.range:
            if isinstance(cls, owlready2.ThingClass):
                range_classes.add(cls)
        return range_classes


    def getDomainNames(self, prop) -> set[str]:
        """
        returns preferred label for the domain classes of a property
        falls back to class local names when no labels are indexed
        """
        names = set()
        for cls in self.getDomainClasses(prop):
            labels = self.getPreferredLabels(cls)
            if labels:
                names.update(labels)
            else:
                names.add(str(cls.name))
        return names


    def getRangeNames(self, prop) -> set[str]:
        """
        returns preferred labels for the range of a property
        object properties: returns labels of range classes
        data properties: returns datatype names
        """
        names = set()
        for item in prop.range:
            if isinstance(item, owlready2.ThingClass):
                labels = self.getPreferredLabels(item)
                if labels:
                    names.update(labels)
                else:
                    names.add(str(item.name))
            else:
                # datatype or other non-class range element
                names.add(str(getattr(item, 'name', str(item))))
        return names


    # individual-specific lookups (KG track):


    def getIndividualByURI(self, uri: str):
        return self._uri_to_individual.get(uri)


    def isIndividual(self, uri: str) -> bool:
        return uri in self._uri_to_individual


    def getLabelsForURI(self, uri: str) -> list[str]:
        from rdflib import URIRef, RDFS
        labels = []
        for _, _, obj in self.graph.triples((URIRef(uri), RDFS.label, None)):
            label_str = str(obj).strip()
            if label_str:
                labels.append(label_str)
        if not labels:
            # fall back to local name
            if '#' in uri:
                labels.append(uri.rsplit('#', 1)[-1])
            elif '/' in uri:
                labels.append(uri.rsplit('/', 1)[-1])
        return labels


    def getInstanceContext(self, uri: str) -> dict:
        """
        retrieves structured context for an individual via SPARQL

        queries the rdflib graph for all relevant triples about the
        individual and partitions them into types, labels, abstract,
        categories, data properties, and object properties.

        Parameters
        ----------
        uri : str
            The IRI of the individual.

        Returns
        -------
        dict with keys:
            uri : str
            labels : list[str]
            types : list[dict]  — [{"uri": str, "label": str}, ...]
            abstract : str | None
            categories : list[str]
            data_properties : list[dict]
            object_properties : list[dict]
        """
        from rdflib import URIRef, RDF, RDFS, Literal, Namespace

        DCT = Namespace("http://purl.org/dc/terms/")
        DBKWIK = Namespace("http://dbkwik.webdatacommons.org/ontology/")

        subject = URIRef(uri)

        # labels
        labels = self.getLabelsForURI(uri)

        # types (rdf:type)
        # filters out owl:NamedIndividual, owl:Thing, etc.
        # (types that the KG track excludes from evaluation)
        excluded_type_suffixes = {
            "NamedIndividual", "Thing",
            "http://dbkwik.webdatacommons.org/ontology/Image",
            "http://www.w3.org/2004/02/skos/core#Concept",
        }
        types = []
        for _, _, obj in self.graph.triples((subject, RDF.type, None)):
            type_uri = str(obj)
            # skip excluded types
            skip = False
            for excl in excluded_type_suffixes:
                if type_uri.endswith(excl) or type_uri == excl:
                    skip = True
                    break
            if skip:
                continue
            type_labels = self.getLabelsForURI(type_uri)
            types.append({
                "uri": type_uri,
                "label": type_labels[0] if type_labels else type_uri.rsplit('/', 1)[-1],
            })

        # abstract/short desc
        abstract = None
        for pred in [RDFS.comment, DBKWIK.abstract]:
            for _, _, obj in self.graph.triples((subject, pred, None)):
                if isinstance(obj, Literal) and str(obj).strip():
                    abstract = str(obj).strip()
                    break
            if abstract:
                break

        # categories (dct:subject)
        categories = []
        for _, _, obj in self.graph.triples((subject, DCT.subject, None)):
            cat_labels = self.getLabelsForURI(str(obj))
            if cat_labels:
                categories.extend(cat_labels)
            else:
                categories.append(str(obj).rsplit('/', 1)[-1])

        # all other triples (partition into data/obj props)
        # skip predicates we've already handled above ^^^
        handled_predicates = {
            str(RDF.type), str(RDFS.label), str(RDFS.comment),
            str(DCT.subject), str(DBKWIK.abstract),
        }
        data_properties = []
        object_properties = []

        for _, pred, obj in self.graph.triples((subject, None, None)):
            pred_uri = str(pred)
            if pred_uri in handled_predicates:
                continue

            pred_labels = self.getLabelsForURI(pred_uri)
            pred_label = pred_labels[0] if pred_labels else pred_uri.rsplit('/', 1)[-1]

            if isinstance(obj, Literal):
                # data property assertion
                datatype = str(obj.datatype) if obj.datatype else "string"
                # simplify common XSD datatype URIs
                if datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
                    datatype = datatype.rsplit('#', 1)[-1]
                data_properties.append({
                    "predicate_uri": pred_uri,
                    "predicate_label": pred_label,
                    "value": str(obj),
                    "datatype": datatype,
                })
            else:
                # object property assertion (URI object)
                obj_uri = str(obj)
                obj_labels = self.getLabelsForURI(obj_uri)
                object_properties.append({
                    "predicate_uri": pred_uri,
                    "predicate_label": pred_label,
                    "object_uri": obj_uri,
                    "object_label": obj_labels[0] if obj_labels else obj_uri.rsplit('/', 1)[-1],
                })

        return {
            "uri": uri,
            "labels": labels,
            "types": types,
            "abstract": abstract,
            "categories": categories,
            "data_properties": data_properties,
            "object_properties": object_properties,
        }


    def compute_predicate_entropies(self, uri_pattern="/property/") -> dict:
        """
        compute Shannon entropy of value distributions for matchable predicates

        for each predicate matching uri_pattern:
            compute the entropy of its value distribution across all subjects in the rdflib graph
            
        higher entropy means the predicate takes many distinct values relatively uniformly
        meaning its more discriminating for entity resolution

        Note we compute once per ontology and cache for reuse across all M_ask pairs

        Parameters
        ----------
        uri_pattern : str
            URI substring filter (only predicates whose URI contains this pattern are included, default: '/property/'.

        Returns
        -------
        dict mapping predicate_uri (str) -> entropy (float, in bits).
        """
        if hasattr(self, '_predicate_entropies'):
            return self._predicate_entropies

        from collections import Counter
        import math

        entropies = {}

        # get unique predicates via the graphs predicate index
        unique_predicates = set(self.graph.predicates(None, None))

        for pred in unique_predicates:
            pred_uri = str(pred)
            if uri_pattern not in pred_uri:
                continue

            # count distinct values for this predicate
            value_counts = Counter()
            for _, _, obj in self.graph.triples((None, pred, None)):
                value_counts[str(obj)] += 1

            total = sum(value_counts.values())
            if total <= 1:
                entropies[pred_uri] = 0.0
                continue

            entropy = 0.0
            for count in value_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)

            entropies[pred_uri] = entropy

        self._predicate_entropies = entropies
        return entropies


    def hasSubjectInGraph(self, uri: str) -> bool:
        """
        Check whether a URI appears as a subject in the rdflib graph
        
        this is a fallback for KG track instances that owlready2 does not enumerate via onto.individuals()
        i.e., if a URI appears as a subject in at least one triple, it is treated as a resolvable entity 
        even if owlready2 has no python object for it
        """
        from rdflib import URIRef
        # ask for a single triple — O(1) in rdflib's indexed store
        for _ in self.graph.triples((URIRef(uri), None, None)):
            return True
        return False


    ###
    # originl methods (for classes):
    ###

    def getClassObjectsContainingName(self, name: str) -> list[owlready2.EntityClass]:
        classes = []
        for cls in list(self.getOntology().classes()):
            if name.lower() in cls.name.lower():
                classes.append(cls)
        return classes

    def getClassIRIsContainingName(self, name: str) -> list[str]:
        classes = []
        for cls in list(self.getOntology().classes()):
            if name.lower() in cls.name.lower():
                classes.append(cls.iri)
        return classes

    def getAncestorsURIsMinusClass(self, cls: owlready2.EntityClass) -> set[str]:
        ancestors_str = self.getAncestorsURIs(cls)
        ancestors_str.remove(cls.iri)
        return ancestors_str

    def getAncestorsURIs(self, cls: owlready2.EntityClass) -> set[str]:
        ancestors_str = set()
        for anc_cls in cls.ancestors():
            ancestors_str.add(anc_cls.iri)
        return ancestors_str

    def getAncestorsNames(self, cls: owlready2.EntityClass) -> set[str]:
        ancestors_str = set()
        for anc_cls in cls.ancestors():
            ancestors_str.add(anc_cls.name)
        return ancestors_str

    def getAncestors(self, cls: owlready2.EntityClass, include_self: bool = True) -> set[owlready2.ThingClass]:
        ancestors_str = set()
        for anc_cls in cls.ancestors(include_self=include_self):
            ancestors_str.add(anc_cls)
        return ancestors_str

    def getDescendantURIs(self, cls: owlready2.EntityClass) -> set[str]:
        descendants_str = set()
        for desc_cls in cls.descendants():
            descendants_str.add(desc_cls.iri)
        return descendants_str

    def getDescendantNames(self, cls: owlready2.EntityClass) -> set[str]:
        descendants_str = set()
        for desc_cls in cls.descendants():
            descendants_str.add(desc_cls.name)
        return descendants_str

    def getDescendants(self, cls: owlready2.EntityClass, include_self: bool = True) -> set[owlready2.ThingClass]:
        descendants_str = set()
        for desc_cls in cls.descendants(include_self=include_self):
            descendants_str.add(desc_cls)
        return descendants_str

    def getDescendantNamesForClassName(self, cls_name: str) -> set[str]:
        cls = self.getClassByName(cls_name)
        descendants_str = set()
        for desc_cls in cls.descendants():
            descendants_str.add(desc_cls.name)
        return descendants_str

    def isSubClassOf(self, sub_cls1: owlready2.EntityClass, sup_cls2: owlready2.EntityClass) -> bool:
        return sup_cls2 in sub_cls1.ancestors()

    def isSuperClassOf(self, sup_cls1: owlready2.EntityClass, sub_cls2: owlready2.EntityClass) -> bool:
        return sup_cls1 in sub_cls2.ancestors()

    def getDomainURIs(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        domain_uris = set()
        for cls in prop.domain:
            with contextlib.suppress(AttributeError):
                domain_uris.add(cls.iri)
        return domain_uris

    def getDatatypeRangeNames(self, prop: owlready2.DataPropertyClass) -> set[str]:
        range_uris = set()
        for cls in prop.range:
            range_uris.add(cls.name)
        return range_uris

    # Only for object properties
    def getRangeURIs(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        range_uris = set()
        for cls in prop.range:
            with contextlib.suppress(AttributeError):
                range_uris.add(cls.iri)
        return range_uris

    def getInverses(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        inv_uris = set()
        for p in prop.inverse:
            inv_uris.add(p.iri)
        return inv_uris

    def getClasses(self) -> set[owlready2.EntityClass]:
        return self.getOntology().classes()

    def getDataProperties(self) -> set[owlready2.DataPropertyClass]:
        return self.getOntology().data_properties()

    def getObjectProperties(self) -> set[owlready2.ObjectPropertyClass]:
        return self.getOntology().object_properties()

    def getIndividuals(self) -> set[owlready2.NamedIndividual]:
        return self.getOntology().individuals()

    # DH: By removing rdflib.Graph as return type hint, we remove the entire
    # dependency on rdflib.  So, above, we can comment-out 'import rdflib'
    #def getGraph(self) -> rdflib.Graph:  
    def getGraph(self):
        return self.graph

    def queryGraph(self, query: str) -> list:
        results = self.graph.query(query)
        return list(results)

    def getQueryForAnnotations(self, ann_prop_uri: str) -> str:
        return f"""SELECT DISTINCT ?s ?o WHERE {{
        {{
        ?s <{ann_prop_uri}> ?o .
        }}
        UNION
        {{
        ?s <{ann_prop_uri}> ?i .
        ?i <http://www.w3.org/2000/01/rdf-schema#label> ?o .
        }}
        }}"""

    def indexAnnotations(self) -> None:
        annotation_uris = AnnotationURIs()
        self.entityToSynonyms = {}
        self.allEntityAnnotations = {}
        self.preferredLabels = {}
        info("Populating annotation index...")

        synonym_uris = annotation_uris.get_annotation_uris_for_synonyms()
        preferred_uris = annotation_uris.get_annotation_uris_for_preferred_labels()
        lexical_uris = annotation_uris.get_annotation_uris_for_lexical_annotations()

        # lexical_uris is a superset of both synonym_uris and preferred_uris
        #
        # we query each annotation property once via direct rdflib triple lookups 
        # (bypassing per-URI SPARQL query parsing overhead)
        # then distribute each result to the appropriate subset dicts
        #
        # this replaces the previous three-call pattern that ran ~58
        # individual SPARQL queries (~21 synonym + ~33 lexical + ~4 preferred)
        # many of which redundantly queried the same URIs

        self._populateAllAnnotationDicts(lexical_uris, synonym_uris, preferred_uris)


    def _populateAllAnnotationDicts(self, all_uris: set, synonym_uris: set, preferred_uris: set) -> None:
        """
        populate all three annotation dictionaries in a single pass

        iterates over the superset of annotation URIs once using direct rdflib triple-pattern lookups 
        distributes each result to the appropriate subset dictionaries based on set membership

        for each annotation property URI both direct annotations (where the object is a literal) and indirect 
        annotations (where the object is an intermediate node whose rdfs:label provides the text) are resolved

        only annotations with language tag ``en`` or ``None`` (untagged) are included, matching the original per-URI SPARQL query

        Parameters
        ----------
        all_uris : set
            The superset of all annotation URIs (lexicalAnnotationURIs)
        synonym_uris : set
            Annotation URIs that should also populate entityToSynonyms
        preferred_uris : set
            Annotation URIs that should also populate preferredLabels
        """
        from rdflib import URIRef, Literal

        rdfs_label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")

        for ann_uri_str in all_uris:
            ann_ref = URIRef(ann_uri_str)
            in_synonyms = ann_uri_str in synonym_uris
            in_preferred = ann_uri_str in preferred_uris

            for s, _, o in self.graph.triples((None, ann_ref, None)):
                subj_str = str(s)

                if isinstance(o, Literal):
                    # direct annotation — apply language filter
                    if o.language is not None and o.language != "en":
                        continue
                    self._distribute_annotation(subj_str, o.value, in_synonyms, in_preferred)
                else:
                    # indirect: annotation points to an intermediate node;
                    # follow to rdfs:label for the actual annotation text
                    for _, _, label in self.graph.triples((o, rdfs_label, None)):
                        if not isinstance(label, Literal):
                            continue
                        if label.language is not None and label.language != "en":
                            continue
                        self._distribute_annotation(subj_str, label.value, in_synonyms, in_preferred)


    def _distribute_annotation(self, subj: str, value, in_synonyms: bool, in_preferred: bool) -> None:
        """
        add an annotation value to the appropriate dictionaries
        - always adds to allEntityAnnotations (the lexical superset)
        - conditionally adds to entityToSynonyms and preferredLabels based on the annotation property category membership
        """
        # always add to allEntityAnnotations (lexical is the superset)
        if subj not in self.allEntityAnnotations:
            self.allEntityAnnotations[subj] = set()
        self.allEntityAnnotations[subj].add(value)

        if in_synonyms:
            if subj not in self.entityToSynonyms:
                self.entityToSynonyms[subj] = set()
            self.entityToSynonyms[subj].add(value)

        if in_preferred:
            if subj not in self.preferredLabels:
                self.preferredLabels[subj] = set()
            self.preferredLabels[subj].add(value)


    def populateAnnotationDicts(self, annotation_uris: set, dictionary: dict) -> None:
        """Populate the given dictionary with annotations from the provided URIs.

        This method queries a graph for annotations based on the provided URIs and
        populates the given dictionary with the results. Only annotations with
        language set to English or None are added to the dictionary.

        note: retained for backward compatibility
        the internal call path in indexAnnotations() now uses _populateAllAnnotationDicts()
        which is faster due to single-pass iteration and direct rdflib triple lookups
                
        Args:
            annotation_uris (list): A list of annotation property URIs to query.
            dictionary (dict): A dictionary to populate with the query results.
                               The keys are the string representations of the
                               annotation subjects, and the values are sets of
                               annotation values.

        Returns:
            None

        """
        for ann_prop_uri in annotation_uris:
            results = self.queryGraph(self.getQueryForAnnotations(ann_prop_uri))
            for row in results:
                try:
                    if row[1].language == "en" or row[1].language is None:
                        if str(row[0]) not in dictionary:
                            dictionary[str(row[0])] = set()
                        dictionary[str(row[0])].add(row[1].value)
                except AttributeError:
                    pass
        return

    def getSynonymsNames(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.entityToSynonyms:
            return set()
        return self.entityToSynonyms[entity.iri]

    def getAnnotationNames(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.allEntityAnnotations:
            return set()
        return self.allEntityAnnotations[entity.iri]

    def getPreferredLabels(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.preferredLabels:
            return set()
        return self.preferredLabels[entity.iri]
