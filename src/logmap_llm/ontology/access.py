"""
logmap_llm.ontology.access
ontology access via owlready2 (ess. OBDA layer, 'object.py' -> ORM)
each OntologyAccess instance creates its own owlready2.World 
which is an isolated SQLite quadstore (prevents shared-state issues)
"""
from __future__ import annotations

import owlready2
import contextlib
import logging
import rdflib
import numpy as np
import json


from json import JSONDecodeError
from pathlib import Path

from enum import Enum

from collections import Counter, defaultdict
from owlready2 import sync_reasoner, sync_reasoner_pellet
from rdflib import URIRef, RDF, RDFS, Literal

# DH: We have removed the dependency on rdflib. It was only needed to
# satisfy a return-type hint on method getGraph(). We removed the hint,
# and so we comment-out the import of rdflib.  Eventually we will remove
# this commented-out import altogether.
# import rdflib 

# ^ JD. many of the new prompts depend on this. However we
#   probably (eventually) swicth this out for accessing directly through
#   LogMap, if I understand correctly (it just requires implementing
#   similar java functions in the LogMap code prior to building)

from logmap_llm.constants import VERBOSE, VERY_VERBOSE
from logmap_llm.utils.logging import debug, info, success, step
from logmap_llm.utils.io import compute_entropy_disk_cache_path
from logmap_llm.utils.io import atomic_json_write
from logmap_llm.constants import (
    DEFAULT_ENTROPY_CACHE_DIR,
    CANONICAL_RDFS_LABEL_URI_REF_STR,
)
from logmap_llm.ontology.vocabularies import (
    OntologyConventionVocabulary,
    DEFAULT_VOCAB,
)
from logmap_llm.ontology.cache import OntologyCache



'''
owlready2 provides the following types of restrictions
(they have the same names than in Protégé):

some     : Property.some(Range_Class)
only     : Property.only(Range_Class)
min      : Property.min(cardinality, Range_Class)
max      : Property.max(cardinality, Range_Class)
exactly  : Property.exactly(cardinality, Range_Class)
value    : Property.value(Range_Individual / Literal value)
has_self : Property.has_self(Boolean value)

See: https://owlready2.readthedocs.io/en/latest/restriction.html

As such, we define a restriction map for resolving each to a string for prompt-injection.
'''

_OWLREADY2_RESTRICTION_TYPE_MAP = {
    owlready2.SOME     : "some",
    owlready2.ONLY     : "only",
    owlready2.MIN      : "min",
    owlready2.MAX      : "max",
    owlready2.EXACTLY  : "exactly",
    owlready2.VALUE    : "value",
    owlready2.HAS_SELF : "has_self",
}

###
# PRIVATE RESTRICTION STRING LITERALS
###

_HAS_SELF_FILLER_REFLEXIVE = "reflexive"
_HAS_SELF_FILLER_NON_REFLEXIVE = "non_reflexive"



###
# REASONING
###

class Reasoner(Enum):
    HERMIT = 0      # Not really adding the right set of entailments
    PELLET = 1      # Slow for large ontologies
    STRUCTURAL = 2  # Basic domain/range propagation
    NONE = 3        # No reasoning



###
# ANNOTATION URIs
###

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

        # Lexical annotations (superset)
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



###
# HELPERS: ORIGINALLY IN 'oracle_prompt_building.py'
#   (ontology loading functions)
###

def load_xs_ontologies(ontology_file_paths: list[str], eager_annotation=True, cache_dir: Path | str | None = None, vocabulary: OntologyConventionVocabulary = DEFAULT_VOCAB) -> list[OntologyAccess]:
    
    cache:OntologyCache | None = OntologyCache(cache_dir=str(cache_dir)) if cache_dir is not None else None

    xs_loaded_ontologies:list[OntologyAccess] = []

    for onto_idx, ontology_fp in enumerate(ontology_file_paths):
        info(f"Loading ontology {str(onto_idx)} from {str(ontology_fp)}")
        xs_loaded_ontologies.append(
            OntologyAccess(
                urionto=ontology_fp, 
                annotate_on_init=eager_annotation, 
                cache=cache,
                vocabulary=vocabulary,
            )
        )
        success(f"Loaded ontology: {xs_loaded_ontologies[onto_idx].get_ontology_iri()}")

    success(f"Finished loading ontologies.")
    return xs_loaded_ontologies


def load_ontologies(onto_src_filepath, onto_tgt_filepath, cache_dir=None, vocabulary: OntologyConventionVocabulary = DEFAULT_VOCAB) -> tuple[OntologyAccess, OntologyAccess]:

    print()

    cache = OntologyCache(cache_dir=cache_dir) if cache_dir is not None else None

    step(f"[STEP 2] Loading the first ontology from {str(onto_src_filepath)}")
    OA_source_onto = OntologyAccess(
        urionto=onto_src_filepath, 
        annotate_on_init=True, 
        cache=cache,
        vocabulary=vocabulary,
    )
    success(f"Loaded ontology: {OA_source_onto.get_ontology_iri()}")

    print()

    step(f"[STEP 2] Loading the second ontology from {str(onto_tgt_filepath)}")
    OA_target_onto = OntologyAccess(
        urionto=onto_tgt_filepath,
        cache=cache,
        vocabulary=vocabulary,
    )
    success(f"Loaded ontology: {OA_target_onto.get_ontology_iri()}")

    print()

    return OA_source_onto, OA_target_onto

###
# END: HELPERS
###



class OntologyAccess:
    """
    Access and query an OWL ontology via owlready2 + rdflib.
    Each instance uses its own owlready2.World (isolated SQLite quadstore)
    This prevents shared-state issues when multiple ontologies are loaded.
    """
    def __init__(self, urionto: str, annotate_on_init: bool = True, cache=None, vocabulary: OntologyConventionVocabulary = DEFAULT_VOCAB) -> None:
        """
        urionto : str - URI or file path to the ontology.
            The ontology this object provides access to (think OntologyAccess for OBDA - ORM-analagous; OntologyObject for entities - Record-analagous).
        annotate_on_init : bool - if true, load the ontology and index annotations immediately.
            Most functionality for access is index-based.
        cache : object, optional - caching interface (see logmap_llm.ontology.cache); if None, then a cache is not used.
            Note the default cache location: "~/.cache/logmap-llm/owlready2/*" for the owlready2 cache (other caches in ~/.cache/logmap-llm).
            see: cache-based constants in constants.py
        vocabulary : OntologyConventionVocabulary — describes this ontology's annotation and URI-structure conventions. 
            Defaults to DEFAULT_VOCAB (general RDF). Typically set per-task by the loader from config.
        """
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
        self.urionto = str(urionto)
        self._cache = cache
        self._vocabulary = vocabulary

        # JD. each instance gets its own isolated World; but when a cache is provided, the world is backed by a 
        # process-private copy (a shared on-disk quadstore) so that the source ontology does not need to be 
        # re-parsed on every run. Without a cache, fall back to a fresh empty world (parses from source in load_ontology)

        if self._cache is not None:
            self.world = self._cache.get_cached_world(self.urionto)
        else:
            self.world = owlready2.World()

        if annotate_on_init:
            self.load_ontology()
            self.indexAnnotations()


    def get_ontology_iri(self) -> str:
        return self.urionto


    def load_ontology(self, reasoner: Reasoner = Reasoner.NONE, memory_java: str = "10240") -> None:
        # when the world was constructed from a cached quadstore, the triples are already present 
        # and we just need to bind the ontology object; otherwise parse from source
        if self._cache is not None:
            self.onto = self.world.get_ontology(self.urionto)
        else:
            self.onto = self.world.get_ontology(self.urionto).load()
        
        owlready2.JAVA_MEMORY = memory_java

        # DH: If we set log level here, then the 2nd call we make
        # to this function, to load the 2nd (target) ontology, writes
        # a lot of '* Owlready2 * ...' messages to the console, which
        # we don't want.  The LogMap-LLM user gets a much better
        # experience if we do NOT set the owlready2 log level at all.
        # We can set it to 0 before calling get_ontology() to undo
        # the effect of the 9 here (i.e. to shut logging off again)
        # but we get the same effect if we simply do NOT set the
        # level here.
        #owlready2.set_log_level(9)

        if reasoner == Reasoner.PELLET:
            try:
                with self.onto: # it does add inferences to ontology
                    # Is this wrt data assertions? Check if necessary
                    # infer_property_values = True, infer_data_property_values = True
                    logging.info("Classifying ontology with Pellet...")
                    sync_reasoner_pellet(x=self.world) # it does add inferences to ontology
                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are %d unsatisfiable classes.", unsat)
            except Exception as e:
                logging.error("Classifying with Pellet failed: %s", e)
                raise e

        elif reasoner == Reasoner.HERMIT:
            try:
                with self.onto: # it does add inferences to ontology
                    logging.info("Classifying ontology with HermiT...")
                    sync_reasoner(x=self.world) # HermiT doe snot work very well....
                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are %d unsatisfiable classes.", unsat)
            except owlready2.OwlReadyOntologyParsingError:
                logging.info("Classifying with HermiT failed.")

        self.graph = self.world.as_rdflib_graph()
        # DH: This used to be a logging.info() statement. But for
        # LogMap-LLM we have promoted it to a print() statement, so
        # the user can always see how big the ontology is that is
        # being processed and prepared from prompt building.
        info(f"There are {len(self.graph)} triples in the ontology", important=True)

        # O(1) URI lookup indices
        self._uri_to_class = {cls.iri: cls for cls in self.onto.classes()}
        self._uri_to_entity = dict(self._uri_to_class)
        self._uri_to_property = {}
        
        for prop in self.onto.properties():
            self._uri_to_entity[prop.iri] = prop
            self._uri_to_property[prop.iri] = prop
        
        self._uri_to_individual = {
            ind.iri: ind for ind in self.onto.individuals()
            if hasattr(ind, 'iri') and ind.iri
        }

        info(f"Indexed {len(self._uri_to_class)} classes.")
        info(f"Indexed {len(self._uri_to_property)} properties.")
        info(f"Indexed {len(self._uri_to_individual)} individuals.")

    ###
    # ONTOLOGY + GRAPH-BASED METHODS
    ###

    def getOntology(self) -> owlready2.Ontology:
        return self.onto


    def getGraph(self) -> rdflib.Graph:
        return self.graph


    def queryGraph(self, query: str) -> list:
        return list(self.graph.query(query))


    ###
    # CLS-BASED LOOK-UP METHODS
    ###


    def getClassByURI(self, uri: str) -> owlready2.EntityClass:
        return self._uri_to_class.get(uri)


    def getClassByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getClasses()):
            if cls.name.lower() == name.lower():
                return cls
        return None


    def getEntityByURI(self, uri: str) -> owlready2.EntityClass:
        return self._uri_to_entity.get(uri)


    def getEntityByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getClasses()):
            if cls.name.lower() == name.lower():
                return cls
        for prop in list(self.getOntology().properties()):
            if prop.name.lower() == name.lower():
                return prop
        return None


    def getClassObjectsContainingName(self, name: str) -> list[owlready2.EntityClass]:
        classes = []
        for cls in list(self.getClasses()):
            if name.lower() in cls.name.lower():
                classes.append(cls)
        return classes
    
    
    def getClassIRIsContainingName(self, name: str) -> list[str]:
        classes = []
        for cls in list(self.getClasses()):
            if name.lower() in cls.name.lower():
                classes.append(cls.iri)
        return classes


    def getAncestorsURIsMinusClass(self, cls: owlready2.EntityClass) -> set[str]:
        ancestors_str = self.getAncestorsURIs(cls)
        ancestors_str.discard(cls.iri)
        return ancestors_str


    def getAncestorsURIs(self, cls: owlready2.EntityClass) -> set[str]:
        return {anc_cls.iri for anc_cls in cls.ancestors()}


    def getAncestorsNames(self, cls: owlready2.EntityClass) -> set[str]:
        return {anc_cls.name for anc_cls in cls.ancestors()}


    def getAncestors(self, cls: owlready2.EntityClass, include_self: bool = True) -> set[owlready2.ThingClass]:
        return {anc_cls for anc_cls in cls.ancestors(include_self=include_self)}


    def getDescendantURIs(self, cls: owlready2.EntityClass) -> set[str]:
        return {desc_cls.iri for desc_cls in cls.descendants()}


    def getDescendantNames(self, cls: owlready2.EntityClass) -> set[str]:
        return {desc_cls.name for desc_cls in cls.descendants()}


    def getDescendants(self, cls: owlready2.EntityClass, include_self: bool = True) -> set[owlready2.ThingClass]:
        return {desc_cls for desc_cls in cls.descendants(include_self=include_self)}

    
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


    def getClasses(self) -> set[owlready2.EntityClass]:
        return self.getOntology().classes()


    ###
    # PROPERTY-BASED LOOK-UP METHODS
    ###


    def getObjectProperties(self) -> set[owlready2.ObjectPropertyClass]:
        return self.getOntology().object_properties()


    def getDataProperties(self) -> set[owlready2.DataPropertyClass]:
        return self.getOntology().data_properties()


    def getPropertyByURI(self, uri: str):
        """Look up a property by IRI.  Returns None if not a property."""
        return self._uri_to_property.get(uri)


    def isProperty(self, uri: str) -> bool:
        """Check whether a URI resolves to a property in this ontology."""
        return uri in self._uri_to_property


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


    def getRangeURIs(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        range_uris = set()
        for cls in prop.range:
            with contextlib.suppress(AttributeError):
                range_uris.add(cls.iri)
        return range_uris


    def getDomainNames(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        """Return preferred label names of the domain classes of a property."""
        pref_labels_for_domain = set()
        try:
            for cls in prop.domain:
                if hasattr(cls, 'iri'):
                    labels = self.getPreferredLabels(cls)
                    if labels:
                        pref_labels_for_domain.update(labels)
                    else:
                        pref_labels_for_domain.add(str(cls.name))
        except Exception:
            if VERBOSE:
                debug("(access.py - OntologyAccess::getDomainNames @raises)")
            pass
        return pref_labels_for_domain


    def getRangeNames(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        """Return preferred label names of the range classes of a property."""
        pref_labels_for_range = set()
        try:
            for cls in prop.range:
                if hasattr(cls, 'iri'):
                    labels = self.getPreferredLabels(cls)
                    if labels:
                        pref_labels_for_range.update(labels)
                    else:
                        pref_labels_for_range.add(str(cls.name))
        except Exception:
            if VERBOSE:
                debug("(access.py - OntologyAccess::getRangeNames @raises)")
            pass
        return pref_labels_for_range


    def getInverses(self, prop: owlready2.ObjectPropertyClass) -> set[str]:
        inv_uris = set()
        for p in prop.inverse:
            inv_uris.add(p.iri)
        return inv_uris


    def getInverseName(self, prop: owlready2.ObjectPropertyClass) -> str | None:
        """Return the label of the inverse property, or None."""
        inverse_prop = getattr(prop, 'inverse_property', None)
        if inverse_prop is None:
            return None
        labels = self.getPreferredLabels(inverse_prop)
        return next(iter(labels), None) if labels else str(inverse_prop.name)
    

    def getDeterministicInverseName(self, prop: owlready2.ObjectPropertyClass) -> str | None:
        """Return the label of the inverse property, or None."""
        inverse_prop = getattr(prop, 'inverse_property', None)
        if inverse_prop is None:
            return None
        labels = self.getPreferredLabels(inverse_prop)
        return min(labels) if labels else str(inverse_prop.name)



    ###
    # individual-specific lookups (KG track)
    ###


    def getIndividuals(self) -> set[owlready2.NamedIndividual]:
        return self.getOntology().individuals()


    def getIndividualByURI(self, uri: str) -> set[owlready2.NamedIndividual]:
        """Look up an individual by IRI.  Returns None if not an individual."""
        return self._uri_to_individual.get(uri)


    def isIndividual(self, uri: str) -> bool:
        """Check whether a URI resolves to an individual in this ontology."""
        return uri in self._uri_to_individual


    def getNonDeterministicLabelsForURI(self, uri: str) -> list[str]:
        """Retrieve rdfs:label values for any URI via the rdflib graph."""
        labels = []
        for _s, _p, obj in self.graph.triples((URIRef(uri), RDFS.label, None)):
            label_str = str(obj).strip()
            if label_str:
                labels.append(label_str)
        if not labels:
            if '#' in uri:
                labels.append(uri.rsplit('#', 1)[-1])
            elif '/' in uri:
                labels.append(uri.rsplit('/', 1)[-1])
        return labels
    

    def getLabelsForURI(self, uri: str) -> list[str]:
        """
        Retrieve rdfs:label values for any URI via the rdflib graph.
        Returns labels sorted lexicographically so that downstream [0]
        indexing is deterministic across runs. rdflib triple iteration
        order is not stable across processes.
        """
        labels = []
        for _s, _p, obj in self.graph.triples((URIRef(uri), RDFS.label, None)):
            label_str = str(obj).strip()
            if label_str:
                labels.append(label_str)
        if not labels:
            if '#' in uri:
                labels.append(uri.rsplit('#', 1)[-1])
            elif '/' in uri:
                labels.append(uri.rsplit('/', 1)[-1])
        return sorted(set(labels))



    ###
    # INDEXING:
    ###



    def indexAnnotations(self) -> None:
        """
        'Lexical URIs' \supseteq ('Synonym URIs' UNION 'Preferred (label) URIs'):

            lexical_uris is a superset of both synonym_uris and preferred_uris.

        We query each annotation property once via direct rdflib triple lookups.
        This bypasses per-URI SPARQL query parsing overhead; then distribute each 
        result to the appropriate subset dicts. This method acts as a replacement
        for the previous 'three-call pattern': 
            
            indexAnnotationDict(x) forall x in (syns, lexical annotations, pref labels)

        Previously ~58 individual SPARQL queries (~21 syn + ~33 lex + ~4 pref); many of 
        these queried the same URIs.
        """
        annotation_uris = AnnotationURIs()
        self.entityToSynonyms = {}
        self.allEntityAnnotations = {}
        self.preferredLabels = {}
        synonym_uris = annotation_uris.get_annotation_uris_for_synonyms()
        preferred_uris = annotation_uris.get_annotation_uris_for_preferred_labels()
        lexical_uris = annotation_uris.get_annotation_uris_for_lexical_annotations()
        self._populateAllAnnotationDicts(lexical_uris, synonym_uris, preferred_uris)



    def _populateAllAnnotationDicts(self, all_uris: set, synonym_uris: set, preferred_uris: set) -> None:
        """
        Populate the given dictionary with annotations from the provided URIs 
        (populates all three annotation dictionaries in a single pass)
        By default, only annotations with language set to English or 
        None are added to the dictionary (you can modify this by using the
        multilingual SKO vocabulary or by defining a custom vocabulary).
            (see logmap_llm.ontology.vocabulary -- ie. vocabulary.py)

        Iterates over the superset of annotation URIs (lexicalAnnotationURIs)
        once using direct rdflib triple-pattern lookups and distributes each
        result to the appropriate subset dictionaries based on set membership.
        
        IMPORTANT: For each annotation property URI, both direct annotations 
        (where the object is a Literal) and indirect annotations (where the object 
        is an intermediate node whose rdfs:label provides the text) are resolved.

        Only annotations with language tag 'en' or 'None' (untagged) are
        included, matching the original per-URI SPARQL query behaviour.

        NOTE: check that DH (which I believe is multilinguial) runs as expected
        with the vocabulary integration and fix (see below).
        """
        rdfs_label = URIRef(CANONICAL_RDFS_LABEL_URI_REF_STR)
        
        for annotation_uri_str in all_uris:
            annotation_uri_ref = URIRef(annotation_uri_str)
            in_synonyms = annotation_uri_str in synonym_uris
            in_preferred = annotation_uri_str in preferred_uris
            
            # fetch (subject, objects) for all (subject, annotation_uri_ref, object) triples
            for sub, _pred, obj in self.graph.triples((None, annotation_uri_ref, None)):
                subject_str = str(sub)

                if isinstance(obj, Literal):
                    # this is a direct annotation (apply language filter) 
                    if not self._vocabulary.accepts_language(obj.language):
                        continue # language not (defined or) accepted by the vocabulary (eg. !@en for default)
                    self._distribute_annotation(subject_str, obj.value, in_synonyms, in_preferred)
                
                else: # this is an 'indirect anotation' (ie. an intermediate node; follow to rdfs:label)
                    for _sub, _pred, label in self.graph.triples((obj, rdfs_label, None)):
                        if not isinstance(label, Literal):
                            if VERBOSE:
                                debug(f"(access.py) OntologyAccess::_populateAllAnnotationDict ... (sub str: {subject_str})")
                                debug(f"(access.py) ... encountered an indirect RDFS label which is not a literal.")
                                debug(f"(access.py) ... we might add some code here, something like: try skos:prefLabel (?)")
                            continue # "we're not the objects you're looking for" -- seems odd
                        if not self._vocabulary.accepts_language(label.language):
                            continue # language not accepted by the registered vocabulary
                        self._distribute_annotation(subject_str, label.value, in_synonyms, in_preferred)
                
                # END: if-else

            # END: for (s,p,o) in G.tiples(*,annotation_uri_ref,*)

        # END: for annotation URI (string) in all URIs

        if VERBOSE and VERY_VERBOSE:
            debug("(access.py) OntologyAccess::_populateAllAnnotationDicts [COMPLETED]")



    def _distribute_annotation(self, subj: str, value, in_synonyms: bool, in_preferred: bool) -> None:
        """
        Adds an annotation value to the appropriate dictionaries.
        Always adds to 'allEntityAnnotations' (the lexical superset).
        Conditionally adds to 'entityToSynonyms' and 'preferredLabels'
        which is based on the annotation propertys category membership
        """
        self.allEntityAnnotations.setdefault(subj, set()).add(value)
        if in_synonyms:
            self.entityToSynonyms.setdefault(subj, set()).add(value)
        if in_preferred:
            self.preferredLabels.setdefault(subj, set()).add(value)



    def getSynonymsNames(self, entity) -> set[str]:
        if entity.iri not in self.entityToSynonyms:
            return set()
        return self.entityToSynonyms[entity.iri]



    def getAnnotationNames(self, entity) -> set[str]:
        if entity.iri not in self.allEntityAnnotations:
            return set()
        return self.allEntityAnnotations[entity.iri]



    def getPreferredLabels(self, entity) -> set[str]:
        if entity.iri not in self.preferredLabels:
            return set()
        return self.preferredLabels[entity.iri]


    # backward-compatible alias
    getPrefferedLabels = getPreferredLabels



    ###
    # CUSTOM ACCESSOR METHODS
    # FOR (SEMI) SOPHISTICATED PROMPT CONSTRUCTION
    ###



    def getInstanceContext(self, uri: str, vocabulary: OntologyConventionVocabulary | None = None, deterministic: bool = True) -> dict:
        """
        Retrieve structured context for an individual via rdflib.
        Returns dict with keys: uri, labels, types, abstract, categories, data_properties, object_properties.
        The labels and types lookups use RDF-general conventions by default (if vocabulary=None). However, 
        for KG evalution on eg. DBpedia type KGs, you'll want to set `ontology_vocabulary="dbpedia_family"`
        in the config.toml. Additionally, for usage on DH (still untested -- TODO) use `ontology_vocabulary=
        "multilingual_skos"`.

        Predicates listed in vocabulary.abstract_predicates, vocabulary.category_predicates, or 
        vocabulary.extra_handled_predicates are excluded from the data_properties / object_properties sweep, 
        so information surfaced under one of the dedicated keys is not also duplicated as a generic property.
        """

        vocab = vocabulary if vocabulary is not None else self._vocabulary

        subject = URIRef(uri)
        labels = self.getLabelsForURI(uri)
        
        types: list[dict] = []
        for _sub, _pred, obj in self.graph.triples((subject, RDF.type, None)):
            type_uri = str(obj)
            type_labels = self.getLabelsForURI(type_uri)
            type_label = (
                type_labels[0] if type_labels
                else type_uri.rsplit('/', 1)[-1]
            )
            types.append({"uri": type_uri, "label": type_label})
        
        if deterministic: # sort by URI (stable det)
            types.sort(key=lambda x: x["uri"])

        # abstracts (TODO: are quite specific to a 
        # particuar family of KGs; should probably 
        # be handled under other predicates ... )
        abstract: str | None = None
        for pred_uri_str in vocab.abstract_predicates:
            pred_ref = URIRef(pred_uri_str)
            for _sub, _pred, obj in self.graph.triples((subject, pred_ref, None)):
                abstract = str(obj)
                break
            if abstract is not None:
                break

        categories: list[str] = []
        for pred_uri_str in vocab.category_predicates:
            pred_ref = URIRef(pred_uri_str)
            for _sub, _pred, obj in self.graph.triples((subject, pred_ref, None)):
                cat_labels = self.getLabelsForURI(str(obj))
                categories.extend(
                    cat_labels if cat_labels
                    else [str(obj).rsplit('/', 1)[-1]]
                )

        handled_predicates: set[str] = {
            str(RDF.type),
            str(RDFS.label),
            *vocab.abstract_predicates,
            *vocab.category_predicates,
            *vocab.extra_handled_predicates,
        }

        data_properties: list[dict] = []
        object_properties: list[dict] = []

        for _sub, pred, obj in self.graph.triples((subject, None, None)):
            pred_uri = str(pred)
            if pred_uri in handled_predicates:
                continue

            pred_labels = self.getLabelsForURI(pred_uri)
            pred_label = (
                pred_labels[0] if pred_labels
                else pred_uri.rsplit('/', 1)[-1]
            )

            if isinstance(obj, Literal):
                datatype = str(obj.datatype) if obj.datatype else "string"
                if datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
                    datatype = datatype.rsplit('#', 1)[-1]
                
                data_properties.append({
                    "predicate_uri": pred_uri,
                    "predicate_label": pred_label,
                    "value": str(obj),
                    "datatype": datatype,
                })    
                continue # reloop
            
            # else: we're delaing with an object property
            obj_uri = str(obj)
            obj_labels = self.getLabelsForURI(obj_uri)
            object_properties.append({
                "predicate_uri": pred_uri,
                "predicate_label": pred_label,
                "object_uri": obj_uri,
                "object_label": (
                    obj_labels[0] if obj_labels
                    else obj_uri.rsplit('/', 1)[-1]
                ),
            })

        # ensures det x runs
        if deterministic:
            data_properties.sort(
                key=lambda x: (
                    x["predicate_uri"], str(x.get("value", ""))
                )
            )
            object_properties.sort(
                key=lambda x: (
                    x["predicate_uri"], x.get("object_uri", "")
                )
            )

        # finally:
        return {
            "uri": uri,
            "labels": labels,
            "types": types,
            "abstract": abstract,
            "categories": categories,
            "data_properties": data_properties,
            "object_properties": object_properties,
        }

    

    def compute_predicate_entropies(self, uri_pattern: str | None = None, *, use_disk_cache: bool = True) -> dict[str, float]:
        """
        Compute Shannon entropy of value distributions for matchable predicates
        higher entropy -> the predicate is more discriminating for entity resolution
        """
        if uri_pattern is None:
            uri_pattern = self._vocabulary.property_uri_substring

        # IN MEMORY CACHE (lives on the 'access' -- connection -- object)
        # so we don't have to recompute the entropy for each prompt in G
        if not hasattr(self, "_entropy_cache"):
            self._entropy_cache: dict[str, dict[str, float]] = {}
        cached = self._entropy_cache.get(uri_pattern)
        if cached is not None:
            return cached

        # ensure the variable is scoped:
        on_disk_ent_cache_path = None

        # ON DISK CACHE:
        # (similar to the owlready2 cache mechanism)
        if use_disk_cache:
            on_disk_ent_cache_path = compute_entropy_disk_cache_path(
                local_onto_fp=self.get_ontology_iri(),
                uri_pattern=uri_pattern,
                entropy_cache_path=DEFAULT_ENTROPY_CACHE_DIR,
            )
            if on_disk_ent_cache_path is not None and on_disk_ent_cache_path.is_file():
                try:
                    with on_disk_ent_cache_path.open("r", encoding="utf-8") as json_entropy_cache_file:
                        loaded_cache = json.load(json_entropy_cache_file)

                    if isinstance(loaded_cache, dict):
                        round_tripable_entropies = {
                            property_uri_key : float(entropy_value)
                            for property_uri_key, entropy_value in loaded_cache.items()
                        } # ensures JSON numerical values are now python floats
                        
                        self._entropy_cache[uri_pattern] = round_tripable_entropies
                        return round_tripable_entropies
                
                except (JSONDecodeError, OSError):
                    pass # cache is bad, fall through & recompute

        # ACCUMULATE ENTROPIES:
        # iterate the entire set of triples such that:
        # calc_ent(p) \forall (s,p,o) \in G : p substr '/property' (is true)

        per_predicate_to_obj_counts: dict[str, Counter] = defaultdict(Counter)
        for _sub, predicate, obj in self.graph.triples((None, None, None)):
            predicate_uri_str = str(predicate)
            if uri_pattern not in predicate_uri_str:
                continue # skip predicates that do not pattern match
            per_predicate_to_obj_counts[predicate_uri_str][str(obj)] += 1
        
        # VECTORISED ENTROPIES CALC (\w np):
        entropies: dict[str, float] = {}

        for predicate_uri, obj_counts in per_predicate_to_obj_counts.items():
            distinct_objs_for_predicate = len(obj_counts)
            if distinct_objs_for_predicate <= 1: # then the entropy is zero
                entropies[predicate_uri] = float(0.0)
                continue # to the next predicate
            vectorised_distinct_obj_counts = np.fromiter(
                obj_counts.values(), # since objs are { str(obj) : cumulative count }
                dtype=np.float64,
                count=distinct_objs_for_predicate
            )
            
            vectorised_probabilities = vectorised_distinct_obj_counts / vectorised_distinct_obj_counts.sum()
            
            # ^ [ (obj_pred_count / total_obj_pred_count)_1, ..., (obj_pred_count / total_obj_pred_count)_N]
            
            entropies[predicate_uri] = float(-(vectorised_probabilities * np.log2(vectorised_probabilities)).sum())
            
            # ^ [ ( p(x)_1 * log_2 * p(x)_1 ) + ... + ( p(x)_N * log_2 * p(x)_N ) ] <-- SHANNON ENTROPY

            # see: https://i.sstatic.net/T0Otu.png (we calculate for all 'containers' -- ie. our dict keys; which = predicates)
        
        # CACHE THE RESULTS ACCORDINGLY (TO USER PREFERENCE):
        self._entropy_cache[uri_pattern] = entropies
        if use_disk_cache and on_disk_ent_cache_path is not None: # cached entropies JSON -> disk
            atomic_json_write(on_disk_ent_cache_path, entropies)

        return entropies



    def getClassRestrictions(self, cls: owlready2.EntityClass) -> list[dict]:
        """
        Extract OWL restrictions declared on cls (owlready2.EntityClass) via 'is_a'.
        For an `owlready2.EntityClass`, the `is_a` attribute returns a hetreogeneous list \w superordinate entities:

            "Owlready2 provides the .is_a attribute for getting the list of superclasses 
            (__bases__ can be used, but with some limits described in Class constructs, 
            restrictions and logical operators). It can also be modified for adding or 
            removing superclasses."
        
        These superordinates (or superclasses) also include restrictions and anonymous class expressions.

        Each restriction in owlready2 has four attributes:

        ```
        # see: https://github.com/pwin/owlready2/blob/master/class_construct.py#L295

        class Restriction(ClassConstruct):
            is_a = ()
            def __init__(self, Property, type, cardinality = None, value = None, ontology = None, bnode = None):

                self.__dict__["property"]    = Property
                self.__dict__["type"]        = type
                self.__dict__["cardinality"] = cardinality
                
                if (not value is None) or (not bnode):
                    if value is None: value = Thing
                        self.__dict__["value"] = value
                
                super().__init__(ontology, bnode)
        ```

        "Restrictions can be modified in place (Owlready2 updates the quadstore 
        automatically), using the following attributes: .property, .type (SOME, 
        ONLY, MIN, MAX, EXACTLY or VALUE), .cardinality and .value (a Class, an 
        Individual, a class contruct or another restriction)."

        See: https://owlready2.readthedocs.io/en/latest/restriction.html

        FILLER RESOLUTION
        -----------------
        The `.value` attribute carries the filler, whose shape depends on the restriction type:

            SOME | ONLY | MIN | MAX | EXACTLY : ThingClass (or class construct)
            VALUE                             : Individual or Literal
            HAS_SELF                          : bool

        The methdo resolves each case to a `filler_name: str | None`, as below:

            - ThingClass fillers: preferred label (uses 'min(...)' for determinism) or class name
            - individual fillers: preferred label or individual name
            - literal    fillers: string-ified literal value
            - boolean (HAS_SELF): "reflexive" / "non-reflexive"
            - class constructs  : [see below]

        CLASS CONSTRUCT RESOLUTION
        --------------------------

        Since 'is_a' can provide anonymous class expressions, its possible that we recieve complex
        class expressions back in the form of `class_construct` (I believe -- I'm pretty sure these 
        would be nested class constructs, ie. formed through via composition; so we'd need to 
        recursively parse them; which is probably too much for the time being -- leave as a TODO).

        TODO
        
        LIMITATIONS
        -----------

        (a) Only 'is_a' is walked. Restrictions declared as part of equivalence
            relations are omitted (for now -- would also involve recursive parsing;
            also, a restriction on an equivalent complex CLS def is semantically
            different to that of one on a superordinate concept [TODO: think about
            this at a later date].

        (b) The tostring / stringify implementations are fairly naive, ie. we resolve
            labels naively (using min -- but is deterministic) and then for class
            construct fillers, we naively use str(...) -- see above comment.

        OWLREADY2 REFERENCES
        ---------------------
        For information on 'is_a', see: https://owlready2.readthedocs.io/en/latest/class.html#creating-and-managing-subclasses
        For information on restrictions, see: https://owlready2.readthedocs.io/en/latest/restriction.html
        """
        xs_restrictions_for_this_cls: list[dict[str, str | None]] = []

        for superordinate_entity in cls.is_a :

            if not isinstance(superordinate_entity, owlready2.Restriction):
                continue # not a restriction (skip)

            restriction_type = _OWLREADY2_RESTRICTION_TYPE_MAP.get(
                superordinate_entity.type, None
            ) # resolves the restriction type to a str or None

            if restriction_type is None:
                continue # (skip)

            # for the restriction, we require a 'property' label
            # (ie. the property that the restriction applies to)
            restriction_property = superordinate_entity.property
            restriction_property_labels: set[str] = (
                self.getPreferredLabels(entity=restriction_property)
                if hasattr(restriction_property, "iri") else set()
            )

            # deterministic 'property' name selection (naive):
            restriction_property_name = (
                min(restriction_property_labels) if restriction_property_labels
                else str(getattr(restriction_property, "name", str(restriction_property)))
            )

            # local scope & if r.filler is None:
            restriction_filler_name = None
            # otherwise: we require a '(rest.) filler' label
            restriction_filler = superordinate_entity.value
            # NOTE: _OWLREADY2_RESTRICTION_TYPE_MAP[owlready2.HAS_SELF] == "has_self" is True
            if restriction_type == "has_self" and restriction_filler is not None:
                restriction_filler_name = (
                    _HAS_SELF_FILLER_REFLEXIVE if bool(restriction_filler)
                    else _HAS_SELF_FILLER_NON_REFLEXIVE
                )

            elif isinstance(restriction_filler, (owlready2.ThingClass, owlready2.Thing)):
                # the typical case: class-typed filler (SOME|ONLY|MIN|MAX|EXACTLY)
                # OR, case: individual-typed filler (for VALUE restrictions)
                # BOTH support preferred label resolution \w equiv impl:
                restriction_filler_labels: set[str] = self.getPreferredLabels(
                    entity=restriction_filler
                ) # apply the same 'naive' deterministic approach as above:
                restriction_filler_name = (
                    min(restriction_filler_labels) if restriction_filler_labels
                    else str(restriction_filler.name)
                )

            elif restriction_filler is not None:
                # case: VALUE \w literal; class constructs (A&B, A|B, !A) & nested restrictions
                # stringification is naive: .name exists on some constructs, str(...) on all
                # TODO: recursively parse class constructs \w nested restrictions (difficult)
                restriction_filler_name = str(getattr(restriction_filler, "name", str(restriction_filler)))

            # finally:

            # gate cardinality by restriction type:
            restriction_cardinality = (
                superordinate_entity.cardinality
                if restriction_type in { "min", "max", "exactly" } else None
            )

            xs_restrictions_for_this_cls.append(
                {
                    "restriction_type": restriction_type,
                    "property_name": restriction_property_name,
                    "filler_name": restriction_filler_name,
                    "cardinality": restriction_cardinality,
                }
            )

        # sort deterministically:
        xs_restrictions_for_this_cls.sort(
            key=lambda restriction: (
                restriction["property_name"] or "",
                restriction["filler_name"] or "",
            )
        )

        return xs_restrictions_for_this_cls



    def getClassRelationalSignature(self, cls: owlready2.EntityClass) -> dict:
        """
        Return properties where CLS appears in domain or range
        """
        as_domain: list[str] = []
        as_range:  list[str] = []
        
        for itr_obj_property in self.onto.object_properties():
            preferred_obj_property_labels = self.getPreferredLabels(itr_obj_property)
            itr_obj_property_label = (
                min(preferred_obj_property_labels) if preferred_obj_property_labels
                else str(itr_obj_property.name)
            )
            if cls in itr_obj_property.domain:
                as_domain.append(itr_obj_property_label)
            if cls in itr_obj_property.range:
                as_range.append(itr_obj_property_label)

        for itr_data_property in self.onto.data_properties():
            preferred_data_property_labels = self.getPreferredLabels(itr_data_property)
            itr_data_property_label = (
                min(preferred_data_property_labels) if preferred_data_property_labels
                else str(itr_data_property.name)
            )
            if cls in itr_data_property.domain:
                as_domain.append(itr_data_property_label)
            # TODO: consider integrating xsd^datatype
            # instead of range (since range is Literal)

        return {"as_domain": sorted(set(as_domain)), "as_range": sorted(set(as_range))}



    def hasSubjectInGraph(self, uri: str) -> bool:
        """
        Check whether uri appears as a subject in the rdflib graph.
        (fallback for KG instances not enumerated by owlready2's individuals() iterator)
        """
        for _ in self.graph.triples((URIRef(uri), None, None)):
            return True
        return False
