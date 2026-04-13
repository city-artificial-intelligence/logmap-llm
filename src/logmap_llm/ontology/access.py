"""
logmap_llm.ontology.access
ontology access via owlready2 (ess. OBDA layer, 'object.py' -> ORM)
each OntologyAccess instance creates its own owlready2.World 
which is an isolated SQLite quadstore (prevents shared-state issues)
"""
from __future__ import annotations

import contextlib
import logging
from enum import Enum

import owlready2
from owlready2 import sync_reasoner, sync_reasoner_pellet

import math
from collections import Counter

from logmap_llm.constants import VERBOSE

# DH: We have removed the dependency on rdflib. It was only needed to
# satisfy a return-type hint on method getGraph(). We removed the hint,
# and so we comment-out the import of rdflib.  Eventually we will remove
# this commented-out import altogether.
#import rdflib

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
        
    def __init__(
        self,
        urionto: str,
        annotate_on_init: bool = True,
        cache=None,
    ) -> None:
        """
        urionto : str - URI or file path to the ontology.
        annotate_on_init : bool - if true, load the ontology and index annotations immediately.
        cache : object, optional - caching interface (Branch 2); if None => no caching.
        """
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
        self.urionto = str(urionto)
        self._cache = cache

        # TODO: check this claim (I can't quite recall whether this was ths issue)
        # JD: each `OntologyAccess`` instance is assigned its own isolated 
        # World (using an independent in-memory SQLite quadstore; this 
        # prevents shared-state issues (i.e., global entity cache issues,
        # and importantly: rdflib store lock contention) that caused 
        # intermittent problems previously when multiple `OntologyAccess`
        # objects were created in the same process using a shared default_world.

        # When cache_dir is provided, the World is backed by a
        # process-private copy of a persistent SQLite quadstore.
        # This skips the expensive OWL/RDF parse on subsequent runs.

        if self._cache is not None:
            self.world = self._cache.get_cached_world(self.urionto)
        else:
            self.world = owlready2.World()

        if annotate_on_init:
            self.load_ontology()
            self.indexAnnotations()

    ###
    # Methods:
    ###

    def get_ontology_iri(self) -> str:
        return self.urionto

    def load_ontology(
        self,
        reasoner: Reasoner = Reasoner.NONE,
        memory_java: str = "10240",
    ) -> None:
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
                with self.onto:
                    logging.info("Classifying ontology with Pellet...")
                    sync_reasoner_pellet(x=self.world)
                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are %d unsatisfiable classes.", unsat)
            except Exception as e:
                logging.error("Classifying with Pellet failed: %s", e)
                raise e

        elif reasoner == Reasoner.HERMIT:
            try:
                with self.onto:
                    logging.info("Classifying ontology with HermiT...")
                    sync_reasoner(x=self.world)
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

        print(f"There are {len(self.graph)} triples in the ontology")
        print(f"Indexed {len(self._uri_to_class)} classes.")
        print(f"Indexed {len(self._uri_to_property)} properties.")
        print(f"Indexed {len(self._uri_to_individual)} individuals.")
        



    def getOntology(self) -> owlready2.Ontology:
        return self.onto

    def getClassByURI(self, uri: str):
        return self._uri_to_class.get(uri)

    def getClassByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.name.lower() == name.lower():
                return cls
        return None

    def getEntityByURI(self, uri: str):
        return self._uri_to_entity.get(uri)

    def getEntityByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.name.lower() == name.lower():
                return cls
        for prop in list(self.getOntology().properties()):
            if prop.name.lower() == name.lower():
                return prop
        return None

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

    def getAncestorsURIs(self, cls) -> set[str]:
        return {anc_cls.iri for anc_cls in cls.ancestors()}

    def getAncestorsNames(self, cls) -> set[str]:
        return {anc_cls.name for anc_cls in cls.ancestors()}

    def getAncestors(self, cls, include_self: bool = True) -> set:
        return {anc_cls for anc_cls in cls.ancestors(include_self=include_self)}

    def getDescendantURIs(self, cls) -> set[str]:
        return {desc_cls.iri for desc_cls in cls.descendants()}

    def getDescendantNames(self, cls) -> set[str]:
        return {desc_cls.name for desc_cls in cls.descendants()}

    def getDescendants(self, cls, include_self: bool = True) -> set:
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
        return list(self.graph.query(query))

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

    def legacy_indexAnnotations(self) -> None:
        annotation_uris = AnnotationURIs()
        self.entityToSynonyms = {}
        self.allEntityAnnotations = {}
        self.preferredLabels = {}
        self.legacy_populateAnnotationDicts(
            annotation_uris.get_annotation_uris_for_synonyms(),
            self.entityToSynonyms,
        )
        self.legacy_populateAnnotationDicts(
            annotation_uris.get_annotation_uris_for_lexical_annotations(),
            self.allEntityAnnotations,
        )
        self.legacy_populateAnnotationDicts(
            annotation_uris.get_annotation_uris_for_preferred_labels(),
            self.preferredLabels,
        )

    def legacy_populateAnnotationDicts(self, annotation_uris: set, dictionary: dict) -> None:
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

    def indexAnnotations(self) -> None:
        """
        lexical_uris is a superset of both synonym_uris and preferred_uris
        We query each annotation property once via direct rdflib triple
        lookups (bypassing per-URI SPARQL query parsing overhead), then
        distribute each result to the appropriate subset dicts.
        this therefore replaces the previous three-call pattern that 
        ran ~58 individual SPARQL queries: ~21 synonym + ~33 lexical 
        + ~4 preferred, many of which redundantly queried the same URIs
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
        Populate all three annotation dictionaries in a single pass.

        Iterates over the superset of annotation URIs (lexicalAnnotationURIs)
        once using direct rdflib triple-pattern lookups and distributes each
        result to the appropriate subset dictionaries based on set membership.

        For each annotation property URI, both direct annotations (where the
        object is a Literal) and indirect annotations (where the object is an
        intermediate node whose rdfs:label provides the text) are resolved.

        Only annotations with language tag 'en' or 'None' (untagged) are
        included, matching the original per-URI SPARQL query behaviour.

        NOTE: This could be problematic for the Digital Humanities track.
        TODO: check that DH (which I believe is multilinguial) runs as expected.
        """
        from rdflib import URIRef, Literal

        rdfs_label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")

        for ann_uri_str in all_uris:
            
            ann_ref = URIRef(ann_uri_str)
            in_synonyms = ann_uri_str in synonym_uris
            in_preferred = ann_uri_str in preferred_uris

            for s, _p, o in self.graph.triples((None, ann_ref, None)):
                subj_str = str(s)
                if isinstance(o, Literal):
                    # direct annotation (apply language filter) TODO: possibly revise
                    if o.language is not None and o.language != "en":
                        continue
                    self._distribute_annotation(subj_str, o.value, in_synonyms, in_preferred)
                else:
                    # indirect: annotation points to an intermediate node;
                    # follow to rdfs:label for the actual annotation text
                    for _s, _p, label in self.graph.triples((o, rdfs_label, None)):
                        if not isinstance(label, Literal):
                            continue
                        if label.language is not None and label.language != "en":
                            continue
                        self._distribute_annotation(subj_str, label.value, in_synonyms, in_preferred)

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

    def getSynonymsNames(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.entityToSynonyms:
            return set()
        return self.entityToSynonyms[entity.iri]

    def getAnnotationNames(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.allEntityAnnotations:
            return set()
        return self.allEntityAnnotations[entity.iri]

    def getPreferredLabels(self, entity) -> set[str]:
        if entity.iri not in self.preferredLabels:
            return set()
        return self.preferredLabels[entity.iri]

    # to ensure backwards compatability
    getPrefferedLabels = getPreferredLabels

    ###
    # PROPERTY-SPECIFIC LOOKUPS
    ###

    def getPropertyByURI(self, uri: str):
        return self._uri_to_property.get(uri)

    def isProperty(self, uri: str) -> bool:
        return uri in self._uri_to_property

    def getDomainNames(self, prop) -> set[str]:
        names = set()
        try:
            for cls in prop.domain:
                if hasattr(cls, 'iri'):
                    labels = self.getPreferredLabels(cls)
                    if labels:
                        names.update(labels)
                    else:
                        names.add(str(cls.name))
        except Exception:
            pass
        return names

    def getRangeNames(self, prop) -> set[str]:
        names = set()
        try:
            for cls in prop.range:
                if hasattr(cls, 'iri'):
                    labels = self.getPreferredLabels(cls)
                    if labels:
                        names.update(labels)
                    else:
                        names.add(str(cls.name))
        except Exception:
            pass
        return names

    def getInverseName(self, prop) -> str | None:
        inv = getattr(prop, 'inverse_property', None)
        if inv is None:
            return None
        labels = self.getPreferredLabels(inv)
        return next(iter(labels), None) if labels else str(inv.name)
    
    ###
    # INDIVIDUAL-SPECIIFC LOOKUPS
    ###

    def getIndividualByURI(self, uri: str):
        return self._uri_to_individual.get(uri)

    def isIndividual(self, uri: str) -> bool:
        return uri in self._uri_to_individual

    def getLabelsForURI(self, uri: str) -> list[str]:
        """fetch rdfs:label values for any URI via the rdflib graph"""
        from rdflib import URIRef, RDFS # lazy import
        labels = []
        
        for _, _, obj in self.graph.triples((URIRef(uri), RDFS.label, None)):
            label_str = str(obj).strip()
            if label_str:
                labels.append(label_str)
        
        if not labels:
            if '#' in uri:
                labels.append(uri.rsplit('#', 1)[-1])
            elif '/' in uri:
                labels.append(uri.rsplit('/', 1)[-1])

        return labels
    
    ###
    # INSTANCE-SPECIFIC (adv.) METHODS
    ###

    def getInstanceContext(self, uri: str) -> dict:
        """
        Fetch structured context for an individual via rdflib

        Returns dict with keys: 
        uri, labels, types, abstract, categories, data_properties, object_properties

        Used in building prompts that use 'the full context'.
        
        NOTE (TODO): We should probably consider using LogMap itself as our OBDA layer
        in which case, logic similar to this can be used to resolve individual types, 
        categories, obj/data props, etc. For e.g., abstracts, we might consider specific
        parameters in parameters.txt that allow for extracting (or accessing) such predicates
        """
        # lazy import:
        from rdflib import URIRef, RDF, RDFS, Literal, Namespace
        
        # namespace specs:
        DCT = Namespace("http://purl.org/dc/terms/")
        DBKWIK = Namespace("http://dbkwik.webdatacommons.org/ontology/")

        subject = URIRef(uri)
        labels = self.getLabelsForURI(uri)

        # fetch types:
        types = []
        for _, _, obj in self.graph.triples((subject, RDF.type, None)):
            type_uri = str(obj)
            type_labels = self.getLabelsForURI(type_uri)
            type_label = type_labels[0] if type_labels else type_uri.rsplit('/', 1)[-1]
            types.append({"uri": type_uri, "label": type_label})

        # fetch abstracts (track-specific):
        abstract = None
        for _, _, obj in self.graph.triples((subject, DBKWIK.abstract, None)):
            abstract = str(obj)
            break
        
        if abstract is None:
            for _, _, obj in self.graph.triples((subject, RDFS.comment, None)):
                abstract = str(obj)
                break

        # fetch categories:
        categories = []
        for _, _, obj in self.graph.triples((subject, DCT.subject, None)):
            cat_labels = self.getLabelsForURI(str(obj))
            categories.extend(cat_labels if cat_labels else [str(obj).rsplit('/', 1)[-1]])

        # fetch predicates:
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
                datatype = str(obj.datatype) if obj.datatype else "string"
                if datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
                    datatype = datatype.rsplit('#', 1)[-1]
                data_properties.append({
                    "predicate_uri": pred_uri,
                    "predicate_label": pred_label,
                    "value": str(obj),
                    "datatype": datatype,
                })
            else:
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
    

    def compute_predicate_entropies(self, uri_pattern: str = "/property/") -> dict:
        """
        Compute Shannon entropy of value distributions for matchable predicates
        higher entropy -> the predicate is more discriminating for entity resolution
        """
        # caching mechanism
        if not hasattr(self, "_entropy_cache"):
            self._entropy_cache: dict[str, dict] = {}
        if uri_pattern in self._entropy_cache:
            return self._entropy_cache[uri_pattern]

        entropies: dict[str, float] = {}

        unique_predicates = set(self.graph.predicates(None, None))

        for pred in unique_predicates:
            pred_uri = str(pred)
            if uri_pattern not in pred_uri:
                continue

            value_counts: Counter = Counter()
            for _s, _p, obj in self.graph.triples((None, pred, None)):
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

        self._entropy_cache[uri_pattern] = entropies
        return entropies
    
    
    def hasSubjectInGraph(self, uri: str) -> bool:
        """
        Check whether *uri* appears as a subject in the rdflib graph.
        fallback for KG track instances not enumerated by owlready2's individuals()
        (as is the case in OAEI KG 2025)
        """
        from rdflib import URIRef
        for _ in self.graph.triples((URIRef(uri), None, None)):
            return True
        return False
    

    ###
    # STUBS
    # NOTE: whether these need implementing depends on the final set of prompts we deicde to use
    # NOTE: (8 April 26) included them for the time being, we can always remove them.
    ###

    def getClassRestrictions(self, cls) -> list[dict]:
        """
        Extract OWL restrictions declared on class via `is_a`.

        Returns a list of dicts with keys:
        
        * restriction_type: str/enum ('some', 'only', 'min', 'max', 'exactly')
        * property_name: str (label),
        * filler_name: str (label),
        * cardinality: 
        
        Sorted by (property_name, filler_name) for reproducibility.
        """
        # TODO: consider migrating to constants.py (?)
        RESTRICTION_TYPE_MAP = {
            owlready2.SOME: "some",
            owlready2.ONLY: "only",
            owlready2.MIN: "min",
            owlready2.MAX: "max",
            owlready2.EXACTLY: "exactly",
        }
        restrictions: list[dict] = []
        for parent in cls.is_a:
            if not isinstance(parent, owlready2.Restriction):
                continue
            rtype = RESTRICTION_TYPE_MAP.get(parent.type)
            if rtype is None:
                continue

            # property label
            prop = parent.property
            prop_labels = (
                self.getPreferredLabels(prop)
                if hasattr(prop, "iri")
                else set()
            )
            prop_name = (
                next(iter(prop_labels), None)
                if prop_labels
                else str(getattr(prop, "name", str(prop)))
            )

            # filler label
            filler = parent.value
            if isinstance(filler, owlready2.ThingClass):
                filler_labels = self.getPreferredLabels(filler)
                filler_name = (
                    next(iter(filler_labels), None)
                    if filler_labels
                    else str(filler.name)
                )
            elif filler is not None:
                filler_name = str(getattr(filler, "name", str(filler)))
            else:
                filler_name = None

            restrictions.append(
                {
                    "restriction_type": rtype,
                    "property_name": prop_name,
                    "filler_name": filler_name,
                    "cardinality": getattr(parent, "cardinality", None),
                }
            )

        restrictions.sort(
            key=lambda r: (r["property_name"] or "", r["filler_name"] or "")
        )
        return restrictions


    def getClassRelationalSignature(self, cls) -> dict:
        """
        Returns properties where the classe (cls) appears in domain or range.
        ie. 
        ```
        {
            'as_domain': [label, ...], 
            'as_range':  [label, ...],
        }
        ```
        each list sorted alphabetically.
        """
        as_domain: list[str] = []
        as_range: list[str] = []
        
        for prop in self.onto.object_properties():
            labels = self.getPreferredLabels(prop)
            label = next(iter(labels), None) if labels else str(prop.name)
            if cls in prop.domain:
                as_domain.append(label)
            if cls in prop.range:
                as_range.append(label)

        for prop in self.onto.data_properties():
            labels = self.getPreferredLabels(prop)
            label = next(iter(labels), None) if labels else str(prop.name)
            if cls in prop.domain:
                as_domain.append(label)

        return {
            "as_domain": sorted(as_domain), 
            "as_range": sorted(as_range)
        }