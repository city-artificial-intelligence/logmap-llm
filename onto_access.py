
import contextlib
import logging
from enum import Enum

import owlready2

# DH: We have removed the dependency on rdflib. It was only needed to
# satisfy a return-type hint on method getGraph(). We removed the hint,
# and so we comment-out the import of rdflib.  Eventually we will remove
# this commented-out import altogether.
#import rdflib

from owlready2 import default_world, get_ontology, sync_reasoner, sync_reasoner_pellet


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
    def __init__(self, urionto: str, annotate_on_init: bool = True) -> None:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
        self.urionto = str(urionto)
        if annotate_on_init:
            self.load_ontology()
            self.indexAnnotations()

    def get_ontology_iri(self) -> str:
        return self.urionto

    def load_ontology(self, reasoner: Reasoner = Reasoner.NONE, memory_java: str = "10240") -> None:
        self.onto: owlready2.Ontology = get_ontology(self.urionto).load()
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
                with self.onto:  # it does add inferences to ontology
                    # Is this wrt data assertions? Check if necessary
                    # infer_property_values = True, infer_data_property_values = True
                    logging.info("Classifying ontology with Pellet...")
                    sync_reasoner_pellet()  # it does add inferences to ontology
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
                    sync_reasoner()  # HermiT doe snot work very well....
                    unsat = len(list(self.onto.inconsistent_classes()))
                    logging.info("Ontology successfully classified.")
                    if unsat > 0:
                        logging.warning("There are %d unsatisfiable classes.", unsat)
            except owlready2.OwlReadyOntologyParsingError:
                logging.info("Classifying with HermiT failed.")

        self.graph = default_world.as_rdflib_graph()
        # DH: This used to be a logging.info() statement. But for
        # LogMap-LLM we have promoted it to a print() statement, so
        # the user can always see how big the ontology is that is
        # being processed and prepared from prompt building.
        print(f"There are {len(self.graph)} triples in the ontology")

    def getOntology(self) -> owlready2.Ontology:
        return self.onto

    def getClassByURI(self, uri: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.iri == uri:
                return cls
        return None

    def getClassByName(self, name: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.name.lower() == name.lower():
                return cls
        return None

    def getEntityByURI(self, uri: str) -> owlready2.EntityClass:
        for cls in list(self.getOntology().classes()):
            if cls.iri == uri:
                return cls
        for prop in list(self.getOntology().properties()):
            if prop.iri == uri:
                return prop
        return None

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
        self.populateAnnotationDicts(annotation_uris.get_annotation_uris_for_synonyms(), self.entityToSynonyms)
        self.populateAnnotationDicts(
            annotation_uris.get_annotation_uris_for_lexical_annotations(), self.allEntityAnnotations
        )
        self.populateAnnotationDicts(annotation_uris.get_annotation_uris_for_preferred_labels(), self.preferredLabels)

    def populateAnnotationDicts(self, annotation_uris: set, dictionary: dict) -> None:
        """Populate the given dictionary with annotations from the provided URIs.

        This method queries a graph for annotations based on the provided URIs and
        populates the given dictionary with the results. Only annotations with
        language set to English or None are added to the dictionary.

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
            return {}
        return self.entityToSynonyms[entity.iri]

    def getAnnotationNames(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.allEntityAnnotations:
            return {}
        return self.allEntityAnnotations[entity.iri]

    def getPrefferedLabels(self, entity: owlready2.Thing) -> set[str]:
        if entity.iri not in self.preferredLabels:
            return {}
        return self.preferredLabels[entity.iri]
