from __future__ import annotations

import owlready2

#from constants import LOGGER
from onto_access import OntologyAccess


class ClassNotFoundError(Exception):
    pass


class PropertyNotFoundError(Exception):
    pass


class InstanceNotFoundError(Exception):
    pass


class OntologyEntryAttr:
    def __init__(
        self, class_uri: str | None, onto: OntologyAccess, onto_entry: owlready2.ThingClass | None = None
    ) -> None:
        assert class_uri is not None or onto_entry is not None
        if class_uri is not None:
            self.thing_class = onto.getClassByURI(class_uri)
        else:
            self.thing_class = onto_entry

        if self.thing_class is None:
            raise ClassNotFoundError(
                f"Class {class_uri} not found in ontology "
                f"{onto.get_ontology_iri()}. This may be a locality-module "
                f"class that owlready2 does not enumerate. The mapping "
                f"involving this class will be skipped."
            )

        self.annotation: dict[str : set | owlready2.ThingClass] = {"class": self.thing_class}
        self.onto: OntologyAccess = onto
        self.annotate_entry(onto)


    def annotate_entry(self, onto: OntologyAccess) -> None:
        #LOGGER.debug(f"Annotating {self.thing_class}")
        self.annotation["uri"] = self.thing_class.iri
        self.annotation["preferred_names"] = onto.getPreferredLabels(self.thing_class)
        self.annotation["synonyms"] = onto.getSynonymsNames(self.thing_class)
        self.annotation["all_names"] = onto.getAnnotationNames(self.thing_class)
        self.annotation["parents"] = onto.getAncestors(self.thing_class, include_self=False)
        # children/descendants are loaded lazily on first access via  _ensure_children_loaded() 
        # avoids costly subtree traversal for large ontologies (e.g. SNOMED) when no template uses them
        self._children_loaded = False

        for key in ["preferred_names", "synonyms", "all_names"]:
            if not self.annotation[key]:
                self.annotation[key] = {str(self.thing_class.name)}


    def _get_entry_names(self, name_type: str) -> set[str]:
        """Get the names of the entry.

        Args:
            name_type (str): The type of name to get, either "preffered", "synonyms", or "all_names".
            entry (OntologyEntryAttr): The entry to get the names of.


        Returns:
            set[str]: The names of the entry.

        """
        return self.annotation[name_type]


    def get_all_entity_names(self) -> set[str]:
        return self.annotation["all_names"]


    def get_preferred_names(self) -> set[str]:
        return self.annotation["preferred_names"]


    def get_synonyms(self) -> set[str]:
        return self.annotation["synonyms"]


    def __wrap_owlready2_objects(self, owlready2_class: owlready2.ThingClass) -> OntologyEntryAttr:
        return OntologyEntryAttr(class_uri=None, onto_entry=owlready2_class, onto=self.onto)


    def __owlready_set2_objects_set(self, owlready2_set: set[owlready2.ThingClass]) -> set[OntologyEntryAttr]:
        return {self.__wrap_owlready2_objects(owlready2_class) for owlready2_class in owlready2_set}


    def get_children(self) -> set[OntologyEntryAttr]:
        self._ensure_children_loaded()
        return self.__owlready_set2_objects_set(self.annotation["children"])


    def get_parents(self) -> set[OntologyEntryAttr]:
        return self.__owlready_set2_objects_set(self.annotation["parents"])


    def _ensure_children_loaded(self) -> None:
        if not self._children_loaded:
            self.annotation["children"] = self.onto.getDescendants(self.thing_class, include_self=False)
            self._children_loaded = True


    def __get_relatives_by_levels(self, max_level: int, relatives_func: callable) -> dict[int, set[OntologyEntryAttr]]:
        """Get the relatives of the entry by all levels up to max_level."""
        current_level = 0
        current_level_entries: set[owlready2.ThingClass] = {self.thing_class}
        relatives_by_levels = {}

        while current_level_entries and current_level <= max_level:
            relatives_by_levels[current_level] = {
                self.__wrap_owlready2_objects(entry) for entry in current_level_entries
            }
            current_level_relatives = set()
            current_entries_indirect_relatives = set()

            for entry in current_level_entries:
                entity_relatives = relatives_func(entry, include_self=False)
                current_level_relatives.update(entity_relatives)

                for relative in entity_relatives:
                    current_entries_indirect_relatives.update(relatives_func(relative, include_self=False))

            current_level_entries = current_level_relatives.difference(current_entries_indirect_relatives)
            current_level += 1

        return relatives_by_levels


    def get_parents_by_levels(self, max_level: int = 3) -> dict[int, set[OntologyEntryAttr]]:
        """Obtain the parents of the entry by all levels up to max_level."""
        return self.__get_relatives_by_levels(max_level, self.onto.getAncestors)


    def get_children_by_levels(self, max_level: int = 3) -> dict[int, set[OntologyEntryAttr]]:
        """Obtain the children of the entry by all levels up to max_level."""
        return self.__get_relatives_by_levels(max_level, self.onto.getDescendants)


    def get_direct_parents(self) -> set[OntologyEntryAttr]:
        """Return set of direct named-class parents of the entry."""
        parents = set()
        for parent in self.thing_class.is_a:
            if isinstance(parent, owlready2.ThingClass) and parent is not owlready2.Thing:
                parents.add(self.__wrap_owlready2_objects(parent))
        return parents


    def get_direct_parent(self) -> OntologyEntryAttr | None:
        """Return a single direct parent of the entry, or None if root."""
        direct_parents = self.get_direct_parents()
        return next(iter(direct_parents), None) if direct_parents else None


    def get_direct_children(self) -> set[OntologyEntryAttr]:
        """
        Return set of direct children of the entry.
        uses owlready2's subclasses() generator for O(k) lookup rather than traversing the full descendant subtree
        """
        children = set()
        for child in self.thing_class.subclasses():
            if isinstance(child, owlready2.ThingClass):
                children.add(self.__wrap_owlready2_objects(child))
        return children


    def get_siblings(self, max_count: int = 2) -> list[OntologyEntryAttr]:
        """
        return up to max_count sibling classes (classes sharing a direct parent)
        siblings are sorted alphabetically by preferred label for reproducibility
        returns an empty list if the concept has no parents or no siblings
        """
        siblings = set()
        for parent in self.get_direct_parents():
            for child in parent.get_direct_children():
                if child != self:
                    siblings.add(child)

        # sort alphabetically by preferred label for reproducibility
        def sort_key(entry: OntologyEntryAttr) -> str:
            names = entry.get_preferred_names()
            return min(names).lower() if names else ""

        sorted_siblings = sorted(siblings, key=sort_key)
        return sorted_siblings[:max_count]


    def get_attribute_relatives_names(self, relatives_name: str, name_type: str) -> list[set[str]]:
        """Get the names of the parents or children of the entry.

        Args:
            relatives_name (str): The relative type to get the names of.
            name_type (str): The type of name to get.

        Returns:
            list: The names of the parents or children of the entry.

        """
        attribute_function = self.get_parents if relatives_name == "parents" else self.get_children
        return [
            entry_names if (entry_names := entry._get_entry_names(name_type)) else {entry}
            for entry in attribute_function()
        ]

    def get_parents_preferred_names(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("parents", "preferred_names")

    def get_children_preferred_names(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("children", "preferred_names")

    def get_parents_synonyms(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("parents", "synonyms")

    def get_children_synonyms(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("children", "synonyms")

    def get_parents_names(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("parents", "all_names")

    def get_children_names(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("children", "all_names")

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return str(self.thing_class)

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return str(self.annotation)

    def __eq__(self, other: OntologyEntryAttr) -> bool:
        """Check if this OntologyEntryAttr is equal to another OntologyEntryAttr."""
        return self.thing_class == other.thing_class

    def __hash__(self) -> int:
        """Return the hash value of the object."""
        return hash(self.thing_class)


# Property entity representation
# parallel to OntologyEntryAttr for OWL object and data properties
# used by Conference track property prompt template

class PropertyEntryAttr:
    """
    ontological context for a property entity (ObjectProperty or DataProperty)
    provides an interface parallel to OntologyEntryAttr but with property-appropriate context
    
    Parameters
    ----------
    prop_uri : str
        The IRI of the property.
    onto : OntologyAccess
        The loaded ontology containing the property.
    """

    def __init__(self, prop_uri: str, onto: OntologyAccess) -> None:
        self.prop = onto.getPropertyByURI(prop_uri)

        if self.prop is None:
            raise PropertyNotFoundError(
                f"Property {prop_uri} not found in ontology {onto.get_ontology_iri()}. Skipping."
            )

        self.onto: OntologyAccess = onto
        
        self.annotation: dict = {
            "property": self.prop,
            "uri": prop_uri,
            "preferred_names": onto.getPreferredLabels(self.prop),
            "synonyms": onto.getSynonymsNames(self.prop),
            "all_names": onto.getAnnotationNames(self.prop),
            "domain_names": onto.getDomainNames(self.prop),
            "range_names": onto.getRangeNames(self.prop),
        }

        # fall back to local name if no labels found
        for key in ["preferred_names", "synonyms", "all_names"]:
            if not self.annotation[key]:
                self.annotation[key] = {str(self.prop.name)}

    def get_preferred_names(self) -> set[str]:
        return self.annotation["preferred_names"]

    def get_synonyms(self) -> set[str]:
        return self.annotation["synonyms"]

    def get_all_entity_names(self) -> set[str]:
        return self.annotation["all_names"]

    def get_domain_names(self) -> set[str]:
        return self.annotation["domain_names"]

    def get_range_names(self) -> set[str]:
        return self.annotation["range_names"]

    def get_domain_synonyms(self) -> set[str]:
        """Return synonyms for domain classes."""
        synonyms = set()
        for cls in self.onto.getDomainClasses(self.prop):
            syns = self.onto.getSynonymsNames(cls)
            if syns:
                synonyms.update(syns)
        return synonyms

    def get_range_synonyms(self) -> set[str]:
        """Return synonyms for range classes (object properties only)."""
        synonyms = set()
        for cls in self.onto.getRangeClasses(self.prop):
            syns = self.onto.getSynonymsNames(cls)
            if syns:
                synonyms.update(syns)
        return synonyms

    def __repr__(self) -> str:
        return str(self.prop)

    def __str__(self) -> str:
        return str(self.annotation)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PropertyEntryAttr):
            return False
        return self.prop == other.prop

    def __hash__(self) -> int:
        return hash(self.prop)


# Instance entity representation — for KG track individual matching

class InstanceEntryAttr:
    """
    ontological context for an individual entity (KG track)

    Parameters
    ----------
    context : dict
        Pre-fetched context dict from OntologyAccess.getInstanceContext().
    uri : str
        The IRI of the individual.
    onto : OntologyAccess
        The loaded ontology containing the individual.
    """

    def __init__(self, context: dict, uri: str, onto: 'OntologyAccess') -> None:
        self.context = context
        self.uri = uri
        self.onto = onto

    def get_preferred_names(self) -> set[str]:
        return set(self.context.get("labels", []))

    def get_all_entity_names(self) -> set[str]:
        return self.get_preferred_names()

    def get_synonyms(self) -> set[str]:
        return self.get_preferred_names()

    def get_type_names(self) -> list[str]:
        return [t["label"] for t in self.context.get("types", [])]

    def get_type_uris(self) -> list[str]:
        return [t["uri"] for t in self.context.get("types", [])]

    def get_abstract(self) -> str | None:
        return self.context.get("abstract")

    def get_categories(self) -> list[str]:
        return self.context.get("categories", [])

    def get_data_properties(self) -> list[dict]:
        return self.context.get("data_properties", [])

    def get_object_properties(self) -> list[dict]:
        return self.context.get("object_properties", [])

    def get_all_properties(self) -> list[dict]:
        tagged = []
        for dp in self.get_data_properties():
            tagged.append({**dp, "kind": "data"})
        for op in self.get_object_properties():
            tagged.append({**op, "kind": "object"})
        return tagged

    def __repr__(self) -> str:
        labels = self.context.get("labels", [])
        label_str = labels[0] if labels else self.uri
        return f"InstanceEntryAttr({label_str})"

    def __str__(self) -> str:
        return str(self.context)

    def __eq__(self, other) -> bool:
        if not isinstance(other, InstanceEntryAttr):
            return False
        return self.uri == other.uri

    def __hash__(self) -> int:
        return hash(self.uri)


# resolve a URI to a class, property, or instance

def resolve_entity(uri: str, onto: OntologyAccess):
    """
    resolve a URI to an OntologyEntryAttr, PropertyEntryAttr, or InstanceEntryAttr

    Tries:
     1. class resolution first (the common case for Bio-ML/Anatomy)
     2. then property resolution (needed for Conference track)
     3. then individual resolution (needed for KG track)

    Parameters
    ----------
    uri : str
        The entity IRI from M_ask.
    onto : OntologyAccess
        The loaded ontology.

    Returns
    -------
    tuple of (entity_attr, entity_type)
        entity_attr : OntologyEntryAttr, PropertyEntryAttr, or InstanceEntryAttr
        entity_type : str, one of 'class', 'property', 'instance'

    Raises
    ------
    ClassNotFoundError
        If the URI cannot be resolved as a class, property, or individual.
    """
    # try class first (most common case):
    cls = onto.getClassByURI(uri)
    if cls is not None:
        return OntologyEntryAttr(class_uri=uri, onto=onto), "class"

    # try property:
    prop = onto.getPropertyByURI(uri)
    if prop is not None:
        return PropertyEntryAttr(prop_uri=uri, onto=onto), "property"

    # try individual (KG track) — owlready2 index:
    ind = onto.getIndividualByURI(uri)
    if ind is not None:
        context = onto.getInstanceContext(uri)
        return InstanceEntryAttr(context=context, uri=uri, onto=onto), "instance"

    # fallback: check if the URI exists as a subject in the rdflib graph
    # this catches KG track instances that owlready2 doesn't enumerate via onto.individuals() 
    # common with DBkWik RDF dumps where entitie are typed via infobox template classes 
    # ... rather than owl:NamedIndividual

    if onto.hasSubjectInGraph(uri):
        context = onto.getInstanceContext(uri)
        return InstanceEntryAttr(context=context, uri=uri, onto=onto), "instance"

    # none of the above — raise for backward compatibility
    raise ClassNotFoundError(
        f"Entity {uri} not found as class, property, or individual in "
        f"ontology {onto.get_ontology_iri()}. The mapping involving this "
        f"entity will be skipped."
    )