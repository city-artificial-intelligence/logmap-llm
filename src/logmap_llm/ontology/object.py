"""
logmap_llm.ontology.entities

Ontology entity classes. Provides the OntologyEntity ABC, ClassEntity 
(renamed from OntologyEntryAttr in prior version), PropertyEntity,
InstanceEntity, and the resolve_entity() dispatch function
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import owlready2

from logmap_llm.ontology.access import OntologyAccess

from logmap_llm.constants import VERBOSE

###
# Exceptions
###

class ClassNotFoundError(Exception):
    pass


class PropertyNotFoundError(Exception):
    pass


class InstanceNotFoundError(Exception):
    pass


###
# ABC
###

class OntologyEntity(ABC):
    """Common abstract base class for ontology entities"""

    @abstractmethod
    def get_preferred_names(self) -> set[str]:
        ...

    @abstractmethod
    def get_all_entity_names(self) -> set[str]:
        ...

    @abstractmethod
    def get_synonyms(self) -> set[str]:
        ...

    def __eq__(self, other) -> bool:
        if not isinstance(other, OntologyEntity):
            return NotImplemented
        return getattr(self, 'thing_class', None) == getattr(other, 'thing_class', None)

    def __hash__(self) -> int:
        return hash(getattr(self, 'thing_class', id(self)))

    @property
    def iri(self) -> str:
        """
        A previously overlooked (during refactoring) late addition (12th April 26):
        Includes the canonical IRI, as accessible from any class that extends the
        ABS by directly calling <TYPE>Entity.iri (which would be most intuitive)
        and has been the source of a great deal of pain!
        Its now sourced from self.annotation["uri"], which every subclass populates
        via their constructor (during 'annotate_entry); may raise a key error if
        some problem has occurred during initialisation (ie. is a feature, not a bug)
        """
        return self.annotation["uri"]


###
# ClassEntity
###

class ClassEntity(OntologyEntity):
    """Represents an OWL class entity with its ontological context"""

    def __init__(self, class_uri: str | None, onto: OntologyAccess, onto_entry: owlready2.ThingClass | None = None) -> None:
        assert class_uri is not None or onto_entry is not None
        if class_uri is not None:
            self.thing_class = onto.getClassByURI(class_uri)
        else:
            self.thing_class = onto_entry

        if self.thing_class is None:
            raise ClassNotFoundError(f"Class {class_uri} not found in ontology.")

        self.annotation: dict = {"class": self.thing_class}
        self.onto: OntologyAccess = onto
        self.annotate_entry(onto)

    def annotate_entry(self, onto: OntologyAccess) -> None:
        self.annotation["uri"] = self.thing_class.iri
        self.annotation["preferred_names"] = onto.getPreferredLabels(self.thing_class)
        self.annotation["synonyms"] = onto.getSynonymsNames(self.thing_class)
        self.annotation["all_names"] = onto.getAnnotationNames(self.thing_class)
        self.annotation["parents"] = onto.getAncestors(self.thing_class, include_self=False)
        self._children_loaded = False

        for key in ["preferred_names", "synonyms", "all_names"]:
            if not self.annotation[key]:
                self.annotation[key] = {str(self.thing_class.name)}

    def _ensure_children_loaded(self) -> None:
        if not self._children_loaded:
            self.annotation["children"] = self.onto.getDescendants(
                self.thing_class, include_self=False
            )
            self._children_loaded = True

    def _get_entry_names(self, name_type: str) -> set[str]:
        return self.annotation[name_type]

    def get_preferred_names(self) -> set[str]:
        return self.annotation["preferred_names"]

    def get_all_entity_names(self) -> set[str]:
        return self.annotation["all_names"]

    def get_synonyms(self) -> set[str]:
        return self.annotation["synonyms"]

    def __wrap_owlready2_objects(self, owlready2_class) -> ClassEntity:
        return ClassEntity(class_uri=None, onto_entry=owlready2_class, onto=self.onto)

    def __owlready_set2_objects_set(self, owlready2_set) -> set[ClassEntity]:
        return {self.__wrap_owlready2_objects(c) for c in owlready2_set}

    def get_children(self) -> set[ClassEntity]:
        self._ensure_children_loaded()
        return self.__owlready_set2_objects_set(self.annotation["children"])

    def get_parents(self) -> set[ClassEntity]:
        return self.__owlready_set2_objects_set(self.annotation["parents"])

    def __get_relatives_by_levels(
        self, max_level: int, relatives_func: callable
    ) -> dict[int, set[ClassEntity]]:
        current_level = 0
        current_level_entries = {self.thing_class}
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
                    current_entries_indirect_relatives.update(
                        relatives_func(relative, include_self=False)
                    )

            current_level_entries = current_level_relatives.difference(
                current_entries_indirect_relatives
            )
            current_level += 1

        return relatives_by_levels

    def get_parents_by_levels(self, max_level: int = 3) -> dict[int, set[ClassEntity]]:
        return self.__get_relatives_by_levels(max_level, self.onto.getAncestors)

    def get_children_by_levels(self, max_level: int = 3) -> dict[int, set[ClassEntity]]:
        return self.__get_relatives_by_levels(max_level, self.onto.getDescendants)

    def get_direct_parents(self) -> set[ClassEntity]:
        direct = set()
        for parent in self.thing_class.is_a:
            if isinstance(parent, owlready2.ThingClass) and parent is not owlready2.Thing:
                direct.add(self.__wrap_owlready2_objects(parent))
        return direct

    def get_direct_parent(self) -> ClassEntity | None:
        for parent in self.thing_class.is_a:
            if isinstance(parent, owlready2.ThingClass) and parent is not owlready2.Thing:
                return self.__wrap_owlready2_objects(parent)
        return None

    def get_direct_children(self) -> set[ClassEntity]:
        direct = set()
        for child in self.thing_class.subclasses():
            if isinstance(child, owlready2.ThingClass):
                direct.add(self.__wrap_owlready2_objects(child))
        return direct

    def get_siblings(self, max_count: int = 2) -> list[ClassEntity]:
        """
        gets sibling classes (sharing a direct parent - excluding self)
        siblings are sorted alphabetically by preferred label for 
        reproducibility and capped at max_count
        """
        siblings: set[ClassEntity] = set()
        for parent in self.get_direct_parents():
            for child in parent.get_direct_children():
                if child != self:
                    siblings.add(child)

        def _sort_key(entry: ClassEntity) -> str:
            names = entry.get_preferred_names()
            return min(names).lower() if names else ""

        return sorted(siblings, key=_sort_key)[:max_count]

    def get_restrictions(self, max_count: int = 3) -> list[dict]:
        """Get OWL restrictions declared on this class."""
        all_restrictions = self.onto.getClassRestrictions(self.thing_class)
        some_first = sorted(
            all_restrictions,
            key=lambda r: (0 if r["restriction_type"] == "some" else 1),
        )
        return some_first[:max_count]

    def get_relational_signature(self) -> dict:
        """Get the relational signature of this class."""
        return self.onto.getClassRelationalSignature(self.thing_class)

    def get_attribute_relatives_names(
        self, relatives_name: str, name_type: str
    ) -> list[set[str]]:
        attribute_function = (
            self.get_parents if relatives_name == "parents" else self.get_children
        )
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
        return str(self.thing_class)

    def __str__(self) -> str:
        return str(self.annotation)


###
# PropertyEntity
###

class PropertyEntity(OntologyEntity):
    """
    Ontological context for a property entity (ObjectProperty or DataProperty).
    Provides an interface parallel to ClassEntity but with property-appropriate
    context: domain classes, range classes/datatypes, and characteristics
    """

    def __init__(self, prop_uri: str, onto: OntologyAccess) -> None:
        self.prop = onto.getPropertyByURI(prop_uri)
        if self.prop is None:
            raise PropertyNotFoundError(
                f"Property {prop_uri} not found in ontology."
            )
        self.onto = onto
        self.annotation = {
            "property": self.prop,
            "uri": prop_uri,
            "preferred_names": onto.getPreferredLabels(self.prop),
            "synonyms": onto.getSynonymsNames(self.prop),
            "all_names": onto.getAnnotationNames(self.prop),
        }
        # Populate domain/range names
        self.annotation["domain_names"] = onto.getDomainNames(self.prop)
        self.annotation["range_names"] = onto.getRangeNames(self.prop)

        for key in ["preferred_names", "synonyms", "all_names"]:
            if not self.annotation[key]:
                self.annotation[key] = {str(self.prop.name)}

    def get_preferred_names(self) -> set[str]:
        return self.annotation["preferred_names"]

    def get_all_entity_names(self) -> set[str]:
        return self.annotation["all_names"]

    def get_synonyms(self) -> set[str]:
        return self.annotation["synonyms"]

    def get_domain_names(self) -> set[str]:
        return self.annotation.get("domain_names", set())

    def get_range_names(self) -> set[str]:
        return self.annotation.get("range_names", set())

    def get_domain_synonyms(self) -> set[str]:
        """Get synonyms for domain classes."""
        result = set()
        try:
            for cls in self.prop.domain:
                if isinstance(cls, owlready2.ThingClass):
                    result.update(self.onto.getSynonymsNames(cls))
        except Exception:
            pass
        return result

    def get_range_synonyms(self) -> set[str]:
        """Get synonyms for range classes."""
        result = set()
        try:
            for cls in self.prop.range:
                if isinstance(cls, owlready2.ThingClass):
                    result.update(self.onto.getSynonymsNames(cls))
        except Exception:
            pass
        return result

    def get_characteristics(self) -> list[str]:
        """Return property characteristics (transitive, symmetric, etc.)."""
        chars = []
        try:
            if getattr(self.prop, 'is_transitive', False):
                chars.append("transitive")
            if getattr(self.prop, 'is_symmetric', False):
                chars.append("symmetric")
            if getattr(self.prop, 'is_functional', False):
                chars.append("functional")
            if getattr(self.prop, 'is_inverse_functional', False):
                chars.append("inverse_functional")
        except Exception:
            pass
        return chars

    def get_inverse_name(self) -> str | None:
        """Return the label of the inverse property, or None."""
        inv = getattr(self.prop, 'inverse_property', None)
        if inv is None:
            return None
        labels = self.onto.getPreferredLabels(inv)
        return next(iter(labels), None) if labels else str(inv.name)
    
    def get_deterministic_inverse_name(self) -> str | None:
        """Return the label of the inverse property, or None."""
        inv = getattr(self.prop, 'inverse_property', None)
        if inv is None:
            return None
        labels = self.onto.getPreferredLabels(inv)
        return min(labels) if labels else str(inv.name)

    def __repr__(self) -> str:
        return f"PropertyEntity({self.prop})"

    def __str__(self) -> str:
        return str(self.annotation)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PropertyEntity):
            return False
        return self.iri == other.iri

    def __hash__(self) -> int:
        return hash(self.annotation.get("uri", id(self)))


###
# InstanceEntity
###

class InstanceEntity(OntologyEntity):
    """
    Ontological context for an individual entity (KG track).
    Retrieves structured context from the rdflib graph via
    OntologyAccess.getInstanceContext()
    """

    def __init__(self, uri: str, onto: OntologyAccess) -> None:
        self.uri = uri
        self.onto = onto
        self.context = onto.getInstanceContext(uri)

        if self.context is None:
            raise InstanceNotFoundError(
                f"Instance {uri} not found in ontology."
            )

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
        return f"InstanceEntity({label_str})"

    def __str__(self) -> str:
        return str(self.context)

    def __eq__(self, other) -> bool:
        if not isinstance(other, InstanceEntity):
            return False
        return self.uri == other.uri

    def __hash__(self) -> int:
        return hash(self.uri)

###
# Entity type dispatch
###

def resolve_entity(uri: str, onto: OntologyAccess):
    """Resolve a URI to a ClassEntity, PropertyEntity, or InstanceEntity."""
    # try cls first
    cls = onto.getClassByURI(uri)
    if cls is not None:
        return ClassEntity(class_uri=None, onto_entry=cls, onto=onto), "class"
    # then try prop
    if hasattr(onto, 'getPropertyByURI'):
        prop = onto.getPropertyByURI(uri)
        if prop is not None:
            return PropertyEntity(uri, onto), "property"
    # then try inst
    if hasattr(onto, 'getIndividualByURI'):
        ind = onto.getIndividualByURI(uri)
        if ind is not None:
            return InstanceEntity(uri, onto), "instance"
    # finally, fallback to rdflib graph (for KG instances not in owlready2 index)
    if hasattr(onto, 'hasSubjectInGraph') and onto.hasSubjectInGraph(uri):
        try:
            return InstanceEntity(uri, onto), "instance"
        except InstanceNotFoundError:
            pass
    # otherwise:
    raise ClassNotFoundError(f"URI {uri} not found as class, property, or individual.")



# Backward-compatible aliases
OntologyEntryAttr = ClassEntity
PropertyEntryAttr = PropertyEntity
InstanceEntryAttr = InstanceEntity
