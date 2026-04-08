'''
NOTE: I suppose this is the OBDA layer, so maybe object.py is more appropraite?
NOTE: we could just use LogMap for this though... (would require further changes to its src)
'''

from __future__ import annotations

from abc import ABC, abstractmethod
import owlready2

from logmap_llm.ontology.access import OntologyAccess


class ClassNotFoundError(Exception):
    pass


class PropertyNotFoundError(Exception):
    pass


class InstanceNotFoundError(Exception):
    pass


class OntologyEntity(ABC):

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



class ClassEntity(OntologyEntity):
    def __init__(
        self, class_uri: str | None, onto: OntologyAccess, onto_entry: owlready2.ThingClass | None = None
    ) -> None:
        assert class_uri is not None or onto_entry is not None
        if class_uri is not None:
            self.thing_class = onto.getClassByURI(class_uri)
        else:
            self.thing_class = onto_entry
        
        if self.thing_class is None:
            raise ClassNotFoundError(f"Class {class_uri} not found in ontology.")

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
        self._children_loaded = False
        
        ##
        # TODO: consider defering this to self._ensure_children_loaded (configurable)
        # NOTE: at present, we defer this to self._ensure_children_loaded (lazily load)
        ##
        #self.annotation["children"] = onto.getDescendants(self.thing_class, include_self=False)
        #self._children_loaded = True

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

    # Legacy API (used throughout existing code)
    # def get_preffered_names(self) -> set[str]:
    #     return self.annotation["preffered_names"]

    def get_preffered_names(self) -> set[str]:
        return self.annotation["preferred_names"]

    def get_all_entity_names(self) -> set[str]:
        return self.annotation["all_names"]

    def get_synonyms(self) -> set[str]:
        return self.annotation["synonyms"]
    
    def __wrap_owlready2_objects(self, owlready2_class: owlready2.ThingClass) -> ClassEntity:
        return ClassEntity(class_uri=None, onto_entry=owlready2_class, onto=self.onto)

    def __owlready_set2_objects_set(self, owlready2_set: set[owlready2.ThingClass]) -> set[ClassEntity]:
        return {self.__wrap_owlready2_objects(c) for c in owlready2_set}

    def get_children(self) -> set[ClassEntity]:
        self._ensure_children_loaded()
        return self.__owlready_set2_objects_set(self.annotation["children"])

    def get_parents(self) -> set[ClassEntity]:
        return self.__owlready_set2_objects_set(self.annotation["parents"])

    def __get_relatives_by_levels(self, max_level: int, relatives_func: callable) -> dict[int, set[ClassEntity]]:
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

            # FIX: previously using eval_entities
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
        siblings: set[ClassEntity] = set()
        for parent in self.get_direct_parents():
            for child in parent.get_direct_children():
                if child != self:
                    siblings.add(child)

        def _sort_key(entry: ClassEntity) -> str:
            names = entry.get_preferred_names()
            return min(names).lower() if names else ""

        return sorted(siblings, key=_sort_key)[:max_count]

    ################## TODO: review

    ###
    # TODO: consider whether implementation is required or not
    # NOTE: included for the time being, consider removing if not required.
    ###
    def get_restrictions(self, max_count: int = 3) -> list[dict]:
        """
        Get OWL restrictions declared on this class.
        See `logmap_llm.oracle.access.getClassRestrictions`.
        """
        all_restrictions = self.onto.getClassRestrictions(self.thing_class)
        some_first = sorted(
            all_restrictions,
            key=lambda r: (0 if r["restriction_type"] == "some" else 1),
        )
        return some_first[:max_count]

    ###
    # TODO: consider whether implementation is required or not
    # NOTE: included for the time being, consider removing if not required.
    ###
    def get_relational_signature(self) -> dict:
        """
        Get the relational signature of this class.
        See `logmap_llm.oracle.access.getClassRelationalSignature`.
        """
        return self.onto.getClassRelationalSignature(self.thing_class)

    
    ################## END TODO: review


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


class PropertyEntity(OntologyEntity):

    def __init__(self, prop_uri: str, onto: OntologyAccess) -> None:

        self.prop = onto.getPropertyByURI(prop_uri)

        if self.prop is None:
            raise PropertyNotFoundError(f"Property {prop_uri} not found in ontology.")
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
        result = set()
        try:
            for cls in self.prop.domain:
                if isinstance(cls, owlready2.ThingClass):
                    result.update(self.onto.getSynonymsNames(cls))
        except Exception:
            pass
        return result

    def get_range_synonyms(self) -> set[str]:
        result = set()
        try:
            for cls in self.prop.range:
                if isinstance(cls, owlready2.ThingClass):
                    result.update(self.onto.getSynonymsNames(cls))
        except Exception:
            pass
        return result

    def get_characteristics(self) -> list[str]:
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
        inv = getattr(self.prop, 'inverse_property', None)
        if inv is None:
            return None
        labels = self.onto.getPreferredLabels(inv)
        return next(iter(labels), None) if labels else str(inv.name)

    def __repr__(self) -> str:
        return f"PropertyEntity({self.prop})"

    def __str__(self) -> str:
        return str(self.annotation)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PropertyEntity):
            return False
        return self.annotation.get("uri") == other.annotation.get("uri")

    def __hash__(self) -> int:
        return hash(self.annotation.get("uri", id(self)))


class InstanceEntity(OntologyEntity):

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
    """
    Resolve a URI to a ClassEntity, PropertyEntity, or InstanceEntity.
    Tries class resolution first, then property, then individual.
    """
    # class:
    cls = onto.getClassByURI(uri)
    if cls is not None:
        return ClassEntity(class_uri=None, onto_entry=cls, onto=onto), "class"
    
    # property:
    if hasattr(onto, 'getPropertyByURI'):
        prop = onto.getPropertyByURI(uri)
        if prop is not None:
            return PropertyEntity(uri, onto), "property"
    
    # individual:
    if hasattr(onto, 'getIndividualByURI'):
        ind = onto.getIndividualByURI(uri)
        if ind is not None:
            return InstanceEntity(uri, onto), "instance"
    
    # fallback:
    if hasattr(onto, 'hasSubjectInGraph') and onto.hasSubjectInGraph(uri):
        try:
            return InstanceEntity(uri, onto), "instance"
        except InstanceNotFoundError:
            pass

    raise ClassNotFoundError(f"URI {uri} not found as class, property, or individual.")


# ensure backwards compatability
OntologyEntryAttr = ClassEntity
PropertyEntryAttr = PropertyEntity
InstanceEntryAttr = InstanceEntity

