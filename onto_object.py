from __future__ import annotations

import owlready2

#from constants import LOGGER
from onto_access import OntologyAccess


class OntologyEntryAttr:
    def __init__(
        self, class_uri: str | None, onto: OntologyAccess, onto_entry: owlready2.ThingClass | None = None
    ) -> None:
        assert class_uri is not None or onto_entry is not None
        if class_uri is not None:
            self.thing_class = onto.getClassByURI(class_uri)
        else:
            self.thing_class = onto_entry
        assert self.thing_class is not None, f"Class {class_uri} not found in ontology {onto.get_ontology_iri()}"

        self.annotation: dict[str : set | owlready2.ThingClass] = {"class": self.thing_class}
        self.onto: OntologyAccess = onto
        self.annotate_entry(onto)

    def annotate_entry(self, onto: OntologyAccess) -> None:
        #LOGGER.debug(f"Annotating {self.thing_class}")
        self.annotation["uri"] = self.thing_class.iri
        self.annotation["preffered_names"] = onto.getPrefferedLabels(self.thing_class)
        self.annotation["synonyms"] = onto.getSynonymsNames(self.thing_class)
        self.annotation["all_names"] = onto.getAnnotationNames(self.thing_class)
        self.annotation["parents"] = onto.getAncestors(self.thing_class, include_self=False)
        self.annotation["children"] = onto.getDescendants(self.thing_class, include_self=False)

        for key in ["preffered_names", "synonyms", "all_names"]:
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

    def get_preffered_names(self) -> set[str]:
        return self.annotation["preffered_names"]

    def get_synonyms(self) -> set[str]:
        return self.annotation["synonyms"]

    def __wrap_owlready2_objects(self, owlready2_class: owlready2.ThingClass) -> OntologyEntryAttr:
        return OntologyEntryAttr(class_uri=None, onto_entry=owlready2_class, onto=self.onto)

    def __owlready_set2_objects_set(self, owlready2_set: set[owlready2.ThingClass]) -> set[OntologyEntryAttr]:
        return {self.__wrap_owlready2_objects(owlready2_class) for owlready2_class in owlready2_set}

    def get_children(self) -> set[OntologyEntryAttr]:
        return self.__owlready_set2_objects_set(self.annotation["children"])

    def get_parents(self) -> set[OntologyEntryAttr]:
        return self.__owlready_set2_objects_set(self.annotation["parents"])

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
        """Return set of direct parents of the entry."""
        parents_by_levels = self.__get_relatives_by_levels(1, self.onto.getAncestors)
        return parents_by_levels[1] if len(parents_by_levels) > 1 else set()

    def get_direct_parent(self) -> OntologyEntryAttr | None:
        parent = next(iter(self.annotation["parents"])) if self.annotation["parents"] else None
        return self.__wrap_owlready2_objects(parent) if parent else None

    def get_direct_children(self) -> set[OntologyEntryAttr]:
        """Return set of direct children of the entry."""
        children_by_levels = self.__get_relatives_by_levels(1, self.onto.getDescendants)
        return children_by_levels[1] if len(children_by_levels) > 1 else set()

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
        return self.get_attribute_relatives_names("parents", "preffered_names")

    def get_children_preferred_names(self) -> list[set[str]]:
        return self.get_attribute_relatives_names("children", "preffered_names")

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
