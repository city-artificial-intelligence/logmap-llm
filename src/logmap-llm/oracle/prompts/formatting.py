from __future__ import annotations

from onto_object import OntologyEntryAttr


def get_name_string(name_set: set | list | OntologyEntryAttr) -> str:
    """Get a string representation of the name set."""
    # If the name_set is a set or list, join the elements with a comma
    if isinstance(name_set, (set, list)):
        return ", ".join(name_set)
    return str(name_set)


def get_single_name(name_set: set | list | str | OntologyEntryAttr) -> str | None:
    """Get a single name from the name set."""
    return next(iter(name_set), None) if isinstance(name_set, (set, list)) else name_set


def select_best_direct_entity_names(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> list[OntologyEntryAttr, OntologyEntryAttr]:
    """If there are multiple direct parents, select one and find child element for it."""
    src_parents = next(iter(src_entity.get_direct_parents()))
    tgt_parents = next(iter(tgt_entity.get_direct_parents()))
    return [
        get_name_string(x.get_preffered_names()) if x else None
        for x in [src_parents, tgt_parents, src_entity, tgt_entity]
    ]


def format_hierarchy(
    hierarchy_dict: dict[int, set[OntologyEntryAttr]], no_level: bool = False, add_thing: bool = True
) -> str:
    formatted = []
    for level, parents in sorted(hierarchy_dict.items()):
        parent_name = get_name_string([get_name_string(i.get_preffered_names()) for i in parents])

        if not add_thing and parent_name == "Thing":
            continue

        if no_level:
            formatted.append(parent_name)
        else:
            formatted.append(f"\tLevel {level}: {parent_name}")

    return formatted if no_level else "\n".join(formatted)


def select_best_direct_entity_names_with_synonyms(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, add_thing: bool = True
) -> list:
    """Select preferred names and synonyms for source and target entities and their direct parents."""

    def get_parent_name(entity: OntologyEntryAttr) -> str | None:
        parent = next(iter(entity.get_direct_parents()), None)
        parent_name = get_name_string(parent.get_preffered_names()) if parent else None
        if parent_name == "Thing" and not add_thing:
            return None
        return parent_name

    def get_clean_synonyms(entity: OntologyEntryAttr) -> list[str]:
        synonyms = list(entity.get_synonyms())
        entity_class = entity.thing_class
        return [] if len(synonyms) == 1 and synonyms[0] == entity_class else synonyms

    src_parent_name = get_parent_name(src_entity)
    tgt_parent_name = get_parent_name(tgt_entity)

    src_entity_name = get_name_string(src_entity.get_preffered_names())
    tgt_entity_name = get_name_string(tgt_entity.get_preffered_names())
    src_synonyms = get_clean_synonyms(src_entity)
    tgt_synonyms = get_clean_synonyms(tgt_entity)

    return [
        src_parent_name,  # string | None
        tgt_parent_name,  # string | None
        src_entity_name,  # string
        tgt_entity_name,  # string
        src_synonyms,  # list of strings
        tgt_synonyms,  # list of strings
    ]


def select_best_sequential_hierarchy_with_synonyms(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, max_level: int
) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    """Select the best synonyms for an entity and its hierarchical parents."""
    src_parents_by_levels = src_entity.get_parents_by_levels(max_level)
    tgt_parents_by_levels = tgt_entity.get_parents_by_levels(max_level)

    def get_synonyms_and_class(parents_by_levels: dict[int, set[OntologyEntryAttr]], idx: int) -> tuple[list[str], str]:
        if len(parents_by_levels) > idx:
            entry = next(iter(parents_by_levels[idx]))
            syns = entry.get_synonyms() if hasattr(entry, "get_synonyms") else []
            cls = str(entry.onto.getClassByURI(entry.annotation["uri"])).split(".")[-1]
            return syns, cls
        return [], ""

    def clean(synonyms: list, cls: str) -> list[str]:
        return [] if len(synonyms) == 1 and next(iter(synonyms)) == cls else synonyms

    src_results = [clean(*get_synonyms_and_class(src_parents_by_levels, i)) for i in range(len(src_parents_by_levels))]
    tgt_results = [clean(*get_synonyms_and_class(tgt_parents_by_levels, i)) for i in range(len(tgt_parents_by_levels))]

    return src_results[0], tgt_results[0], src_results[1:], tgt_results[1:]
