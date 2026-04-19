'''
MISC utils
'''
from pydantic import BaseModel
from pathlib import Path

from logmap_llm.evaluation.io import (
        load_entity_types_from_initial_alignment,
        load_mapping_pairs,
    )
from logmap_llm.evaluation.metrics import (
    classify_conference_pair,
    compute_prf,
)

###
# DEBUG HELPER (TYPE SWITCH FOR RESPONSE FMT)
###

def resolve_response_format_to_str(resp_fmt) -> str:
    '''
    ad-hoc type-switch
    returns the resp_fmt value as str for debug logging
    '''
    if resp_fmt is None:
        return "Plain (None)"
    
    if isinstance(resp_fmt, type) and issubclass(resp_fmt, BaseModel):
        return resp_fmt.__name__

    if isinstance(resp_fmt, BaseModel):
        return f"UNEXPECTED instance of {type(resp_fmt).__name__} (expected the class itself, not an instance)"
    
    if isinstance(resp_fmt, dict):
        return f"raw dict schema (keys: {sorted(resp_fmt.keys())})"

    return f"UNKNOWN type={type(resp_fmt).__name__}"



def compute_conference_m1_m2_stratified(
    system_mappings_path: Path,
    reference_path: Path,
    initial_alignment_path: Path | None = None,
    class_reference_path: Path | None = None,
    property_reference_path: Path | None = None,
) -> dict | None:
    """
    Note: this function is retained under `misc.py` since it should operate at the process orchestration level, 
    rather than (as was previously implemented, at the evaluation level). Therefore, we delegate the use of this 
    function to the same system-level of abstraction as the config generators (TODO).
    
    Though, it is noted that: we don't yet have a process-level orchestration layer. The existing experimental results 
    have been constructed by generating configs (eg. in the case of conference) for each of the ontology pairs (21 pairs) 
    and then using ad-hoc scripts to aggregate the results. This 'works', but it's not... ideal, since a single change or 
    ablation means either a. reproducing all configs or b. manually modifying all existing configs; ie. manually updating 
    keys across 21 configs to perform an ablation (room for error -- its brittle).

    TODO: add 'process orchestration' layer. 'paths.py' module ensures that reads and writes dont cross over processes. 
    In rare cases where separate processes require access to the same file for both read and write operations 
    (as is the case with owlready2, since its quadstore only allows readwrite connections/operations), we apply a lock to 
    the file until a copy (of the file) has been made to a temporary location (eg. /tmp/unique/path/per/process). 
    Each process competes for the lock on init, copies the file, then releases the lock, then resumes execution 
    (this is already implemented, its just the orchestration that needs to be better managed). After all processes
    have completed, the results are then aggregated and stratefied to produce highly interpretable output and can also 
    be fed to scripts for plotting and visualising (TODO: we need some v.nice plotting scripts).

    THIS FUNCTION
    -------------

    Computes the OAEI conference track M1 (class) and M2 (property) stratified metrics for a single task pair; uses 
    the same underlying compute_prf primitive as standard precision/recall/f1, but applies the computation separately 
    to  class-only and property-only subsets of the alignment. 
    
    eg. it can be called directly from an interactive notebook or analysis script when you want to inspect M1/M2 breakdowns
    
    Per-type reference files (can be generated using ad-hoc script)
    ---------------------------------------------------------------
    
    The helper looks for per-entity-type reference files via the naming convention used by the OAEI Conference track; for a
    reference file at 'confOf-iasted.tsv', it expects 'confOf-iasted_class.tsv' and 'confOf-iasted_property.tsv' in the same 
    directory. These paths can be overridden via the 'class_reference_path' and 'property_reference_path' parameters if the 
    caller wants to point at non-conventional locations; if neither per-type reference file exists and no overrides are 
    provided, the helper simply returns None and it is the callers responsibility to interpret this as:
           
           "M1/M2 stratification not applicable for this task"
    
    URI entity type classification:
    -------------------------------

    The system pairs (in each mapping) are classified as class-class or property-property mapping type pairs. These use the LogMap 
    entity type codes read from the initial alignment file; ie. (initial_alignment_path \w the pipe-delimited m_ask format).
    
    If this path is not provided, the helper falls back to using the full system pair set for both M1 and M2 buckets -- the resulting 
    metrics are still internally consistent but effectively disable the split (M1 == M2 == unstratified); callers that care about the 
    M1/M2 distinction !! MUST !! provide the initial alignment path (and it is their responsibility to do so)

    (Finally) This function returns dict | None. The dict has the following shape:

        A dict with zero, one, or two entries:

            {
                "m1_class":    metric dict from compute_prf,
                "m2_property": metric dict from compute_prf,
            }

    Each metric dict has its source field tagged "conference_stratified" so downstream consumers of the JSON can more easily distinguish 
    stratified  from standard entries. Returns None if neither per-type reference exists.
    """
    reference_path = Path(reference_path)
    ref_dir = reference_path.parent
    ref_stem = reference_path.stem

    class_ref_path = Path(
        class_reference_path
        if class_reference_path is not None
        else ref_dir / f"{ref_stem}_class.tsv"
    )

    property_ref_path = Path(
        property_reference_path
        if property_reference_path is not None
        else ref_dir / f"{ref_stem}_property.tsv"
    )

    if not class_ref_path.exists() and not property_ref_path.exists():
        return None

    system_pairs = load_mapping_pairs(Path(system_mappings_path))

    # build URI -> entity type code lookup - missing initial alignment
    # is the degenerate case where both buckets report the full pair set
    uri_types = {}
    if initial_alignment_path is not None:
        uri_types = load_entity_types_from_initial_alignment(
            Path(initial_alignment_path)
        )

    results: dict[str, dict] = {}

    if class_ref_path.exists():
        class_ref = load_mapping_pairs(class_ref_path)

        if uri_types:
            class_system = {
                (s, t) for s, t in system_pairs
                if classify_conference_pair(s, t, uri_types) == "class"
            }
        else:
            class_system = system_pairs
        m1 = compute_prf(class_system, class_ref)
        m1["source"] = "conference_stratified"
        results["m1_class"] = m1

    if property_ref_path.exists():
        property_ref = load_mapping_pairs(property_ref_path)

        if uri_types:
            property_system = {
                (s, t) for s, t in system_pairs
                if classify_conference_pair(s, t, uri_types) == "property"
            }
        else:
            property_system = system_pairs
        m2 = compute_prf(property_system, property_ref)
        m2["source"] = "conference_stratified"
        results["m2_property"] = m2

    return results if results else None

