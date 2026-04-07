# LogMap-llm Rework

## Rework Description Document

### Branching Structure

```
---------------------------------------------------------------------------------------------------------------------
|         city-artificial-intelligence/logmap-llm           |               jonathondilworth/logmap-llm             |
---------------------------------------------------------------------------------------------------------------------
 review/april-26    release/april-26       main             │     main       rework/april-26
       │                  │                 │               │      │              |
       │                  │                 ├───────────────FORK─>>├────BRANCH───>├─BRANCH───> • (branch) architecture/port-from-jd-extended
       │                  │                 │                      │              │            │ 
       │                  │                 │               |      │              │            • (commit) package-scaffolding
       │                  │                 │                      │              │            │
       │                  │                 │               |      │              │            • (commit) pipeline-decomposition
       │                  │                 │                      │ [OPTIONAL]   │            │
       │                  │                 │               | fc1  │<─MERGE─ ─ ─ ─│<─MERGE──── • (commit) abstractions
       │                  │                 │                      │              │
       │                  │                 │               |      │              │
       │                  │                 │                      │              ├─BRANCH───> • (branch) refactor/* (depends on previous branch)
       │                  │                 │               |      │              │            │
       │                  │                 │               |      │              │<─ - - - -  • (commit) base-patches
       │                  │                 │               |      │ [OPTIONAL]   │            │
       │                  │                 │                 fc2  │<─MERGE─ ─ ─ ─│<─MERGE──── • (commit) kg-refinement-bypass
       │                  │                 │               |      │              │
       │                  │                 │                      │              │
       │                  │                 │               |      ├              ├─BRANCH───> • (branch) core/* (depends on architecture/port-from-jd-extended)
       │                  │                 │                      │              │            │
       │                  │                 │               |      │              │<─ - - - -  • (commit) owlready2-cache
       │                  │                 │                      │              │            │
       │                  │                 │               |      │              │<─ - - - -  • (commit) consultation-layer-mods
       │                  │                 │                      │ [OPTIONAL]   │            │
       │                  │                 │               | fc3  │<─MERGE─ ─ ─ ─│<─MERGE──── • (commit) answer-format
       │                  │                 │                      │              │
       │                  │                 │               |      │              │
       │                  │                 │                      ├              ├─BRANCH───> • (branch) core-2/* (depends on core/*)
       │                  │                 │               |      │              │            │
       │                  │                 │                      │              │<─ - - - -  • (commit) onto-entity-extend
       │                  │                 │               |      │              │            │
       │                  │                 │                      │              │            ├─BRANCH───> • (branch) core-2/prompt-templates/*
       │                  │                 │               |      │              │            │            │
       │                  │                 │                      │              │            │            • (commit) prompt-templates/<template_name>
       │                  │                 │               |      │              │            │            │
       │                  │                 │                      │              │            │           ...
       │                  │                 │               |      │              │            │            │
       │                  │                 │                      │              │<─ - - - -  │<─ - - - -  • (commit) prompt-templates/sibling-retrieval
       │                  │                 │               |      │              │            │            │
       │                  │                 │                      │              │<─ - - - -  │<─ - - - -  • (commit) prompt-templates/bidirectional-consultation
       │                  │                 │               |      │              │            │            │
       │                  │                 │                      │              │<─ - - - -  │<─MERGE──── • (commit) prompt-templates/few-shot
       │                  │                 │               |      │ [OPTIONAL]   │            │
       │                  │                 │                 fc4  │<─MERGE─ ─ ─ ─│<─MERGE─────• (commit) nlf-verbaliser
       │                  │                 │               |      │              │
       │                  │                 │                      │              │
       │                  │                 │               |      ├              ├─BRANCH───> • (branch) infra/* (depends on *)
       │                  │                 │                      │              │            │
       │                  │                 │               |      │              │<─ - - - -  • (commit) evaluation
       │                  │                 │                      │              │            │
       │                  │                 │               |      │              │<─ - - - -  • (commit) orchestration
       │                  │                 │                      │ [OPTIONAL]   │            │
       │                  │                 │               │ fc5  │<─MERGE─ ─ ─ ─│<─MERGE──── • (commit) results-visualisation
       │                  │                 │                      │              │
       │<─[PULL REQUEST (jd:main->cai/logmap-llm:review-april-26)]─────────────── •
       │                  │                 │               | 
       • REVIEW #1        │                 │                 
       │                  │                 │               |       fc.x -> feature checkpoint x
       • REVIEW #2        │                 │                 
       │                  │                 │               | 
       • FIXES #1+2──PR──>•                 │                 
       │                  │                 │               | 
       •                  │      [OPTIONAL] │             
       │                  •─ ─ ─ ─ ──MERGE─>│               | 
       │                  │                 │                 
       │                  │                 │               | 
       ------------------------------------------------------
```


## Branch Notes

#### architecture/port-from-jd-extended

This revision ports/merges architectural changes from the `jd-extended` branch into the original codebase. Numerous smaller changes are also included & discussed below.

**Major Changes**

* Restructuring the [original code](https://github.com/city-artificial-intelligence/logmap-llm) to accommodate for package scaffolding (e.g., rather than `from constants import *`, we now write `from logmap_llm.constants import *`, etc). We also include a `pyproject.toml` file in the root dir and separate pipeline components in `src/logmap_llm`.

* The pipeline is now composed of [individual 'step' functions](https://github.com/jonathondilworth/logmap-llm/blob/architecture/port-from-jd-extended/src/logmap_llm/pipeline/steps.py); each 'step' produces an output that the following step consumes, while many parameters are accessible via a shared `ctx` object. The commit history surrounding these changes is provided [under this branch](https://github.com/jonathondilworth/logmap-llm/tree/jd-extended).

**Smaller Changes**

* A new LogMap build that can detect undeclared ABox predicates as properties for the OAEI KG track is included.
* Many stubs for future changes that are in the process of being tested and merged in are included.
* The `OntologyAccess` now its own Owlready2 world, rather than the default world (to avoid readwrite contention when processes run in parallel). This is further helped in future patches when an owlready2 quadstore cache is included.
* M_ask column names are provided as an immutable tuple and are copied as a mutable list when needed.
* Project file paths are largely handled by the `logmap_llm.pipeline.paths` object (which reads from config) rather than using many hardcoded paths. This becomes increasingly important when future changes are merged in, in which we require a means of managing the complexity of managing many ablation experiments and ensuring parallel experiments are not overwriting the results from other experiments, etc.
* `logmap_llm.log_utils` provides a utility to write a log file and use colour-coded warnings that print to the terminal.
* `onto_object.py` is now found under `logmap_llm.ontology.entities` and provides an ABS for an entity, extending this class for classes, properties and instances.
* some of the TODOs in `oracle_consultation.py` and `oracle_consultation_managers.py` have been addressed, and are now found under `logmap_llm.oracle.consultation` and `logmap_llm.oracle.manager`.
* Additional modules that contain utilities, boilerplate and stubs have also been included.

**This is probably the largest set of changes before merging in the remaining features.**

_An outline of the feature and planned feature branches are available to review under [logmap_llm_rework.md](https://github.com/jonathondilworth/logmap-llm/blob/architecture/port-from-jd-extended/logmap_llm_rework.md)._

#### refactor/base-patches

BRIEF-DESC-GOES-HERE

1. **base-patches:**
2. **kg-realignment-bypass:**

#### core/*

BRIEF-DESC-GOES-HERE

1. **owlready2-cache:**
2. **consultation-layer-mods:**
3. **answer-format:**

#### core-2/*

BRIEF-DESC-GOES-HERE

1. **onto-entity-extend:**
2. **prompt-templates/\*:**
3. **prompt-templates/sibling-retrieval:**
4. **prompt-templates/few-shot:**
5. **nlf-verbaliser:**

#### infra/*

BRIEF-DESC-GOES-HERE

1. **evaluation:**
2. **orchestration:**
3. **results-visualisation:**
