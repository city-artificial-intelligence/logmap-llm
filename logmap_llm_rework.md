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

BRIEF-DESC-GOES-HERE

1. **package-scaffolding:**
2. **pipeline-decomposition:**
3. **abstractions:**

#### refactor/base-patches

BRIEF-DESC-GOES-HERE

1.  **base-patches:**
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
