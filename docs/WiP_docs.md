# LogMap-LLM: New Features, Rework, Patch Notes & Documentation (review/april-26)

_(initial writing on these docs is WiP)_

### SiblingSelector

SiblingSelector is used in two circumstances: (1) **during prompt building**, when contextual descriptions for prompt injection include sibling classes, and (2) **during few-shot example construction**, when contrastive examples should include a near-miss.

The SiblingSelector provides multiple ways to select siblings to use for the prior use-cases. This is important when using OWL ontologies pre-classification (ie. in circumstances where the ontology represents the stated view, rather than the inferred view). This is because it is not uncommon for a parent class to have **MANY** children (any given class could have a large number of siblings). For example, in SNOMED CT, a $\mathrm{Metabolic Disorder} \sqsubseteq \mathrm{Disease}$, and $\mathrm{Disease}$ has 170 children, meaning $\mathrm{MetabolicDisorder}$ has 169 siblings. As such, selecting the most appropriate sibling classes to use for grounding the language models response (or during in-context learning with few-shot examples) is an important consideration. To address this challenge, we implement a SiblingSelector, which: 

1. Accepts a given class, denoted $C$, as input.

2. Traverses to the direct parent classes, constructing the set $\hat{C}$ _(accounting for multiple inheritance)_: 

  $$\hat{C} = \mathrm{parents}(C)$$
 
3. Obtains the direct children $\forall p \in \hat{C}$ as the unionised set $\hat{C}_{ch}$: 

$$\hat{C}_{ch} = \big\{ \bigcup\limits_{p\ \in\ \hat{C}} \mathrm{children}(p)\ \big\}$$
 
4. Computes the sibling set $S$: 

$$\mathcal{S} = \hat{C}_{ch}\ \textbackslash\ \big\{\ C\ \big\}$$
 
 5. Selects the most appropriate siblings in $S$ (ie. the top-k siblings) according to any one of the following criteria:

	 * Order siblings alphanumerically by `rdfs:label` _(naive)_.
	 * Order siblings by `rdfs:label` length, prioritising the shortest labelled sibling _(naive)_.
	 * Embed each $s \in S$ using the SapBERT model and order the resulting embeddings by maximal cosine similarity, selecting the top-k siblings.
	 * Embed each $s \in S$ using the Sentence Transformer model `all-MiniLM-L12-v2` and order the siblings by maximal cosine similarity, selecting the top-k siblings.

**Note that SapBERT uses a contrastive learning objective which aims to identify near-synonymous terms.** For example, $\mathrm{Nutritional Disorder}$ matches $\mathrm{Nutritional Deficiency Associated Condition}$. Both classes $\sqsubseteq \mathrm{Disease}$. This exemplifies the use case for constructing a 'near-miss' example for a mapping either class. **However, it is important to note that we do not explicitly check whether the example constructed for a near-miss could in fact be true under inference.** As such, this does not necessarily guarantee that the provided negative example holds; and is therefore important to note that this _could_ confuse a language model. However, the implementation exists explicitly to test this. That is, we hypothesise that for most cases, contrastive near-miss examples _should_ help.

#### Additional Notes (SiblingSelector):

* The code includes a `max_candidates` parameter which defaults to 50 _(but can be configured or tuned)_. In the default case, if $>50$ siblings exist, the candidate sibling set is sliced according to: `all_siblings = list(all_siblings)[:max_candidates]` and is simply a mechanism to avoid incurring too much computational cost during pipeline runs. However, it would be advised to set `max_candidates` to `max(sibling_count)` upon bootstrapping the pipeline by running the `obtain_max_sibling_count.py --owl-file /path/to/ontology` found under the `/scripts` directory, then adjusting the `DEFAULT_MAX_SIBLING_CANDIDATES` in `constants.py` to the returned value before performing end-to-end alignment. 
* In cases where $|S| \leq k$ _(where $k$ is the specified value for `DEFAULT_TOP_K` in `constants.py`)_ the `SiblingSelector` doesn't bother computing the embeddings and simply returns the entire set, each with a score of $1.0$ in place of cosine similarity. However, it is advisable to set $k <<$ `DEFAULT_MAX_SIBLING_CANDIDATES` (eg. $k=[1, 3]$) to avoid bloating the prompts that are set to `require_siblings=True`.
* We also implement an IRI-keyed embedding cache. This is to account for situations where the same class appears in $>1$ mapping within $\mathcal{M}_{ask}$ and meaningfully contributes to the efficiency of the pipeline _(the pipeline is actually **quite efficient!**).

### Evaluation Metrics

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \equiv \frac{|\mathcal{M}_S \cap \mathcal{M}_{RA}|}{|\mathcal{M}_S|}
$$

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \equiv \frac{|\mathcal{M}_S \cap \mathcal{M}_{RA}|}{|\mathcal{M}_{RA}|}
$$

$$
\text{F}_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Templates

_TODO (see code comments)_
### OracleConsultation

_TODO (see code comments)_
### OracleConsultationManager

_TODO (see code comments)_
### OntologyAccess

_TODO (see code comments)_
### OntologyObject

_TODO (see code comments)_
### OntologyCache

_TODO (see code comments)_
