## LogMapLLM
Large Language Models as Oracles for Ontology Alignment with LogMap (work in progress).

There are many methods and systems to tackle the ontology alignment problem, yet a major challenge persists in producing high-quality mappings among a set of input ontologies. Adopting a human-in-the-loop approach during the alignment process has become essential in applications requiring very accurate mappings. However, user involvement is expensive when dealing with large ontologies. In this work, we evaluate the feasibility of using Large Language Models (LLM) to aid the ontology alignment problem. The use of LLMs is focused only on the validation of a subset of correspondences where an ontology alignment system (e.g., LogMap) is very uncertain. We have conducted an extensive analysis over several tasks of the Ontology Alignment Evaluation Initiative (OAEI), eveluating the performance of several state-of-the-art LLMs using different ontology-driven prompt templates. In the OAEI 2025 Bio-ML track, LogMap with an LLM-based Oracle has achieved the top-2 overall results. LLM efficacy is also assessed against simulated Oracles with varying error rates.

Current efforts are focusing on the creation of an integrated LogMapLLM pipeline (this repository).

### Additional repositories

- LogMap Ontology Alignment System: [https://github.com/ernestojimenezruiz/logmap-matcher](https://github.com/ernestojimenezruiz/logmap-matcher)
- Experiments with different LLMs (and prompts) as diagnostic tools (e.g., Oracles): [https://github.com/city-artificial-intelligence/rai-ukraine-kga-llm](https://github.com/city-artificial-intelligence/rai-ukraine-kga-llm)
 

### References

- Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, Ernesto Jiménez-Ruiz, and Artur Garcez. **Large Language Models as Oracles for Ontology Alignment**. [arXiv](https://ernestojimenezruiz.github.io/assets/pdf/llm-oa-oracles-2025.pdf). 2025
- Ernesto Jiménez-Ruiz, Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, and Artur Garcez. **LogMap Family welcomes LogMapLLM in the OAEI 2025**. OM 2025: The 20th International Workshop on Ontology Matching collocated with the 24th International Semantic Web Conference (ISWC). 2025.

