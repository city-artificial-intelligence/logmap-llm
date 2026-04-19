"""
logmap_llm.ontology.vocabularies

We provide a set of vocabulary presets which describe how different KG 
families encode common annotation concepts (eg. 'abstract', 'description 
text', and [topical] 'categories'); these vocabularies also define how
URI structure should be interpreted; for instance, for any dbpedia family
of KG, identifying properties during alignment would rely on matching:

    if '*/property/*' matches URI -> extracts the property from the end of URI

Similarly for classes and instances, /class/, /resource/, etc. might be used.
The vocabularies we provide here are set under [alignmentTask] in the config.toml.

LogMap-LLM's ontology-access layer is KG-agnostic: it reads the vocabulary
attached to each OntologyAccess instance rather than hard-coding any
naming convention. Presets here cover the OAEI 2025 tracks LogMap-LLM is
evaluated against:

    (1) Anatomy
    (2) BioML
    (3) LargeBio
    (4) Conference
    (5) Knowledge Graph (KG)
    (6) Circular Economy (CE)
    (7) Digital Humanities (DH)

You can add your own/new vocabularies by instantiating:

    'OntologyConventionVocabulary'

and (optionally) registering the instance under a string name for 
config-driven selection.

field(s) semantics
------------------

    abstract_predicates      : ordered tuple; first predicate with a value wins

    category_predicates      : unordered; values from all are unioned

    extra_handled_predicates : predicates to exclude from the generic
                               data/object property sweep even though they are
                               not surfaced under `abstract` or `categories`
                               (e.g. rdfs:comment used as a fallback abstract
                               but still appearing explicitly)

    property_uri_substring   : substring identifying property URIs in this KG
                               (default ""; empty matches everything)

    class_uri_substring      : as above, for class URIs

    instance_uri_substring   : as above, for instance URIs

    language_filter          : accepted rdfs:label / annotation language tags
                               ("en",) for monolingual English; empty tuple
                               accepts any language (including untagged)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from rdflib import Namespace, RDFS
from rdflib.namespace import DCTERMS, SKOS

###
# COMMON NAMESPACES
# -----------------
###

DBO    = Namespace("http://dbpedia.org/ontology/")
DBKWIK = Namespace("http://dbkwik.webdatacommons.org/ontology/")
SCHEMA = Namespace("http://schema.org/")

###
# THE DEFAULT VOCABULARY DATACLASS
#   To provide extensions, simply instanciate an 'OntologyConventionVocabulary'
#   and assign it to a 'constant', which can then be imported into:
#       
#       src/logmap_llm/ontology/access.py
# 
#   and set as 'resolved_vocabulary=<VOCAB_NAME>' under [alignmentTask] 
#   in the selected `config.toml`; see below for examples of:
#
#       1. DEFAULT_VOCAB (ALL TASKS \ {KG,DH})
#       2. DBPEDIA_FAMILY_VOCAB (KG)
#       3. MULTILINGUAL_SKOS_VOCAB (DH)
#
#   These are usable by 
###

@dataclass(frozen=True)
class OntologyConventionVocabulary:
    """
    Frozen container describing an onto/kg family of annotation and URI-structure 
    conventions. Instances are immutable, hashable, and safe to share across 
    OntologyAccess objs. Predicate fields hold URI strings (not rdflib URIRef objects).
    This allows the dataclass to be serialisable with JSON (for config round-tripping)
    """
    name: str = "default"

    abstract_predicates:      tuple[str, ...] = (str(RDFS.comment),)
    category_predicates:      tuple[str, ...] = ()
    extra_handled_predicates: tuple[str, ...] = ()

    property_uri_substring: str = ""
    class_uri_substring:    str = ""
    instance_uri_substring: str = ""

    language_filter: tuple[str, ...] = ("en",)

    def accepts_language(self, lang: str | None) -> bool:
        """
        True if a literal with the given language tag (or None for untagged literals) 
        should be kept under this vocabulary. Empty language_filter = 'accept everything', 
        which is the correct default for fully multilingual tracks (such as DH).
        """
        if not self.language_filter:
            return True
        if lang is None:
            return True  # untagged literals are always accepted
        return lang in self.language_filter



###
# STANDARD PRESETS
# ----------------
# The 'out-of-the-box' configuration uses the 'DEFAULT_VOCAB'
# it is considered 'safe for any graph' & extracts rdfs:comment 
# as abstract and no categories...
###

# General RDF (default):
# Extracts rdfs:comment as abstract and no categories.
# Used by: anatomy, bio-ml, conference, and circular economy.

DEFAULT_VOCAB = OntologyConventionVocabulary(
    name="default",
)



# VOCABULARY: DBpedia-family
# --------------------------
# (DBpedia proper, DBkWik, and other MediaWiki derivatives) 
# Used by the OAEI Knowledge Graph track:
# The 'abstract resolution order' is: 
#   (1) DBkWik's namespace
#   (2) DBpedia's for portability across DBpedia-shaped datasets
#   (3) rdfs:comment fallback

DBPEDIA_FAMILY_VOCAB = OntologyConventionVocabulary(
    name="dbpedia_family",
    abstract_predicates=(
        str(DBKWIK.abstract),
        str(DBO.abstract),
        str(RDFS.comment),
    ),
    category_predicates=(str(DCTERMS.subject),),
    extra_handled_predicates=(str(RDFS.comment),),
    property_uri_substring="/property/",
    class_uri_substring="/class/",
    instance_uri_substring="/resource/",
)



# VOCABULARY: SKOS thesauri with multilingual labels
# --------------------------------------------------
# Used by the OAEI Digital Humanities track: 
#   - Empty language_filter accepts all language tags.
#   - skos:definition / skos:scopeNote are the 'conventional 
#   - abstract analogues'.
#   - skos:inScheme substitutes for category membership.

MULTILINGUAL_SKOS_VOCAB = OntologyConventionVocabulary(
    name="multilingual_skos",
    abstract_predicates=(
        str(SKOS.definition),
        str(SKOS.scopeNote),
        str(RDFS.comment),
    ),
    category_predicates=(str(SKOS.inScheme),),
    extra_handled_predicates=(),
    language_filter=(),
)



###
# PRESET REGISTRY
# ---------------
# For (name -> instance) resolution enabling config-driven vocabulary 
# selection; ie. specify:
#
#   'ontology_vocabulary=<VOCAB_NAME>' under [alignmentTask]
#
# in `*config.toml` when running `python -m logmap_llm --config configs/*config.toml`.
###

PRESET_REGISTRY: dict[str, OntologyConventionVocabulary] = {
    DEFAULT_VOCAB.name: DEFAULT_VOCAB,
    DBPEDIA_FAMILY_VOCAB.name: DBPEDIA_FAMILY_VOCAB,
    MULTILINGUAL_SKOS_VOCAB.name: MULTILINGUAL_SKOS_VOCAB,
}


def get_preset(name: str) -> OntologyConventionVocabulary:
    """
    Resolve an 'OntologyConventionVocabulary' by name.
    """
    try:
        return PRESET_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown vocabulary preset '{name}'. " 
            f"Known presets: {known}. "
        ) from exc


def register_preset(vocab: OntologyConventionVocabulary) -> None:
    """
    Register an additional vocabulary instance under its `name` attribute,
    making it available via `get_preset(name)`. For use by downstream
    packages extending LogMap-LLM with custom KG conventions.
    """
    if not vocab.name:
        raise ValueError("Vocabulary 'name' must be non-empty to register.")
    PRESET_REGISTRY[vocab.name] = vocab

