# %% [markdown]
# ---
#
# ### Python script driver for LogMap-LLM
# 
# This Python script driver for LogMap-LLM provides a command-line,
# non-interactive LogMap-LLM user experience. You launch LogMap-LLM
# at the command line and it does everything for you, writing its
# output to the console.
#
# A Jupyter notebook driver for LogMap-LLM also exists. It provides
# an interactive, Python-programmer LogMap-LLM user experience.
# 
# ---

# %% [markdown]
# Basic imports

# %%
import argparse
import os
import os.path
import sys
import tomllib
import pandas as pd
import json
import subprocess
import time
import platform
from datetime import datetime, timezone

import oracle_prompt_building as opb
import oracle_consultation as oc
import developer_prompts as dp
from constants import PAIRS_SEPARATOR
from log_utils import TeeWriter, error, warning, warn, info, step, success

#########
# HELPERS
#########

# TODO: move helpers to somewhere more appropriate.

# Right. So, while this contributes to continuing to build technical debt
# i.e., the use of owlready2, etc. It is, for now, a neccesary evil, as
# fetching sibling, etc. is not supported natively through LogMap.
# TODO: consider a better approach than spawning a subprocess for running
# multiple JVMs (forcing us to ensure that we can round-trip serialised JSON
# values in maintaining a consistent internal representation)
def normalise_prediction_column(series):
    """
    normalises oracle prediction values for CSV round-tripping
    CSV serialisation means python booleans are converted to strings
    this restores them to native booleans while preserving 
    string values like "error" and "skipped"
    """
    def _normalise(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            if val.lower() in ("true", "yes"):
                return True
            if val.lower() in ("false", "no"):
                return False
        return val  # "error", "skipped", NaN, etc.
    return series.map(_normalise)


def format_duration(seconds):
    """format a duration in seconds to a human-readable string"""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def classify_endpoint(base_url):
    """classify the LLM endpoint type from the base_url config value"""
    if base_url is None:
        return "OpenRouter"
    base_url_lower = base_url.lower()
    if "openrouter.ai" in base_url_lower:
        return "OpenRouter"
    if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
        return "vLLM (local)"
    return "UNKNOWN"


def get_gpu_info():
    """attempt to fetch GPU information via nvidia-smi; returns a string for reporting"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            info_parts = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    info_parts.append(
                        f"  GPU {i}: {parts[0]} -- "
                        f"{parts[1]} MiB total, {parts[2]} MiB used, {parts[3]} MiB free"
                    )
            return '\n'.join(info_parts) if info_parts else "Not available"
        return "Not available"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "Not available (nvidia-smi not found)"


# %%
# arrange for modules to be reloaded automatically, so changes are
# recognised seamlessly
#%load_ext autoreload
#%autoreload 2

# %% [markdown]
# JPype imports

# %%
# Import the module
import jpype

# Allow Java modules to be imported
import jpype.imports

# Import all standard Java types into the global scope
from jpype.types import *

# %% [markdown]
# Load the LogMap-LLM configuration file

# %%
parser = argparse.ArgumentParser(
    description="LogMap-LLM: LLM-enhanced ontology alignment"
)
parser.add_argument(
    "--config", "-c",
    type=str,
    default="logmap-llm-config-basic.toml",
    help="Path to the TOML configuration file (default: logmap-llm-config-basic.toml)",
)
parser.add_argument(
    "--reuse-align",
    action="store_true",
    default=False,
    help="Override config to reuse existing LogMap alignment (skip Step 1).",
)
parser.add_argument(
    "--reuse-prompts",
    action="store_true",
    default=False,
    help="Override config to reuse existing prompts (skip Steps 1 & 2). Implies --reuse-align.",
)
parser.add_argument(
    "--track",
    type=str,
    default=None,
    choices=["conference", "bioml", "anatomy", "knowledgegraph"],
    help="Track identifier (overrides config [alignmentTask] track).",
)
parser.add_argument(
    "--no-cache",
    action="store_true",
    default=False,
    help="Disable owlready2 quadstore caching (parse ontologies from scratch).",
)
args = parser.parse_args()

config_path = args.config
if not os.path.isfile(config_path):
    error(f"Configuration file not found: {config_path}")
    sys.exit(1)

with open(config_path, mode="rb") as fp:
    config = tomllib.load(fp)

info(f"Configuration loaded from: {config_path}")

# Apply CLI overrides for pipeline reuse flags
if args.reuse_prompts:
    config['pipeline']['build_oracle_prompts'] = 'reuse'
    config['pipeline']['align_ontologies'] = 'reuse'
    info("  [CLI override] build_oracle_prompts = 'reuse'")
    info("  [CLI override] align_ontologies = 'reuse'")
elif args.reuse_align:
    config['pipeline']['align_ontologies'] = 'reuse'
    info("  [CLI override] align_ontologies = 'reuse'")

# resolve track, CLI --track overrides config [alignmentTask] track
track = args.track or config.get('alignmentTask', {}).get('track', None)
if track:
    info(f"  Track: {track}")



# TIMING AND LOGGING:

# record the pipeline start time (also allows timestamp for filenames)
pipeline_start_time = time.time()
run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# determine output directory for logs and results
results_dir = config['outputs']['logmapllm_output_dirpath']
os.makedirs(results_dir, exist_ok=True)

# start the tee-logger: all subsequent stdout goes to both terminal and log file
log_filename = f"pipeline_log_{run_timestamp}.txt"
log_filepath = os.path.join(results_dir, log_filename)
original_stdout = sys.stdout
tee = TeeWriter(log_filepath, original_stdout)
sys.stdout = tee

print()
info(f"Pipeline log file: {log_filepath}")

# initialise timing accumulators (None = step not executed)
timing = {
    'step1_alignment': None,
    'step2_ontology_loading': None,
    'step2_prompt_building': None,
    'step3_oracle_consultation': None,
    'step4_refinement': None,
    'step5_evaluation': None,
}

# initialise containers for oracle params and evaluation results
oracle_params = {}
eval_results = None
eval_output_text = ""
n_consultations = 0
m_ask_df_ext = None


# %% [markdown]
# Build JVM classpath and JVM options

# %% [markdown]
# TODO: when LogMap-LLM is a package, we'll want to discover and set LogMap the dirpath automatically somehow

# %%
# TODO: decide the best way to set the logmap_dirpath
#logmap_dirpath = '/Users/davidherron/research/logmap-20251230/'
logmap_dirpath = os.path.join(os.getcwd(), 'logmap')

# path to main LogMap jar file
logmap_jar = os.path.join(logmap_dirpath, 'logmap-matcher-4.0.jar')
jpype.addClassPath(logmap_jar)

# path to LogMap dependency jar files
logmap_dep = os.path.join(logmap_dirpath, 'java-dependencies/*')
jpype.addClassPath(logmap_dep)

# LogMap jvm options
jvmOptions = [
    "-Xms500M", 
    "-Xmx25G",
    "-DentityExpansionLimit=10000000",
    "--add-opens=java.base/java.lang=ALL-UNNAMED"
]

# %% [markdown]
# Check if a JVM (Java Virtual Machine) is running

# %%
if jpype.isJVMStarted():
    error("JPype JVM running unexpectedly; version: " + str(jpype.getJVMVersion()))
    print()
    error('LogMap-LLM aborting prematurely')
    print()
    print('LogMap-LLM session ending')
    print()
    raise RuntimeError('Unexpected system condition encountered')

# %% [markdown]
# Start a JVM

# %%
if not jpype.isJVMStarted():
    jpype.startJVM(*jvmOptions)

# %% [markdown]
# Confirm a JVM is running

# %%
if not jpype.isJVMStarted():
    error('JPype JVM not running')
    print()
    error('LogMap-LLM aborting prematurely')
    print()
    print('LogMap-LLM session ending')
    print()
    raise RuntimeError('Unexpected system condition encountered')


# %% [markdown]
# ---
# 
# Now that we have imported JPype and started a JVM, we can import and call Java classes.
# 
# ---

# %% [markdown]
# Java imports for basic LogMap usage

# %%
from uk.ac.ox.krr.logmap2 import LogMapLLM_Interface

# %% [markdown]
# Python imports that contain Java imports

# %%
import bridging as br

# %% [markdown]
# Prepare the filepaths of the source and target ontologies the way LogMap expects 

# %%
task_name = config['alignmentTask']['task_name']
onto_src_filepath = config['alignmentTask']['onto_source_filepath']
onto_tgt_filepath = config['alignmentTask']['onto_target_filepath']
onto_src_filepath_logmap = "file:" + config['alignmentTask']['onto_source_filepath']
onto_tgt_filepath_logmap = "file:" + config['alignmentTask']['onto_target_filepath']

# %% [markdown]
# Instantiate a LogMapLLM interface to LogMap for the specified alignment task

# %%
logmap2_LogMapLLM_Interface = LogMapLLM_Interface(onto_src_filepath_logmap, 
                                                  onto_tgt_filepath_logmap, 
                                                  task_name)

# %% [markdown]
# Configure the LogMapLLM interface to LogMap for the initial alignment task

# %%
# boolean: True = generate extended m_ask, False = generate standard m_ask
#generate_extended_m_ask = False
generate_extended_m_ask = config['alignmentTask']['generate_extended_mappings_to_ask_oracle']
logmap2_LogMapLLM_Interface.setExtendedQuestions4LLM(generate_extended_m_ask)

# %%
# Set dirpath where LogMap should look for its configuration file parameters.txt
logmap_parameters_dirpath = config['alignmentTask']['logmap_parameters_dirpath']
# If the user has configured a dirpath, we use that. Otherwise, we use the dirpath
# for LogMap itself, which should contain a parameters.txt file.
if logmap_parameters_dirpath == "" or logmap_parameters_dirpath is None:
    logmap_parameters_dirpath = logmap_dirpath
# Ensure this particular dirpath ends with a directory separator character. Without 
# it, LogMap can't find its parameters.txt file and reverts to its default parameter
# settings. And it writes a message to stdout telling us about that. LogMap carries
# on without aborting, but this scenario prevents the user from configuring LogMap,
# and the error message clutters LogMap-LLM's output.
if not logmap_parameters_dirpath.endswith(os.sep):
    logmap_parameters_dirpath = logmap_parameters_dirpath + os.sep
logmap2_LogMapLLM_Interface.setPathToLogMapParameters(logmap_parameters_dirpath)

# %%
# set a dir path into which LogMap will save its 
# (an empty string means 'do not save any output files')

#logmap_outputs_dir_path = '/Users/dave/research/logmap-usage/mappings1'
#logmap_outputs_dir_path = ""

logmap_outputs_dir_path = config['outputs']['logmap_initial_alignment_output_dirpath']
logmap2_LogMapLLM_Interface.setPathForOutputMappings(logmap_outputs_dir_path)

# %% [markdown]
# ---
# 
# ## Begin LogMap-LLM session dialog with the user
# 
# ---

# %%

print()
info('LogMap-LLM session beginning')
print()
info(f'Alignment task name: {task_name}')
print()
print('Source ontology:')
print(onto_src_filepath)
print()
print('Target ontology:')
print(onto_tgt_filepath)


# %% [markdown]
# ---
# 
# ## pipeline step 1: Align Ontologies
# 
# ---

# %%
print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
step('Step 1: Align ontologies and obtain mappings to ask an Oracle')
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

step1_start = time.time()

if config['pipeline']['align_ontologies'] == 'align':
    # perform an initial alignment so we can get a fresh m_ask
    info("Performing fresh initial LogMap alignment ...")
    print()
    logmap2_LogMapLLM_Interface.performAlignment()
    success("Initial alignment complete")
    mappings = logmap2_LogMapLLM_Interface.getLogMapMappings()
    print()
    info(f'Number of mappings in initial alignment: {len(mappings)}')
    m_ask_java = logmap2_LogMapLLM_Interface.getLogMapMappingsForLLM()
    m_ask_df = br.java_mappings_2_python(m_ask_java)
    # Note: we don't need to save m_ask_df to a file because LogMap
    # does that automatically, as part of performing an alignment
    timing['step1_alignment'] = time.time() - step1_start
elif config['pipeline']['align_ontologies'] == 'reuse':
    # bypass an initial alignment and reuse an existing m_ask
    # saved to a file in an alignment conducted previously
    info("Reusing existing initial LogMap alignment ...")
    print()
    filename = task_name + '-logmap_mappings.txt'
    filepath = os.path.join(logmap_outputs_dir_path, filename)
    mappings = pd.read_csv(filepath, sep='|', header=None)
    info(f'Number of mappings in initial alignment: {len(mappings)}')
    print()
    filename = task_name + '-logmap_mappings_to_ask_oracle_user_llm.txt'
    print('Loading mappings to ask an Oracle from file:')
    print(filename)
    filepath = os.path.join(logmap_outputs_dir_path, filename)
    # Handle empty M_ask: LogMap may produce no uncertain mappings for
    # small ontology pairs (common in Conference track).
    if os.path.getsize(filepath) == 0:
        warning("M_ask file is empty — LogMap found no uncertain mappings.")
        warning("All LogMap mappings are high-confidence for this pair.")
        m_ask_df = pd.DataFrame(columns=br.get_m_ask_column_names())
    else:
        m_ask_df = pd.read_csv(filepath, sep='|', header=None)
        m_ask_df.columns = br.get_m_ask_column_names()
    timing['step1_alignment'] = 'reused'
elif config['pipeline']['align_ontologies'] == 'bypass':
    info('Bypassing initial LogMap alignment')
    m_ask_df = None
    timing['step1_alignment'] = 'bypassed'
else:
    raise ValueError(f"Value for align_ontologies not recognised: {config['pipeline']['align_ontologies']}")

if m_ask_df is not None:
    print()
    info(f"Number of mappings to ask an Oracle: {len(m_ask_df)}")

# Guard: if M_ask is empty (no uncertain mappings), skip steps 2-4
# this is common for Conference track pairs where LogMap is confident
# about all its mappings... We still proceed to Step 5 (evaluation)
empty_m_ask = m_ask_df is not None and len(m_ask_df) == 0

if empty_m_ask:
    print()
    warning("M_ask is empty - no mappings to consult the Oracle about.")
    warning("Skipping Steps 2-4 (prompt building, oracle consultation, refinement).")
    warning("Proceeding to evaluation of the unrefined LogMap alignment.")
    print()
    # Override pipeline config to bypass Steps 2-4 cleanly using
    # the existing bypass mechanism in each step.
    config['pipeline']['build_oracle_prompts'] = 'bypass'
    config['pipeline']['consult_oracle'] = 'bypass'
    config['pipeline']['refine_alignment'] = 'bypass'

# %% [markdown]
# ---
# 
# ## pipeline step 2: Build Oracle Prompts
# 
# ---

# %%
print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
step('Step 2: Build user prompts for mappings to ask an LLM Oracle')
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

oupt_name = config['oracle']['oracle_user_prompt_template_name']
# JD. bidirectional mode is neccesary for for subs prompts
bidirectional_mode = opb.is_bidirectional_template(oupt_name) 

if config['pipeline']['build_oracle_prompts'] == 'build':

    # JD. this is a bit botched, but we spawn a new subprocess
    # to run build oracle prompts and k-shot examples, etc.
    # save these to disk, then access them later.
    # TODO: build a better solution?

    print()
    print()
    step('--------------------------------------------------------------')
    step('EXECUTING PHASE TWO: DELEGATING TO SUBPROCESS DUE TO OWLREADY2')
    step('--------------------------------------------------------------')
    warn('THIS MAY AFFECT TIMING DATA REPORTING')
    step('--------------------------------------------------------------')
    step('                        STARTING                              ')
    step('--------------------------------------------------------------')
    build_script_env = os.environ.copy()
    step2_start = time.time()
    dedicated_build_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'dedicated_stage_two_script.py'
    )
    build_script_cmd = [
        sys.executable, '-u',
        dedicated_build_script,
        '--config', config_path
    ]
    # forward CLI overrides to the subprocess so it applies the
    # same config mutations as the parent process
    if args.reuse_align:
        build_script_cmd.append('--reuse-align')
    if args.reuse_prompts:
        build_script_cmd.append('--reuse-prompts')
    if track:
        build_script_cmd.extend(['--track', track])
    if args.no_cache:
        build_script_cmd.append('--no-cache')
    build_script_proc = subprocess.Popen(
        build_script_cmd,
        env=build_script_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Stream subprocess output line-by-line through tee (real-time in terminal + log)
    for line in build_script_proc.stdout:
        print(line, end='')

    build_script_proc.wait()

    timing['step2_prompt_building'] = time.time() - step2_start

    step('-------------------------------------------------------------')
    step('            FINISHING DELEGATION PROCEDURE                   ')
    step('-------------------------------------------------------------')

    if build_script_proc.returncode != 0:
        error(f'Step 2 subprocess exited with code {build_script_proc.returncode}')
        error('Prompt building failed. Cannot continue.')
        sys.stdout = original_stdout
        tee.close()
        sys.exit(1)
    step('-------------------------------------------------------------')
    success('                       COMPLETE                              ')
    step('-------------------------------------------------------------')
    print()
    print()
    info('Reusing prepared LLM Oracle user prompts from subprocess ....')
    print()
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
    print('Loading LLM Oracle user prompts from file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'r') as fp:
        m_ask_oracle_user_prompts = json.load(fp)
    success("COMPLETE.")

elif config['pipeline']['build_oracle_prompts'] == 'reuse':
    info('Reusing existing LLM Oracle user prompts ...')
    print()
    # reuse oracle user prompts created previously and saved in a file on disk
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
    print('Loading LLM Oracle user prompts from file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'r') as fp:
        m_ask_oracle_user_prompts = json.load(fp)
    timing['step2_prompt_building'] = 'reused'
elif config['pipeline']['build_oracle_prompts'] == 'bypass':
    info('Bypassing use of LLM Oracle user prompts')
    m_ask_oracle_user_prompts = None
    timing['step2_prompt_building'] = 'bypassed'
else:
    raise ValueError(f"Value for build_oracle_prompts not recognised: {config['pipeline']['build_oracle_prompts']}")

if m_ask_oracle_user_prompts is not None:
    print()
    info(f"Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}")

    # validate that prompt key format matches consultation mode
    # standard prompts use 'src|tgt' keys; bidirectional prompts use
    # 'src|tgt|direction' keys; a mismatch indicates the loaded JSON
    # was built with a different template type than currently selected...
    if m_ask_oracle_user_prompts:
        sample_key = next(iter(m_ask_oracle_user_prompts))
        keys_have_direction = sample_key.count(PAIRS_SEPARATOR) >= 2
        if bidirectional_mode and not keys_have_direction:
            raise ValueError(
                f"Bidirectional template '{oupt_name}' selected but loaded prompts "
                f"lack direction keys (e.g. '...|forward'). The prompt JSON was "
                f"likely built with a non-bidirectional template. Rebuild prompts "
                f"or select a matching template."
            )
        elif not bidirectional_mode and keys_have_direction:
            raise ValueError(
                f"Standard template '{oupt_name}' selected but loaded prompts "
                f"have direction keys (e.g. '...|forward'). The prompt JSON was "
                f"likely built with a bidirectional template. Rebuild prompts "
                f"or select a matching template."
            )

    print()


# %% [markdown]
# ---
# 
# ## pipeline step 3: Consult Oracle
# 
# ---

# %%
print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
step("Step 3: Consult Oracle for mappings to ask")
step('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

api_key = config['oracle']['openrouter_apikey']
model_name = config['oracle']['openrouter_model_name']
base_url = config['oracle'].get('base_url', None)
enable_thinking = config['oracle'].get('enable_thinking', None)
interaction_style = config['oracle'].get('interaction_style', None)
# new kwargs
supports_chat_template_kwargs = config['oracle'].get('supports_chat_template_kwargs', None)
max_completion_tokens = config['oracle'].get('max_completion_tokens', None)
failure_tolerance = config['oracle'].get('failure_tolerance', None)
temperature = config['oracle'].get('temperature', None)
top_p = config['oracle'].get('top_p', None)
reasoning_effort = config['oracle'].get('reasoning_effort', None)

# few-shot configuration
few_shot_k = config['oracle'].get('few_shot_k', 0)
# static examples (hard/random)
few_shot_examples = None
# per-query pool (hard-similar <-- this is still WiP feature)
few_shot_pool = None
# M_ask pair labels for hard-similar
pair_labels = None 

negative_strategy = config['oracle'].get('few_shot_negative_strategy', 'hard')

if few_shot_k > 0:
    dirpath = config['outputs']['logmapllm_output_dirpath']

    if negative_strategy == 'hard-similar':
        # load pool for per-query selection
        pool_json_path = os.path.join(dirpath, f"{task_name}-{oupt_name}-few_shot_pool.json")
        pool_embs_path = os.path.join(dirpath, f"{task_name}-{oupt_name}-few_shot_pool_embs.npy")
        labels_path = os.path.join(dirpath, f"{task_name}-{oupt_name}-m_ask_pair_labels.json")

        if os.path.isfile(pool_json_path) and os.path.isfile(pool_embs_path):
            import numpy as np
            from few_shot_examples import FewShotPool

            with open(pool_json_path, 'r') as fp:
                pool_candidates = json.load(fp)
            pool_embeddings = np.load(pool_embs_path)

            # initialise SiblingSelector for query embedding (same model as step 2)
            from sibling_retrieval import SiblingSelector, DEFAULT_GENERAL_MODEL
            if track in ('conference', 'knowledgegraph'):
                pool_selector = SiblingSelector(
                    model_name_or_path=DEFAULT_GENERAL_MODEL, pooling="mean")
            else:
                pool_selector = SiblingSelector()

            few_shot_pool = FewShotPool(
                candidates=pool_candidates,
                embeddings=pool_embeddings,
                sibling_selector=pool_selector,
                k=few_shot_k,
                seed=config['oracle'].get('few_shot_seed', 42),
                answer_format=answer_format,
            )
            info(f"Loaded few-shot pool: {len(pool_candidates)} candidates")

            # Load M_ask pair labels
            if os.path.isfile(labels_path):
                with open(labels_path, 'r') as fp:
                    pair_labels = json.load(fp)
                info(f"Loaded M_ask pair labels ({len(pair_labels)} pairs)")
            else:
                warn(f"M_ask pair labels not found: {labels_path}")
                warn("hard-similar selection will be unable to compute query embeddings")
                few_shot_pool = None
                few_shot_k = 0
        else:
            warn(f"hard-similar pool files not found; falling back to zero-shot")
            few_shot_k = 0

    else:
        # static loading path (hard / random)
        fs_filename = task_name + '-' + oupt_name + '-few_shot_examples.json'
        fs_filepath = os.path.join(dirpath, fs_filename)
        if os.path.isfile(fs_filepath):
            with open(fs_filepath, 'r') as fp:
                few_shot_examples = [tuple(pair) for pair in json.load(fp)]
            info(f"Loaded {len(few_shot_examples)} few-shot examples from {fs_filename}")
        else:
            warn(f"few_shot_k={few_shot_k} but examples file not found: {fs_filename}")
            warn("Falling back to zero-shot")
            few_shot_k = 0

# resolve the developer prompt from config via the registry
dev_prompt_name = config['oracle'].get('oracle_dev_prompt_template_name', 'class_equivalence')
developer_prompt_text = dp.get_developer_prompt(dev_prompt_name)

# apply answer format adaptation to the developer prompt
# user prompts are already built with the correct format (set in dedicated_stage_two_script.py during step 2)
# TODO: apply a better solution to this, see developer_prompts.py
answer_format = config['oracle'].get('answer_format', 'true_false')
developer_prompt_text = dp.adapt_developer_prompt(developer_prompt_text, answer_format)

# build entity-type-aware developer prompt map; when property or instance developer prompts are configured, 
# the map provides per-entity-type overrides so that e.g. property candidates receive DEV_PROMPT_PROPERTY_EQUIVALENCE 
# instead of the class developer prompt. 
# when not configured (Bio-ML, Anatomy), the map is None and all consultations use the single developer_prompt_text
developer_prompt_map = None
prop_dev_name = config['oracle'].get('oracle_property_dev_prompt_template_name', None)
inst_dev_name = config['oracle'].get('oracle_instance_dev_prompt_template_name', None)

if prop_dev_name or inst_dev_name:
    developer_prompt_map = {}
    # cls entries use the primary developer prompt (already resolved above)
    if developer_prompt_text is not None:
        developer_prompt_map["CLS"] = developer_prompt_text
    if prop_dev_name:
        prop_text = dp.get_developer_prompt(prop_dev_name)
        prop_text = dp.adapt_developer_prompt(prop_text, answer_format)
        developer_prompt_map["OPROP"] = prop_text
        developer_prompt_map["DPROP"] = prop_text
        info(f"  Property developer prompt: {prop_dev_name}")
    if inst_dev_name:
        inst_text = dp.get_developer_prompt(inst_dev_name)
        inst_text = dp.adapt_developer_prompt(inst_text, answer_format)
        developer_prompt_map["INST"] = inst_text
        info(f"  Instance developer prompt: {inst_dev_name}")

# we default to 24, since we're using 24 cores (1/2 the available threads)
# though, this is configurable through config -- TODO: consider optimal
# max_workers for parallelised experiments
max_workers = config['oracle'].get('max_workers', 24)

local_oracle_predictions_filepath = None
local_oracle_predictions_dirpath = None

if config['pipeline']['consult_oracle'] == 'consult':
    model_name = config['oracle']['openrouter_model_name']
    info(f'Consulting LLM Oracle {model_name}')
    if bidirectional_mode:
        print('with bidirectional subsumption prompts for mappings to ask ...')
    else:
        print('with user prompts for mappings to ask ...')
    print()
    step3_start = time.time()
    if bidirectional_mode:
        # TODO: probably we can modify this to unpack a dict
        m_ask_df_ext, oracle_params = oc.consult_oracle_bidirectional(
            m_ask_oracle_user_prompts,
            api_key,
            model_name,
            max_workers,
            m_ask_df,
            base_url=base_url,
            enable_thinking=enable_thinking,
            interaction_style=interaction_style,
            supports_chat_template_kwargs=supports_chat_template_kwargs,
            developer_prompt_text=developer_prompt_text,
            developer_prompt_map=developer_prompt_map,
            max_completion_tokens=max_completion_tokens,
            failure_tolerance=failure_tolerance,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            few_shot_examples=few_shot_examples,
            answer_format=answer_format,
            few_shot_pool=few_shot_pool,
            pair_labels=pair_labels,
        )
    else:
        m_ask_df_ext, oracle_params = oc.consult_oracle_for_mappings_to_ask(
            m_ask_oracle_user_prompts,
            api_key,
            model_name,
            max_workers,
            m_ask_df,
            base_url=base_url,
            enable_thinking=enable_thinking,
            interaction_style=interaction_style,
            supports_chat_template_kwargs=supports_chat_template_kwargs,
            developer_prompt_text=developer_prompt_text,
            developer_prompt_map=developer_prompt_map,
            max_completion_tokens=max_completion_tokens,
            failure_tolerance=failure_tolerance,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            few_shot_examples=few_shot_examples,
            answer_format=answer_format,
            few_shot_pool=few_shot_pool,
            pair_labels=pair_labels,
        )
    
    timing['step3_oracle_consultation'] = time.time() - step3_start
    
    if m_ask_oracle_user_prompts is not None:
        if bidirectional_mode:
            # each candidate generates 2 prompts (forward + reverse);
            # report count as candidates, not individual LLM calls.
            n_consultations = len(m_ask_oracle_user_prompts) // 2
        else:
            n_consultations = len(m_ask_oracle_user_prompts)

elif config['pipeline']['consult_oracle'] == 'reuse':
    info('Reusing existing LLM Oracle predictions')
    # reuse oracle predictions created previously and saved in a file on disk
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_with_oracle_predictions.csv'
    print('Loading LLM Oracle predictions for the mappings_to_ask from file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    m_ask_df_ext = pd.read_csv(filepath)

    # restore boolean types lost during CSV round-trip
    for col in ['Oracle_prediction', 'Oracle_prediction_forward', 'Oracle_prediction_reverse']:
        if col in m_ask_df_ext.columns:
            m_ask_df_ext[col] = normalise_prediction_column(m_ask_df_ext[col])

    timing['step3_oracle_consultation'] = 'reused'

elif config['pipeline']['consult_oracle'] == 'local':
    local_oracle_predictions_dirpath = config['oracle']['local_oracle_predictions_dirpath']
    info('Local Oracle prediction .csv file(s) will be loaded from directory:')
    print(local_oracle_predictions_dirpath)
    m_ask_df_ext = None
    timing['step3_oracle_consultation'] = 'local'
elif config['pipeline']['consult_oracle'] == 'bypass':
    info('Bypassing Oracle consultations')
    m_ask_df_ext = None
    timing['step3_oracle_consultation'] = 'bypassed'
else:
    raise ValueError(f"Value for consult_oracle not recognised: {config['pipeline']['consult_oracle']}")


if m_ask_df_ext is not None:
    preds = m_ask_df_ext['Oracle_prediction']
    nr_mappings = len(preds)
    nr_errors = sum(preds == 'error')
    nr_skipped = sum(preds == 'skipped')
    nr_completions = nr_mappings - nr_errors - nr_skipped
    nr_true = sum(preds == True)
    nr_false = sum(preds == False)
    width = len(str(nr_mappings))
    nr_true_s = str(nr_true).rjust(width)
    nr_false_s = str(nr_false).rjust(width)
    nr_errors_s = str(nr_errors).rjust(width)
    nr_skipped_s = str(nr_skipped).rjust(width)
    print()
    info(f"Number of mappings to ask an Oracle: {nr_mappings}")
    info(f"Number of LLM Oracle consultations : {nr_completions}")
    info(f"Number of mappings predicted True  : {nr_true_s}")
    info(f"Number of mappings predicted False : {nr_false_s}")
    if nr_errors > 0:
        warning(f"Number of consultation failures : {nr_errors_s}")
    else:
        print(f"Number of consultation failures : {nr_errors_s}")
    if nr_skipped > 0:
        warn(f"Number of mappings skipped : {nr_skipped_s}")
    print()

    # Additional reporting for bidirectional mode
    if bidirectional_mode and 'Oracle_prediction_forward' in m_ask_df_ext.columns:
        fwd_preds = m_ask_df_ext['Oracle_prediction_forward']
        rev_preds = m_ask_df_ext['Oracle_prediction_reverse']
        n_fwd_true = sum(fwd_preds == True)
        n_fwd_false = sum(fwd_preds == False)
        n_rev_true = sum(rev_preds == True)
        n_rev_false = sum(rev_preds == False)
        print("Bidirectional detail:")
        print(f"  Forward  (src SQSUBSETEQ tgt): {n_fwd_true} True, {n_fwd_false} False")
        print(f"  Reverse  (tgt SQSUBSETEQ src): {n_rev_true} True, {n_rev_false} False")
        print(f"  Aggregated (AND rule): {nr_true} equivalence, {nr_false} not equivalent")
        print()

if config['pipeline']['consult_oracle'] == 'consult' and m_ask_df_ext is not None:
    # save the extended m_ask dataframe (that contains the LLM Oracle predictions)
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_with_oracle_predictions.csv'
    print("Oracle predictions for 'mappings to ask' saved to file:")
    print(filename)
    filepath = os.path.join(dirpath, filename)
    m_ask_df_ext.to_csv(filepath)


# %% [markdown]
# ---
# 
# ## Pipeline step 4: Refine Alignment
# 
# ---

# %%
print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
step('Step 4: Refine alignment using Oracle mapping predictions')
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

logmap_outputs_dir_path = config['outputs']['logmap_refined_alignment_output_dirpath']
logmap2_LogMapLLM_Interface.setPathForOutputMappings(logmap_outputs_dir_path)

step4_start = time.time()

if config['pipeline']['refine_alignment'] == 'refine':

    # TODO: for the KG track, we need to re-test using LogMap since for re-alignment as we believe the prior issue has since been patched; 
    # i.e., previously we got a null pointer exception, but we suspect this MAY be due to:
    # 
    #   considerAdditionalFeedbackForInstanceMapping bug in CandidateOracleManager
    #
    # However, since OAEI still publish results for KG alignment with LogMap, I assume they use the following method 
    # since we reproduce their results exactly using this method for evaluation:
    #
    # KG track -- python-based refinement method (which bypasses LogMap's java-based refinement):
    #
    # logmap java refinement applies structural ontology repair logic that is only meaningful for class/property mappings
    # for instance mappings (the bulk of the KG track), it crashes with NPEs... This could be due to the fact that 
    # class hierarchy indices don't contain instance URIs. As such, (as a work-around) we produce the refined alignment directly:
    #
    #   refined = (initial - M_ask) UNION { m \in M_ask : oracle(m) = True }
    #

    if track == 'knowledgegraph' and m_ask_df_ext is not None:
        info("KG track: producing refined alignment in Python (bypassing LogMap Java refinement) ...")
        print()

        # load the initial alignment as a set of (src, tgt) pairs with metadata
        initial_align_dir = config['outputs']['logmap_initial_alignment_output_dirpath']
        initial_mappings_path = os.path.join(initial_align_dir, f"{task_name}-logmap_mappings.txt")
        initial_df = pd.read_csv(initial_mappings_path, sep='|', header=None)
        initial_df.columns = ['source_entity_uri', 'target_entity_uri', 'relation', 'confidence', 'entityType']

        # build set of M_ask pairs for fast lookup
        m_ask_pairs = set()
        for _, row in m_ask_df.iterrows():
            m_ask_pairs.add((row.iloc[0], row.iloc[1]))

        # keep non-M_ask mappings (high-confidence, not sent to oracle)
        refined_rows = []
        for _, row in initial_df.iterrows():
            pair = (row['source_entity_uri'], row['target_entity_uri'])
            if pair not in m_ask_pairs:
                refined_rows.append(row)

        # add M_ask mappings where oracle predicted true
        n_oracle_accepted = 0
        for _, row in m_ask_df_ext.iterrows():
            if row['Oracle_prediction'] == True:
                refined_rows.append(pd.Series({
                    'source_entity_uri': row['source_entity_uri'],
                    'target_entity_uri': row['target_entity_uri'],
                    'relation': row['relation'],
                    'confidence': row.get('Oracle_confidence', row.get('confidence', 1.0)),
                    'entityType': row['entityType'],
                }))
                n_oracle_accepted += 1

        refined_df = pd.DataFrame(refined_rows)

        # write as TSV (DeepOnto-compatible format for evaluation)
        os.makedirs(logmap_outputs_dir_path, exist_ok=True)
        refined_tsv_path = os.path.join(logmap_outputs_dir_path, f"{task_name}-logmap_mappings.tsv")
        eval_df = refined_df[['source_entity_uri', 'target_entity_uri', 'confidence']].copy()
        eval_df.columns = ['SrcEntity', 'TgtEntity', 'Score']
        eval_df.to_csv(refined_tsv_path, sep='\t', index=False)

        n_initial = len(initial_df)
        n_m_ask = len(m_ask_pairs)
        n_high_conf = n_initial - n_m_ask
        n_refined = len(refined_df)

        success("Python-based alignment refinement complete")
        print()
        info(f" Initial alignment mappings  : {n_initial}")
        info(f" High-confidence (kept)      : {n_high_conf}")
        info(f" M_ask sent to oracle        : {n_m_ask}")
        info(f" Oracle accepted (True)      : {n_oracle_accepted}")
        info(f" Refined alignment mappings  : {n_refined}")

        timing['step4_refinement'] = time.time() - step4_start

    elif m_ask_df_ext is not None:
        # refine the initial alignment using the m_ask Oracle predictions
        info("Refining initial LogMap alignment with LLM Oracle predictions ...")
        print()
        m_ask_oracle_preds_java = br.python_oracle_mapping_predictions_2_java(m_ask_df_ext)
        info(f'Number of mappings predicted True by Oracle given to LogMap: {len(m_ask_oracle_preds_java)} ')
        print()
        logmap2_LogMapLLM_Interface.performAlignmentWithLocalOracle(m_ask_oracle_preds_java)
        success("Alignment refinement complete")
        mappings_java = logmap2_LogMapLLM_Interface.getLogMapMappings()
        print()
        info(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')
        timing['step4_refinement'] = time.time() - step4_start
    elif local_oracle_predictions_dirpath is not None:
        # refine the initial alignment using local Oracle predictions for m_ask
        info("Refining initial LogMap alignment with local Oracle predictions ...")
        print()
        logmap2_LogMapLLM_Interface.performAlignmentWithLocalOracle(local_oracle_predictions_dirpath)
        success("Alignment complete")
        mappings_java = logmap2_LogMapLLM_Interface.getLogMapMappings()
        print()
        info(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')
        timing['step4_refinement'] = time.time() - step4_start
    else:
        warning('Step 4 bypassed due to Oracle consultation failures in Step 3')
        timing['step4_refinement'] = 'bypassed (no oracle predictions)'
elif config['pipeline']['refine_alignment'] == 'bypass':
    info('Bypassing alignment refinement')
    timing['step4_refinement'] = 'bypassed'
else:
    raise ValueError(f"Value for refine_alignment not recognised: {config['pipeline']['refine_alignment']}")


# %%

# %% [markdown]
# ---
# 
# ## Pipeline step 5: Evaluate Alignment (optional)
# 
# ---

# %%

eval_config = config.get('evaluation', {})
run_evaluation = eval_config.get('evaluate', False)

if run_evaluation:
    print()
    step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    step('Step 5: Evaluate alignment against reference')
    step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    ref_path = eval_config.get('reference_alignment_path', None)

    if ref_path is None or not os.path.isfile(ref_path):
        print()
        warning('Reference alignment not found, skipping evaluation.')
        if ref_path:
            print(f'Configured path: {ref_path}')
        print('Set [evaluation] reference_alignment_path in your config.')
    else:
        step5_start = time.time()

        # find system mappings TSV
        refined_dir = config['outputs']['logmap_refined_alignment_output_dirpath']
        system_mappings_path = os.path.join(refined_dir, f"{task_name}-logmap_mappings.tsv")

        # find oracle predictions CSV
        oracle_predictions_path = None
        llm_output_dir = config['outputs']['logmapllm_output_dirpath']
        for fname in os.listdir(llm_output_dir):
            if 'predictions' in fname and fname.endswith('.csv'):
                oracle_predictions_path = os.path.join(llm_output_dir, fname)
                break

        # fallback: if refinement was bypassed (e.g. empty M_ask), use the initial alignment file instead
        # convert pipe-delimited format to TSV so the evaluator can parse it
        if not os.path.isfile(system_mappings_path):
            initial_dir = config['outputs']['logmap_initial_alignment_output_dirpath']
            initial_mappings = os.path.join(initial_dir, f"{task_name}-logmap_mappings.txt")
            if os.path.isfile(initial_mappings):
                warning("Refined alignment not found — evaluating initial LogMap alignment.")
                # convert pipe-delimited initial alignment to TSV
                fallback_tsv = os.path.join(llm_output_dir, f"{task_name}-initial_alignment_for_eval.tsv")
                try:
                    init_df = pd.read_csv(initial_mappings, sep='|', header=None)
                    if len(init_df.columns) >= 2:
                        eval_df = init_df.iloc[:, :2]
                        eval_df.columns = ['SrcEntity', 'TgtEntity']
                        eval_df['Score'] = 1.0
                        eval_df.to_csv(fallback_tsv, sep='\t', index=False)
                        system_mappings_path = fallback_tsv
                        info(f"Converted initial alignment to TSV ({len(eval_df)} mappings)")
                    else:
                        warning("Initial alignment has unexpected format — skipping evaluation.")
                except Exception as e:
                    warning(f"Failed to convert initial alignment: {e}")
            else:
                warning("Neither refined nor initial alignment found — skipping evaluation.")

        # optional paths for semi-supervised and ranking evaluation
        train_ref_path = eval_config.get('train_alignment_path', None)
        if train_ref_path and not os.path.isfile(train_ref_path):
            train_ref_path = None

        test_cands_path = eval_config.get('test_cands_path', None)
        if test_cands_path and not os.path.isfile(test_cands_path):
            test_cands_path = None

        # metrics selection
        metrics = eval_config.get('metrics', ['global', 'oracle'])
        if isinstance(metrics, str):
            metrics = [m.strip() for m in metrics.split(',')]

        eval_json_path = os.path.join(llm_output_dir, 'evaluation_results.json')

        # JD. spawning a subprocess for evaluation is neccesary (again), since DeepOnto requires
        # use of a JVM, and we need to avoid JPype/JVM conflicts / issues (runs evaluate.py)
        # not the nicest of solutions, but it works for now.

        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate.py')
        eval_cmd = [
            sys.executable,
            eval_script,
            '--system-mappings', system_mappings_path,
            '--reference', ref_path,
            '--task-name', task_name,
            '--metrics', *metrics,
            '--output-json', eval_json_path,
        ]

        if oracle_predictions_path:
            eval_cmd.extend(['--oracle-predictions', oracle_predictions_path])
        if train_ref_path:
            eval_cmd.extend(['--train-reference', train_ref_path])
        if test_cands_path:
            eval_cmd.extend(['--test-cands', test_cands_path])

        # pass initial alignment for Conference M1/M2 stratification
        if track == 'conference':
            initial_align_dir = config['outputs']['logmap_initial_alignment_output_dirpath']
            initial_align_file = os.path.join(initial_align_dir, f"{task_name}-logmap_mappings.txt")
            if os.path.isfile(initial_align_file):
                eval_cmd.extend(['--initial-alignment', initial_align_file])

        # optional: force custom evaluation (skip DeepOnto even if installed)
        if eval_config.get('force_custom_eval', False):
            eval_cmd.append('--no-deeponto')

        # forward KG task flag for partial gold standard evaluation
        is_kg = eval_config.get('is_kg_task', False)
        if is_kg:
            eval_cmd.append('--kg-task')

        # forward track hint for auto-discovery
        if track:
            eval_cmd.extend(['--track', track])

        # JVM memory for DeepOnto (default 8g) -- note, we can probably extend
        # but i'm not sure that its neccesarily optimal to allocate a larger pool of memory to the JVM
        # for a basic process (eval is executed quickly)
        jvm_memory = eval_config.get('jvm_memory', '8g')
        eval_cmd.extend(['--jvm-memory', str(jvm_memory)])

        eval_env = os.environ.copy()
        for var_name in ('JAVA_MEMORY', 'DEEPONTO_JVM_MEMORY', 'JVM_MEMORY'):
            eval_env[var_name] = str(jvm_memory)

        print()
        # capture subprocess output so it appears in both terminal and log
        result = subprocess.run(
            eval_cmd,
            env=eval_env,
            input=f"{jvm_memory}\n",
            text=True,
            capture_output=True,
        )

        # print captured output (goes through tee to both terminal and log)
        if result.stdout:
            print(result.stdout, end='')
            eval_output_text = result.stdout
        if result.stderr:
            print(result.stderr, end='', file=original_stdout)

        if result.returncode != 0:
            warning(f'Evaluation subprocess exited with code {result.returncode}')

        timing['step5_evaluation'] = time.time() - step5_start

        # load evaluation results JSON if it was written
        if os.path.isfile(eval_json_path):
            with open(eval_json_path, 'r') as f:
                eval_results = json.load(f)

# pipeline complete -- compute total time

pipeline_end_time = time.time()
pipeline_total_time = pipeline_end_time - pipeline_start_time


# step 6: display metrics and timings

print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
step('Timing Metrics')
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

def format_timing_value(val):
    """format a timing value: number -> human-readable duration, string -> N/A label"""
    if isinstance(val, (int, float)):
        return format_duration(val)
    elif isinstance(val, str):
        return f"N/A ({val})"
    else:
        return "N/A"

print(f" Step 1  LogMap initial alignment   : {format_timing_value(timing['step1_alignment'])}")
print(f" Step 2  Ontology loading + prompts : {format_timing_value(timing['step2_prompt_building'])}")
print(f" Step 3  LLM Oracle consultation    : {format_timing_value(timing['step3_oracle_consultation'])}")

# compute average consultation time
avg_consultation_str = "N/A"
if isinstance(timing['step3_oracle_consultation'], (int, float)) and n_consultations > 0:
    avg_time = timing['step3_oracle_consultation'] / n_consultations
    if bidirectional_mode:
        avg_consultation_str = f"{avg_time:.3f}s ({n_consultations} candidates, {n_consultations * 2} LLM calls)"
    else:
        avg_consultation_str = f"{avg_time:.3f}s ({n_consultations} consultations)"

print(f" Average per consultation : {avg_consultation_str}")

print(f" Step 4  LogMap refinement      : {format_timing_value(timing['step4_refinement'])}")
print(f" Step 5  Evaluation             : {format_timing_value(timing['step5_evaluation'])}")
print()
print(f" Total pipeline wall-clock time : {format_duration(pipeline_total_time)}")

# step 7 - display experimental parameters

print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
step('Experimental Parameters')
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

endpoint_type = classify_endpoint(base_url)

print(f" Task name           : {task_name}")
print(f" Config file         : {config_path}")
print(f" Run timestamp (UTC) : {run_timestamp}")
print()
print(f" LLM model     : {model_name}")
print(f" Endpoint type : {endpoint_type}")
print(f" Base URL      : {base_url if base_url else 'N/A (OpenRouter default)'}")
print()

# report oracle parameters
if oracle_params:
    print(f" Interaction style     : {oracle_params.get('interaction_style', 'N/A')}")
    print(f" Temperature           : {oracle_params.get('temperature', 'N/A')}")
    print(f" Top-p                 : {oracle_params.get('top_p', 'N/A')}")
    print(f" Reasoning effort      : {oracle_params.get('reasoning_effort', 'N/A')}")
    print(f" Max completion tokens : {oracle_params.get('max_completion_tokens', 'N/A')}")
    print(f" Enable thinking       : {oracle_params.get('enable_thinking', 'N/A')}")
    print(f" Chat template kwargs  : {oracle_params.get('supports_chat_template_kwargs', 'N/A')}")
    print(f" Max workers (threads) : {oracle_params.get('max_workers', max_workers)}")
    print(f" Failure tolerance     : {oracle_params.get('failure_tolerance', 'N/A')}")
    # response format
    rf = oracle_params.get('response_format', 'N/A')
    rf_name = rf.__name__ if hasattr(rf, '__name__') else str(rf)
    print(f" Response format : {rf_name}")
else:
    # If oracle was not consulted, show config values
    print(f" Interaction style    : {interaction_style if interaction_style else 'auto'}")
    print(f" Enable thinking      : {enable_thinking if enable_thinking is not None else 'N/A'}")
    print(f" Chat template kwargs : {supports_chat_template_kwargs if supports_chat_template_kwargs is not None else 'auto'}")
    print(f" (Oracle not consulted -- detailed params unavailable)")

print()
print(f" Developer prompt name : {dev_prompt_name}")
print(f" Developer prompt set  : {developer_prompt_text is not None}")
if developer_prompt_text is not None:
    # Truncate long prompts for display (full text goes to results file)
    display_text = developer_prompt_text[:120] + "..." if len(developer_prompt_text) > 120 else developer_prompt_text
    print(f" Developer prompt text : {display_text}")
if developer_prompt_map:
    print(f" Developer prompt map : {list(developer_prompt_map.keys())}")
    for etype, text in developer_prompt_map.items():
        if etype != "CLS":
            display = text[:80] + "..." if len(text) > 80 else text
            print(f" {etype:6s} -> {display}")
print()
print(f" User prompt template : {oupt_name}")
print(f" Answer format : {answer_format}")

# show an example prompt if available
if m_ask_oracle_user_prompts:
    first_key = next(iter(m_ask_oracle_user_prompts))
    first_prompt = m_ask_oracle_user_prompts[first_key]
    print()
    print(f" Example prompt (first mapping):")
    print(f" Mapping: {first_key}")
    print(f" ---")
    for line in first_prompt.strip().split('\n'):
        print(f" {line}")
    print(f" ---")

# token usage summary
if m_ask_df_ext is not None and 'Oracle_input_tokens' in m_ask_df_ext.columns:
    total_input_tokens = m_ask_df_ext['Oracle_input_tokens'].sum()
    total_output_tokens = m_ask_df_ext['Oracle_output_tokens'].sum()
    if not pd.isna(total_input_tokens):
        print()
        print(f" Total input tokens  : {int(total_input_tokens):,}")
        print(f" Total output tokens : {int(total_output_tokens):,}")
        print(f" Total tokens        : {int(total_input_tokens + total_output_tokens):,}")

# gpu info
print()
print(f"  GPU info:")
gpu_info = get_gpu_info()
for line in gpu_info.split('\n'):
    print(f"    {line}")

# env data/settings
print()
print(f" Python version : {platform.python_version()}")
try:
    import owlready2
    print(f"  owlready2 version : {owlready2.__version__}")
except (ImportError, AttributeError):
    print(f"  owlready2 version : N/A")
print(f"  Platform : {platform.platform()}")

# step 8 -- write expr_run_results.txt to disk

print()
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
step('Writing experiment results to disk')
step('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

results_filename = f"expr_run_results_{run_timestamp}.txt"
results_filepath = os.path.join(results_dir, results_filename)

with open(results_filepath, 'w', encoding='utf-8') as rf:
    rf.write("=" * 72 + "\n")
    rf.write("LogMap-LLM Experiment Run Results\n")
    rf.write("=" * 72 + "\n")
    rf.write(f"Run timestamp (UTC): {run_timestamp}\n")
    rf.write(f"Config file: {config_path}\n")
    rf.write(f"Task: {task_name}\n")
    rf.write("\n")

    # evaluation results
    rf.write("-" * 72 + "\n")
    rf.write("Evaluation Results\n")
    rf.write("-" * 72 + "\n")
    if eval_results:
        rf.write(json.dumps(eval_results, indent=2))
        rf.write("\n")
    elif eval_output_text:
        rf.write(eval_output_text)
        rf.write("\n")
    else:
        rf.write("No evaluation results available.\n")
    rf.write("\n")

    # timing metrics
    rf.write("-" * 72 + "\n")
    rf.write("Timing Metrics\n")
    rf.write("-" * 72 + "\n")
    rf.write(f"Step 1  LogMap initial alignment   : {format_timing_value(timing['step1_alignment'])}\n")
    rf.write(f"Step 2  Ontology loading + prompts : {format_timing_value(timing['step2_prompt_building'])}\n")
    rf.write(f"Step 3  LLM Oracle consultation    : {format_timing_value(timing['step3_oracle_consultation'])}\n")
    rf.write(f"        Average per consultation   : {avg_consultation_str}\n")
    rf.write(f"Step 4  LogMap refinement          : {format_timing_value(timing['step4_refinement'])}\n")
    rf.write(f"Step 5  Evaluation                 : {format_timing_value(timing['step5_evaluation'])}\n")
    rf.write(f"Total pipeline wall-clock time     : {format_duration(pipeline_total_time)}\n")
    rf.write("\n")

    # experimental parameters
    rf.write("-" * 72 + "\n")
    rf.write("Experimental Parameters\n")
    rf.write("-" * 72 + "\n")
    rf.write(f"Task name     : {task_name}\n")
    rf.write(f"Config file   : {config_path}\n")
    rf.write(f"LLM model     : {model_name}\n")
    rf.write(f"Endpoint type : {endpoint_type}\n")
    rf.write(f"Base URL      : {base_url if base_url else 'N/A (OpenRouter default)'}\n")
    rf.write("\n")

    if oracle_params:
        rf.write(f"Interaction style     : {oracle_params.get('interaction_style', 'N/A')}\n")
        rf.write(f"Temperature           : {oracle_params.get('temperature', 'N/A')}\n")
        rf.write(f"Top-p                 : {oracle_params.get('top_p', 'N/A')}\n")
        rf.write(f"Reasoning effort      : {oracle_params.get('reasoning_effort', 'N/A')}\n")
        rf.write(f"Max completion tokens : {oracle_params.get('max_completion_tokens', 'N/A')}\n")
        rf.write(f"Enable thinking       : {oracle_params.get('enable_thinking', 'N/A')}\n")
        rf.write(f"Chat template kwargs  : {oracle_params.get('supports_chat_template_kwargs', 'N/A')}\n")
        rf.write(f"Max workers (threads) : {oracle_params.get('max_workers', max_workers)}\n")
        rf.write(f"Failure tolerance     : {oracle_params.get('failure_tolerance', 'N/A')}\n")
        rf_val = oracle_params.get('response_format', 'N/A')
        rf_name_f = rf_val.__name__ if hasattr(rf_val, '__name__') else str(rf_val)
        rf.write(f"Response format       : {rf_name_f}\n")
    rf.write("\n")

    rf.write(f"Developer prompt name     : {dev_prompt_name}\n")
    rf.write(f"Developer prompt set      : {developer_prompt_text is not None}\n")
    if developer_prompt_text is not None:
        rf.write(f"Developer prompt text      : {developer_prompt_text}\n")
    rf.write(f"User prompt template      : {oupt_name}\n")
    rf.write(f"Answer format             : {answer_format}\n")
    rf.write("\n")

    # token usage
    if m_ask_df_ext is not None and 'Oracle_input_tokens' in m_ask_df_ext.columns:
        total_in = m_ask_df_ext['Oracle_input_tokens'].sum()
        total_out = m_ask_df_ext['Oracle_output_tokens'].sum()
        if not pd.isna(total_in):
            rf.write(f"Total input tokens  : {int(total_in):,}\n")
            rf.write(f"Total output tokens : {int(total_out):,}\n")
            rf.write(f"Total tokens        : {int(total_in + total_out):,}\n")
    rf.write("\n")

    rf.write(f"GPU info:\n{gpu_info}\n\n")
    rf.write(f"Python version : {platform.python_version()}\n")
    rf.write(f"Platform       : {platform.platform()}\n")

    # Example prompt (full, not truncated)
    if m_ask_oracle_user_prompts:
        rf.write("\n")
        rf.write("-" * 72 + "\n")
        rf.write("Example Prompt (first mapping)\n")
        rf.write("-" * 72 + "\n")
        first_key = next(iter(m_ask_oracle_user_prompts))
        rf.write(f"Mapping: {first_key}\n\n")
        rf.write(m_ask_oracle_user_prompts[first_key])
        rf.write("\n")

success(f"Experiment results written to: {results_filepath}")

# end session


# %%

print()
success('LogMap-LLM session ending')
print()

# close the tee logger and restore original stdout
sys.stdout = original_stdout
tee.close()

info(f"Pipeline log saved to: {log_filepath}")