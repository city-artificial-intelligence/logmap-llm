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
import os.path
import tomllib
import pandas as pd
import json
import oracle_prompt_building as opb
import oracle_consultation as oc 

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
with open("logmap-llm-config-basic.toml", mode="rb") as fp:
    config = tomllib.load(fp)

# %% [markdown]
# Display the configuration parameter settings

# %%
#print(f'task name: {config['alignmentTask']['task_name']}')
#print(f'onto source: {config['alignmentTask']['onto_source_filepath']}')
#print(f'onto target: {config['alignmentTask']['onto_target_filepath']}')
#print(f'extended mappings_to_ask: {config['alignmentTask']['generate_extended_mappings_to_ask_oracle']}')
#print(f'logmap_parameters_dirpath: {config['alignmentTask']['logmap_parameters_dirpath']}')
#print()
#print(f'openrouter apikey: {config['oracle']['openrouter_apikey']}')
#print(f'openrouter LLM model name: {config['oracle']['openrouter_model_name']}')
#print(f'oracle dev prompt template: {config['oracle']['oracle_dev_prompt_template_name']}')
#print(f'oracle user prompt template: {config['oracle']['oracle_user_prompt_template_name']}')
#print()
#print(f'logmapllm output dirpath: {config['outputs']['logmapllm_output_dirpath']}')
#print(f'logmap initial alignment output dirpath: {config['outputs']['logmap_initial_alignment_output_dirpath']}')
#print(f'logmap refined alignment output dirpath: {config['outputs']['logmap_refined_alignment_output_dirpath']}')
#print()
#print(f'align ontologies: {config['pipeline']['align_ontologies']}')
#print(f'build oracle prompts: {config['pipeline']['build_oracle_prompts']}')
#print(f'consult oracle: {config['pipeline']['consult_oracle']}')
#print(f'refine alignment: {config['pipeline']['refine_alignment']}')


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
    print("JVM running, version:", jpype.getJVMVersion())
else:
    pass
    #print("JVM is not running!")

# %% [markdown]
# Start a JVM

# %%
if not jpype.isJVMStarted():
    jpype.startJVM(*jvmOptions)

# %% [markdown]
# Confirm a JVM is running

# %%
if not jpype.isJVMStarted():
    print("PROBLEM: JVM is not running!")
else:
    pass
    #print("JVM version:", jpype.getJVMVersion())

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
# blank for default; or just ignore this setter for default case
#path_to_logmap_parameters = '/Users/dave/research/logmap-usage/logmap-params/'
#path_to_logmap_parameters = ""  
path_to_logmap_parameters = config['alignmentTask']['logmap_parameters_dirpath']
logmap2_LogMapLLM_Interface.setPathToLogMapParameters(path_to_logmap_parameters)

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
print('LogMap-LLM session beginning')
print()
print(f'Alignment task name: {task_name}')
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
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 1: Align ontologies and obtain mappings to ask an Oracle')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

if config['pipeline']['align_ontologies'] == 'align':
    # perform an initial alignment so we can get a fresh m_ask
    print("Performing fresh initial LogMap alignment ...")
    print()
    logmap2_LogMapLLM_Interface.performAlignment()
    print("Initial alignment complete")
    mappings = logmap2_LogMapLLM_Interface.getLogMapMappings()
    print()
    print(f'Number of mappings in initial alignment: {len(mappings)}')
    m_ask_java = logmap2_LogMapLLM_Interface.getLogMapMappingsForLLM()
    m_ask_df = br.java_mappings_2_python(m_ask_java)
    # Note: we don't need to save m_ask_df to a file because LogMap
    # does that automatically, as part of performing an alignment
elif config['pipeline']['align_ontologies'] == 'reuse':
    # bypass an initial alignment and reuse an existing m_ask
    # saved to a file in an alignment conducted previously
    print("Reusing existing initial LogMap alignment ...")
    print()
    filename = task_name + '-logmap_mappings.txt'
    filepath = os.path.join(logmap_outputs_dir_path, filename)
    mappings = pd.read_csv(filepath, sep='|', header=None)
    print(f'Number of mappings in initial alignment: {len(mappings)}')
    print()
    filename = task_name + '-logmap_mappings_to_ask_oracle_user_llm.txt'
    print('Loading mappings to ask an Oracle from file:')
    print(filename)
    filepath = os.path.join(logmap_outputs_dir_path, filename)
    m_ask_df = pd.read_csv(filepath, sep='|', header=None)
    m_ask_df.columns = br.get_m_ask_column_names()
else:
    raise ValueError(f'Value for align_ontologies not recognised: {config['pipeline']['align_ontologies']}')

print()
print(f"Number of mappings to ask an Oracle: {len(m_ask_df)}")

# %% [markdown]
# ---
# 
# ## pipeline step 2: Build Oracle Prompts
# 
# ---

# %%
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 2: Build user prompts for mappings to ask an Oracle')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

oupt_name = config['oracle']['oracle_user_prompt_template_name']

if config['pipeline']['build_oracle_prompts'] == 'build':
    print('Building fresh Oracle user prompts ...')
    print()
    m_ask_oracle_user_prompts = opb.build_oracle_user_prompts(oupt_name,
                                                              onto_src_filepath,
                                                              onto_tgt_filepath, 
                                                              m_ask_df)
elif config['pipeline']['build_oracle_prompts'] == 'reuse':
    print('Reusing existing Oracle user prompts ...')
    print()
    # reuse oracle user prompts created previously and saved in a file on disk
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
    print('Loading oracle user prompts from file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'r') as fp:
        m_ask_oracle_user_prompts = json.load(fp)
elif config['pipeline']['build_oracle_prompts'] == 'bypass':
    print('Bypassing use of Oracle user prompts')
    m_ask_oracle_user_prompts = None
else:
    raise ValueError(f'Value for build_oracle_prompts not recognised: {config['pipeline']['build_oracle_prompts']}')

if m_ask_oracle_user_prompts is not None:
    print()
    print(f"Number of Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}")
    print()

if config['pipeline']['build_oracle_prompts'] == 'build':
    # save the newly built oracle user prompts to a .json file so they can be reused
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
    print('Oracle user prompts saved to file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'w') as fp:
        json.dump(m_ask_oracle_user_prompts, fp)


# %% [markdown]
# OPTIONAL: Inspect some of the generated Oracle user prompts
#

# %%
#cnt = 0
#for key, val in m_ask_oracle_user_prompts.items():
#    cnt += 1
#    if cnt > 2:
#        break
#    print(key)
#    print(val)
#    print()

# %% [markdown]
# ---
# 
# ## pipeline step 3: Consult Oracle
# 
# ---

# %%
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
print("Step 3: Consult LLM Oracle for each 'mapping to ask'")
print('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

api_key = config['oracle']['openrouter_apikey']
model_name = config['oracle']['openrouter_model_name']

# TODO: externalise max_workers in the config.toml file, so the user
# has control without having to modify Python code
max_workers = 2

if config['pipeline']['consult_oracle'] == 'consult':
    model_name = config['oracle']['openrouter_model_name']
    print(f'Consulting LLM Oracle: {model_name}')
    print('with the user prompts for the mappings to ask ...')
    print()
    m_ask_df_ext = oc.consult_oracle_for_mappings_to_ask(
        m_ask_oracle_user_prompts,
        api_key,
        model_name,
        max_workers,
        m_ask_df
    )
elif config['pipeline']['consult_oracle'] == 'reuse':
    print('Reusing existing Oracle consultation outcomes (predictions)')
    # reuse Oracle predictions created previously and saved in a file on disk
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_with_oracle_predictions.csv'
    print('Loading Oracle predictions for the mappings_to_ask from file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    m_ask_df_ext = pd.read_csv(filepath)
elif config['pipeline']['consult_oracle'] == 'bypass':
    print('Bypassing Oracle consultations')
    m_ask_df_ext = None
else:
    raise ValueError(f'Value for consult_oracle not recognised: {config['pipeline']['consult_oracle']}')

if m_ask_df_ext is not None:
    preds = m_ask_df_ext['Oracle_prediction']
    nr_mappings = len(preds)
    nr_errors = sum(preds == 'error')
    nr_completions = nr_mappings - nr_errors
    nr_true = sum(preds == True)
    nr_false = sum(preds == False)
    width = len(str(nr_mappings))
    nr_true = str(nr_true).rjust(width)
    nr_false = str(nr_false).rjust(width)
    nr_errors = str(nr_errors).rjust(width)
    print()
    print(f"Number of mappings to ask an Oracle: {nr_mappings}")
    print(f"Number of LLM Oracle consultations : {nr_completions}")
    print(f"Number of mappings predicted True  : {nr_true}")
    print(f"Number of mappings predicted False : {nr_false}")
    print(f"Number of consultation failures    : {nr_errors}")
    print()

if config['pipeline']['consult_oracle'] == 'consult' and m_ask_df_ext is not None:
    # save the extended m_ask dataframe (that contains the LLM Oracle predictions)
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_with_oracle_predictions.csv'
    print("Oracle predictions for 'mappings to ask' saved to file:")
    print(filename)
    filepath = os.path.join(dirpath, filename)
    m_ask_df_ext.to_csv(filepath)


# %%
#if m_ask_df_ext is not None:
#    m_ask_df_ext.head()

# %% [markdown]
# ---
# 
# ## Pipeline step 4: Refine Alignment
# 
# ---

# %%
print()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print('Step 4: Refine alignment using LLM Oracle mapping predictions')
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
print()

logmap_outputs_dir_path = config['outputs']['logmap_refined_alignment_output_dirpath']
logmap2_LogMapLLM_Interface.setPathForOutputMappings(logmap_outputs_dir_path)

if config['pipeline']['refine_alignment'] == 'refine':
    if m_ask_df_ext is not None:
        # refine the initial alignment using the m_ask Oracle predictions
        print("Refining initial LogMap alignment ...")
        print()
        m_ask_oracle_preds_java = br.python_oracle_mapping_predictions_2_java(m_ask_df_ext)
        print(f'Number of mappings predicted True by Oracle given to LogMap: {len(m_ask_oracle_preds_java)} ')
        print()
        logmap2_LogMapLLM_Interface.performAlignmentWithLocalOracle(m_ask_oracle_preds_java)
        print("Alignment refinement complete")
        mappings_java = logmap2_LogMapLLM_Interface.getLogMapMappings()
        print()
        print(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')
    else:
        print('Step 4 bypassed due to Oracle consultation failures in Step 3')
elif config['pipeline']['refine_alignment'] == 'bypass':
    print('Bypassing alignment refinement')
else:
    raise ValueError(f'Value for refine_alignment not recognised: {config['pipeline']['refine_alignment']}')


# %%

print()
print('LogMap-LLM session ending')
print()
