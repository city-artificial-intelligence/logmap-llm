"""
Originally ported from:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/logmap_interface.py

Note that the LogMapLLM_Interface java-based implementation is viewable at:

    https://github.com/ernestojimenezruiz/logmap-matcher/blob/master/src/main/java/uk/ac/ox/krr/logmap2/LogMapLLM_Interface.java

Methods exposed by the java-based implementation are callable via this python interface.

Specific methods of note include:

- The class constructor, which accepts two ontology URI strings and a task name.
- `setExtendedQuestions4LLM`, which accepts a boolean to toggle extended mappings on/off.
- `setPathToLogMapParameters`, which accepts a string (filepath) to allow custom LogMap parameter settings.
- `setPathForOutputMappings`, which accepts a string (filepath) for specifying an output mappings dir.
- `performAlignment`, will perform the alignment without requiring an oracle.
- `getLogMapMappings`, will return LogMap mappings following an alignment.
- `getLogMapMappingsForLLM`, will return LogMap mappings to ask (M_ask) following an alignment.
- `performAlignmentWithLocalOracle`, which accepts oracle predictions; performs an alignment accounting for predictions.

Note: to use this interface, we must ensure that:

1. JPype has been imported (with logmap being included within its classpath)
2. A JPype JVM has been started and is running
3. LogMapLLM_Interface resolves from uk.ac.ox.krr.logmap2

The above steps are contained within this Python class's constructor. Other important notes:

- Ontology filepaths provided to the interface require a 'file:' URI prefix.
- The specified parameters directory path must end with `os.sep` (e.g., `/`).
- The output directory is mutable (this has implications when running multiple processes at once).

LogMapInterface (class) example usage:

```
from logmap_llm.interface import (
    start_jvm,
    LogMapInterface
)
import pandas as pd

# by convention, we include `logmap` within the current working directory (at the root level)
logmap_dirpath = os.path.join(os.getcwd(), 'logmap')

# ensure the JPype JVM is running:
start_jvm("/home/jon/my-logmap-llm/logmap/")

# now its safe to import bridging (since the JVM is running)
import logmap_llm.bridging as br

# construct an interface:
logmap_interface = LogMapInterface(
    src_uri="/home/jon/my-logmap-llm/datasets/bio-ml/omim-ordo/omim.owl",
    tgt_uri="/home/jon/my-logmap-llm/datasets/bio-ml/omim-ordo/ordo.owl",
    task_name="omim-ordo",
)

# specify the desired parameters location (by default it exists in the logmap_dirpath):
logmap_interface.set_parameters_dir(logmap_dirpath)

# specify a setting for extended mappings:
logmap_interface.set_extended_questions_for_llm(False)

# specify the current output directory (NOTE: this directory MUST EXIST):
logmap_interface.set_output_dir("/home/jon/my-logmap-llm/results/bio-ml/omim-ordo/initial-alignment")

# perform the initial alignment:
logmap_interface.perform_alignment()

# obtain the initial mappings:
initial_alignment_mappings: pd.DataFrame = br.java_mappings_2_python(
    logmap_interface.get_mappings()
)

# obtain the mappings to ask:
m_ask_df: pd.DataFrame = br.java_mappings_2_python(
    logmap_interface.get_mappings_for_llm()
)

# verify the output:
print(initial_alignment_mappings)
print(m_ask_df)

# verify the output:
print(initial_alignment_mappings.shape)
print(m_ask_df.shape)

# STEPS TO CONSULT ORACLE ...
# oracle_predictions = ...

# update the output dir to the refinement location:
# logmap_interface.set_output_dir("/home/jon/my-logmap-llm/results/bio-ml/omim-ordo/refined-alignment")

# obtain the refined mappings:
# logmap_interface.refine_alignment(oracle_predictions)
```

TODO: TESTS

Write tests according to the following specification:

  TEST ONE (end-2-end)

  1. download mappings
  2. setup the directory structure
  3. perform the above code execution
  4. verify the output is as expected
  
  TEST TWO/THREE/FOUR (ensuring the interface works as intended)
  
  SETUP: 
      configure a LogMap interface \w custom params & custom output_dir & custom M_ask bool
  TEST:
      set custom params
  ASSERT:
      get_path_to_parameters == custom values specified
      get_path_to_output_mappings == custom output_dir specified
      get_extract_extended_questions_for_llm == custom value specified for M_ask bool
"""

import os
import jpype                # type: ignore
import jpype.imports        # type: ignore
from jpype.types import *   # type: ignore
from pathlib import Path
from typing import Optional

from logmap_llm.constants import VERBOSE
from logmap_llm.config.schema import LogMapLLMConfig


def start_jvm(logmap_dir: str | Path) -> None:
    """
    TODO: doc-comment.
    """
    # NOTE: 'logmap-matcher-4.0.jar' and 'java-dependencies' should
    # not be hardcoded, as they could feasibly change in future
    logmap_jar = os.path.join(logmap_dir, 'logmap-matcher-4.0.jar')
    logmap_dep = os.path.join(logmap_dir, 'java-dependencies/*')
    
    jpype.addClassPath(logmap_jar)
    jpype.addClassPath(logmap_dep)
    
    # perform checks & start JVM
    if jpype.isJVMStarted():
        raise RuntimeError("JVM already running unexpectedly")

    # TODO: JVM params should be configurable
    jpype.startJVM(
        "-Xms500M",
        "-Xmx25G",
        "-DentityExpansionLimit=10000000",
        "--add-opens=java.base/java.lang=ALL-UNNAMED"
    )

    if not jpype.isJVMStarted():
        raise RuntimeError("JVM failed to start")



class LogMapInterface:
    """
    A python wrapper for the LogMap java interface `LogMapLLM_Interface`.
    Provides a convenient abstraction for interfacing with LogMap.
    """
    def __init__(self, src_uri: str, tgt_uri: str, task_name: str):

        # if jpype.isJVMStarted():
        #     raise RuntimeError("Cannot instanciate a LogMapInterface without JPype LogMap JVM running.")
              # NOTE ^ always raises an exception ...

        self._src_uri = "file:" + src_uri
        self._tgt_uri = "file:" + tgt_uri
        self._task_name = task_name

        from uk.ac.ox.krr.logmap2 import LogMapLLM_Interface  # type: ignore

        self._interface = LogMapLLM_Interface(
            self._src_uri, self._tgt_uri, self._task_name,
        )


    @classmethod
    def create_and_configure(cls, src_uri: str, tgt_uri: str, task_name: str, logmap_dir: str | Path, extended_m_ask: bool, output_dir: str | Path) -> "LogMapInterface":
        """
        A convenient factory for creating and configuring a LogMapInterface.
        (since we cannot overload the constructor in Python).
        """
        instance = cls(src_uri, tgt_uri, task_name)
        instance.set_parameters_dir(logmap_dir)
        instance.set_extended_questions_for_llm(extended_m_ask)
        instance.set_output_dir(output_dir)
        return instance
    

    @classmethod
    def create_from_cfg(cls, cfg: LogMapLLMConfig, logmap_dir: str | Path | None = None) -> "LogMapInterface":
        """
        Backwards compatability
        """
        this_logmap_dir = cfg.alignmentTask.logmap_parameters_dirpath if logmap_dir is None else logmap_dir
        instance = cls.create_and_configure(
            src_uri=cfg.alignmentTask.onto_source_filepath,
            tgt_uri=cfg.alignmentTask.onto_target_filepath,
            task_name=cfg.alignmentTask.task_name,
            logmap_dir=this_logmap_dir,
            extended_m_ask=cfg.alignmentTask.generate_extended_mappings_to_ask_oracle,
            output_dir=cfg.outputs.logmap_initial_alignment_output_dirpath,
        )
        return instance



    def set_parameters_dir(self, logmap_dir: str | Path) -> None:
        """
        Specify the LogMap directory (or a custom directory) where `Parameters.txt`
        for LogMap is found. Do not include `/parameters.txt`, it should be the base dir.
        For example: `/home/user/logmap-llm/logmap/`.
        """
        logmap_dir_path = str(logmap_dir)
        if not logmap_dir_path.endswith(os.sep):
            logmap_dir_path += os.sep
        self._interface.setPathToLogMapParameters(logmap_dir_path)


    def set_output_dir(self, dirpath: str | Path) -> None:
        """
        Sets the directory LogMap writes its output (e.g., alignment, M_ask) to.
        Should be specified before step 1 (initial alignment), and step 4 (refine alignment).
        """
        self._interface.setPathForOutputMappings(str(dirpath))


    def set_extended_questions_for_llm(self, produce_extended_questions: bool) -> None:
        """Sets whether LogMap should produce the extended M_ask set."""
        self._interface.setExtendedQuestions4LLM(produce_extended_questions)


    def perform_alignment(self) -> None:
        """Perform an alignment with LogMap."""
        self._interface.performAlignment()


    def get_mappings(self):
        """Returns LogMap mappings, following an alignment."""
        return self._interface.getLogMapMappings()


    def get_mappings_for_llm(self):
        """Returns LogMap M_ask set, following an alignment."""
        return self._interface.getLogMapMappingsForLLM()


    def refine_alignment(self, oracle_predictions) -> None:
        """Refine an alignment with LogMap."""
        self._interface.performAlignmentWithLocalOracle(oracle_predictions)

    ###
    # TODO: update LogMap src. see below.
    ###

    ## NOTE: requires changes to LogMap src (the variable is not currently public)
    # def get_path_to_parameters(self) -> str:
    #     """Fetches and returns LogMap's public instance variable `path_to_paramaters`"""
    #     return self._interface.path_to_paramaters


    ## NOTE: requires changes to LogMap src (the variable is not currently public)
    # def get_extract_extended_questions_for_llm(self) -> bool:
    #     """Fetches and returns LogMap's public instance variable `extractExtendedQuestions4LLM`."""
    #     return self._interface.extractExtendedQuestions4LLM


    ## NOTE: requires changes to LogMap src (the variable is not currently public)
    # def get_path_to_output_mappings(self) -> str:
    #     """Fetches and returns LogMap's public instance variable `path_to_output_mappings`"""
    #     return self._interface.path_to_output_mappings