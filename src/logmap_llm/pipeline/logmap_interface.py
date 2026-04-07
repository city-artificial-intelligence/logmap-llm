"""
Ported from:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/logmap_interface.py


"""

import os
import jpype
import jpype.imports
from jpype.types import * # type: ignore
from pathlib import Path
from logmap_llm.config.schema import LogMapLLMConfig


def start_jvm(logmap_dir: str | Path) -> None:
    """configures classpath and boots jpype JVM for LogMap"""
    logmap_jar = os.path.join(logmap_dir, 'logmap-matcher-4.0.jar')
    logmap_dep = os.path.join(logmap_dir, 'java-dependencies/*')
    jpype.addClassPath(logmap_jar)
    jpype.addClassPath(logmap_dep)
    # perform checks & start JVM
    if jpype.isJVMStarted():
        raise RuntimeError("JVM already running unexpectedly")
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
    python wrapper for the LogMap java interface
    encapsulates LogMap-Java API, important notes:
      - ontology paths require a 'file:' URI prefix
      - parameters directory path must end with os.sep
      - output directory is mutable (i.e., init-align -> refined-align)
    IMPORTANT: this interface will only work if:
        1. JPype has been imported
        2. JVM has started
        3. LogMapLLM_Interface resolves from uk.ac.ox.krr.logmap2
            -> if 1 & 2 then 3 should be fine..
    """

    def __init__(self, cfg: LogMapLLMConfig, logmap_dir: str | Path):
        # Java imports for basic LogMap usage
        from uk.ac.ox.krr.logmap2 import LogMapLLM_Interface  # type: ignore
        # obtain relevant experimental params from the cfg & process for LogMap
        self.task_name = cfg.alignmentTask.task_name
        src_uri = "file:" + cfg.alignmentTask.onto_source_filepath
        tgt_uri = "file:" + cfg.alignmentTask.onto_target_filepath
        # instanciate LogMap via uk.ac.ox.krr.logmap2.LogMapLLM_Interface
        # & specify the neccesary configuration (extd-qestions & params dir)
        self._interface = LogMapLLM_Interface(
            src_uri, tgt_uri, self.task_name
        )
        self._interface.setExtendedQuestions4LLM(
            cfg.alignmentTask.generate_extended_mappings_to_ask_oracle
        )
        self._set_parameters_dir(
            cfg.alignmentTask.logmap_parameters_dirpath, logmap_dir
        )

    def _set_parameters_dir(self, configured_path: str, logmap_dir: str | Path) -> None:
        """
        resolve and set the LogMap parameters directory
        falls back to the LogMap installation directory if no path is configured.
        appends a trailing separator as required by the LogMap Java imlementation
        """
        path = configured_path if configured_path else str(logmap_dir)
        if not path.endswith(os.sep):
            path += os.sep
        self._interface.setPathToLogMapParameters(path)

    def set_output_dir(self, dirpath: str | Path) -> None:
        """
        set the directory LogMap writes alignment output to
        called before step 1 (initial) and step 4 (refined) with:
        --> !! different directories !! <--
        """
        self._interface.setPathForOutputMappings(str(dirpath))

    def perform_alignment(self) -> None:
        self._interface.performAlignment()

    def get_mappings(self):
        return self._interface.getLogMapMappings()

    def get_mappings_for_llm(self):
        return self._interface.getLogMapMappingsForLLM()

    def refine_alignment(self, oracle_predictions) -> None:
        self._interface.performAlignmentWithLocalOracle(oracle_predictions)