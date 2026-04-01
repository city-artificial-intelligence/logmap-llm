"""
log_utils.py -- provides lovely coloured terminal output
"""

import re
import sys

# ANSI colour codes
_RESET = "\033[0m"
# bright red: serious errors, serious warnings, pipeline failures
_BRIGHT_RED = "\033[91m"
# pastel yellow: less serious warnings
_PASTEL_YELLOW = "\033[38;5;229m"
# pastel sky blue: info
_PASTEL_BLUE = "\033[38;5;117m"
# green — success / completion
_GREEN = "\033[38;5;114m"
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


# print helpers:

def error(msg: str) -> None:
    print(f"{_BRIGHT_RED}ERROR: {msg}{_RESET}")


def warning(msg: str) -> None:
    print(f"{_BRIGHT_RED}WARNING: {msg}{_RESET}")


def warn(msg: str) -> None:
    print(f"{_PASTEL_YELLOW}WARNING: {msg}{_RESET}")


def info(msg: str) -> None:
    print(f"{_PASTEL_BLUE}{msg}{_RESET}")


def step(msg: str) -> None:
    print(f"{_PASTEL_BLUE}{msg}{_RESET}")


def success(msg: str) -> None:
    print(f"{_GREEN}{msg}{_RESET}")


class TeeWriter:
    """write to both the original stdout **and** a log file simultaneously"""

    def __init__(self, log_path: str, original_stdout):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.original_stdout = original_stdout

    def write(self, data: str) -> int:
        self.original_stdout.write(data)
        self.log_file.write(strip_ansi(data))
        return len(data)

    def flush(self) -> None:
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()

    def isatty(self) -> bool:
        """
        delegates to the real terminal 
        fixes a bug where sentence-transformers / tqdm / hf hub
        would call sys.stdout.isatty() and crash the pipeline
        """
        return self.original_stdout.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self.original_stdout, "encoding", None) or "utf-8"

    def fileno(self) -> int:
        return self.original_stdout.fileno()

    def writable(self) -> bool:
        return True
