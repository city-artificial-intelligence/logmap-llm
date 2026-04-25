"""
logmap_llm.log_utils: coloured terminal output and ANSI-safe TeeWriter.

Originally ported from jd-extended, see:

    https://github.com/jonathondilworth/logmap-llm/tree/jd-extended

fns:

- error()      : bright red for serious errors / pipeline failures
- critical()   : bright red for critical warnings
- warning()    : bright red for serious warnings (draws attention)
- warn()       : pastel yellow for less serious warnings
- info()       : pastel sky-blue for important informational messages
- step()       : pastel sky-blue for step headers / section banners
- success()    : green for successful completion messages
- fatal()      : bright red, raises exception (NoReturn)
- strip_ansi() : remove ANSI escape sequences from a string
- TeeWriter    : duplicates stdout to terminal + log file

NOTE: see: https://ansi-generator.pages.dev/
& https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
"""

import re
from typing import NoReturn

# COLOUR CODES

_RESET = "\033[0m"
_BRIGHT_RED = "\033[91m"
_PASTEL_YELLOW = "\033[0;38;5;186;49m"
_BRIGHT_YELLOW_BOLD_AND_UNDERLINED = "\033[1;4;93;49m" # for critical warnings 
_DEFAULT_YELLOW = "\033[0;33;49m" # kinda orange
_PASTEL_BLUE = "\033[38;5;117m"
_IMPORTANT_STEP_PURPLE = "\033[0;38;2;255;179;219;49m"
_GREEN = "\033[38;5;114m"
_ITALIC_MAGENTA_W_BG = "\033[3;95;40m"
_METRIC_BLUE = "\033[0;94m"
_METRIC_FAINT_GREEN = "\033[0;38;5;157;49m"

# regex to strip ANSI escape sequences for log-file output
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """
    Remove all ANSI escape sequences from *text*.
    """
    return _ANSI_RE.sub("", text)


# COLOURED PRINT HELPERS:

def error(msg: str) -> None:
    """
    Print a serious error in bright red. Prefixed with 'ERROR'
    """
    print(f"{_BRIGHT_RED}[ERROR] {msg}{_RESET}")


def critical(msg: str) -> None:
    """
    Print a critical warning in bright red. Prefixed with 'CRITICAL WARNING'
    """
    print(f"{_BRIGHT_RED}[CRITICAL WARNING] {msg}{_RESET}")


def warning(msg: str) -> None:
    """
    Print a serious warning in bright yesllow (bold and underlined). Prefixed with 'SERIOUS WARNING'.
    """
    print(f"{_BRIGHT_YELLOW_BOLD_AND_UNDERLINED}[SERIOUS WARNING] {msg}{_RESET}")


def warn(msg: str) -> None:
    """
    Print a less-serious warning in pastel yellow. Prefixed with WARNING.
    """
    print(f"{_PASTEL_YELLOW}[WARNING] {msg}{_RESET}")


def info(msg: str, important: bool = False) -> None:
    """
    Print important information in pastel sky-blue. Prefixed with INFO.
    """
    print(f"{_PASTEL_BLUE if not important else _IMPORTANT_STEP_PURPLE}[INFO] {msg}{_RESET}")


def step(msg: str, important: bool = False) -> None:
    """
    Print a step / section header in pastel sky-blue.
    By convention, we include [Step X], e.g., "[Step 1] Align ontologies".
    """
    print(f"{_PASTEL_BLUE if not important else _IMPORTANT_STEP_PURPLE}{msg}{_RESET}")


def success(msg: str) -> None:
    """
    Print a success / completion message in green. Prefixed with SUCCES.
    """
    print(f"{_GREEN}[SUCCESS] {msg}{_RESET}")


def fatal(msg: str, exception_cls: type[Exception] = ValueError) -> NoReturn:
    """
    Print an error message and raise an exception.
    """
    error(msg)
    raise exception_cls(msg)


def debug(msg: str) -> None:
    """
    Print a debug message in magenta on a black background (italisied). Prefixed \w DEBUG.
    Specifically:
    ```sh
    echo -e '\033[3;95;40mHello, World!\033[0m'
    ```
    """
    print(f"{_ITALIC_MAGENTA_W_BG}[DEBUG] {msg}{_RESET}")


def metric(msg: str) -> None:
    """
    Print a metric message in 'high intensity' blue.
    """
    print(f"{_METRIC_FAINT_GREEN}[METRIC] {msg}{_RESET}")


###
# TeeWriter: terminal and log-file duplicator
###

class TeeWriter:
    """
    Writes to both the original stdout AND a log file simultaneously.
    """
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
        """Delegate to the real terminal."""
        return self.original_stdout.isatty()

    @property
    def encoding(self) -> str:
        """Return the encoding of the underlying terminal stream."""
        return getattr(self.original_stdout, "encoding", None) or "utf-8"

    def fileno(self) -> int:
        """Return the file-descriptor of the underlying terminal stream."""
        return self.original_stdout.fileno()

    def writable(self) -> bool:
        return True
