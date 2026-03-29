"""Pre-configured Loguru logger with Rich output."""

from loguru import logger
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


def _is_jupyter() -> bool:
    """Detect if running inside a Jupyter notebook (ZMQ-based kernel)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "zmq" in type(shell).__module__
    except Exception:
        return False


# beautify tracebacks with Rich
install()

# In Jupyter, force terminal mode so all output goes through stdout as a
# contiguous ANSI stream instead of creating separate display() elements.
console = Console(force_terminal=True) if _is_jupyter() else Console()

# remove default handler
logger.remove()

# add rich handler
logger.add(
    RichHandler(
        console=console,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=True,
    ),
    level="DEBUG",
    format="{message}",
)


# ensure future `from loguru import logger` returns this configured instance
sys.modules["loguru"].logger = logger  # type: ignore[attr-defined]

__all__ = ["logger", "console"]

