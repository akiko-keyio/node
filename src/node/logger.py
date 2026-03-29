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

_RICH_HANDLER_KWARGS: dict = dict(
    markup=True, show_time=True, show_level=True, show_path=True,
)

# add rich handler (store ID so reporter can temporarily swap it out)
_handler_id: int | None = logger.add(
    RichHandler(console=console, **_RICH_HANDLER_KWARGS),
    level="DEBUG",
    format="{message}",
)


def _remove_rich_handler() -> None:
    """Remove the RichHandler. Call before Live to avoid stdout conflicts."""
    global _handler_id
    if _handler_id is not None:
        logger.remove(_handler_id)
        _handler_id = None


def _restore_rich_handler() -> None:
    """Re-add the RichHandler after Live exits."""
    global _handler_id
    if _handler_id is None:
        _handler_id = logger.add(
            RichHandler(console=console, **_RICH_HANDLER_KWARGS),
            level="DEBUG",
            format="{message}",
        )


# ensure future `from loguru import logger` returns this configured instance
sys.modules["loguru"].logger = logger  # type: ignore[attr-defined]

__all__ = ["logger", "console"]

