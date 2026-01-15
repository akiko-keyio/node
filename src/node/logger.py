"""Pre-configured Loguru logger with Rich output."""

from loguru import logger
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# beautify tracebacks with Rich
install()

console = Console()

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

