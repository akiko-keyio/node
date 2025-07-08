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

if __name__ == "__main__":  # pragma: no cover
    logger.debug("\u8fd9\u662f\u4e00\u4e2a\u8c03\u8bd5\u4fe1\u606f")
    logger.info("\u7528\u6237 {user} \u767b\u5f55\u6210\u529f", user="alice")
    import time

    time.sleep(2)
    logger.warning("\u78c1\u76d8\u7a7a\u95f4\u53ea\u5269\u4e0b {percent}%", percent=5)
    logger.error("\u65e0\u6cd5\u8fde\u63a5\u6570\u636e\u5e93\uff01")
