import logging

from rich.console import Console
from rich.logging import RichHandler

from agentic_economics.config import Settings


def setup_logging(settings: Settings, verbose: bool = False, trace: bool = False) -> None:
    """Логи в logs/agentic.log; --verbose — уровень DEBUG; --trace — ход работы оркестратора и ставки в терминал (NFR-4)."""
    logs_dir = settings.path("logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(logs_dir / "agentic.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose or trace else logging.INFO)
    if trace:
        console = RichHandler(console=Console(stderr=True), show_path=False, markup=False, rich_tracebacks=False)
        console.setLevel(logging.DEBUG)
        console.addFilter(lambda record: record.name.startswith("agentic_economics"))
        root.addHandler(console)
