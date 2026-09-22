import shutil
from pathlib import Path

import pytest
import yaml

from agentic_economics.config import Settings, load_settings
from agentic_economics.storage.db import make_sessionmaker

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_real_secrets(monkeypatch):
    """Тесты не должны видеть настоящие ключи GigaChat из окружения или .env."""
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_ACCESS_TOKEN", "GIGACHAT_SCOPE", "GIGACHAT_VERIFY_SSL_CERTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agentic_economics.config.load_dotenv", lambda **_: None)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    agents_dir = tmp_path / "agents"
    shutil.copytree(ROOT / "agents", agents_dir)
    shutil.copy(ROOT / "data" / "seed.yaml", tmp_path / "seed.yaml")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["paths"] = {
        "db": str(tmp_path / "var" / "test.db"),
        "tasks_dir": str(tmp_path / "tasks"),
        "agents_dir": str(agents_dir),
        "seed_file": str(tmp_path / "seed.yaml"),
        "logs_dir": str(tmp_path / "logs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def settings(config_file: Path) -> Settings:
    return load_settings(config_file)


@pytest.fixture
def seeded(settings: Settings) -> Settings:
    from agentic_economics.seed import reset_demo

    reset_demo(settings)
    return settings


@pytest.fixture
def db(seeded: Settings):
    with make_sessionmaker(seeded)() as session:
        yield session
