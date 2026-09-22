"""Загрузка конфигурации из config.yaml и переменных окружения."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_ENV = "AE_CONFIG"
DEFAULT_CONFIG = "config.yaml"


class PathsConfig(BaseModel):
    db: Path = Path("var/agentic.db")
    tasks_dir: Path = Path("tasks")
    agents_dir: Path = Path("agents")
    seed_file: Path = Path("data/seed.yaml")
    logs_dir: Path = Path("logs")


class LLMConfig(BaseModel):
    model: str = "GigaChat"
    temperature: float = 0.3
    timeout: float = 60
    max_retries: int = 3


class AuctionWeights(BaseModel):
    price: float = 0.5
    confidence: float = 0.3
    rating: float = 0.2


class AuctionConfig(BaseModel):
    max_candidates: int = Field(5, ge=1)
    bid_timeout: float = Field(30, gt=0)
    weights: AuctionWeights = AuctionWeights()


class WalletConfig(BaseModel):
    currency: str = "VRUB"
    platform_fee: float = Field(0.05, ge=0, lt=1)
    default_client_balance: float = Field(1_500_000, ge=0)


class GigaChatSecrets(BaseSettings):
    """Секреты GigaChat — только из окружения или .env, никогда из config.yaml."""

    model_config = SettingsConfigDict(env_prefix="GIGACHAT_", extra="ignore")

    credentials: SecretStr | None = None
    scope: str = "GIGACHAT_API_PERS"
    verify_ssl_certs: bool = True


class Settings(BaseModel):
    root: Path
    paths: PathsConfig = PathsConfig()
    llm: LLMConfig = LLMConfig()
    auction: AuctionConfig = AuctionConfig()
    wallet: WalletConfig = WalletConfig()

    def path(self, name: str) -> Path:
        """Абсолютный путь из секции paths."""
        p = getattr(self.paths, name)
        return p if p.is_absolute() else self.root / p

    @property
    def gigachat(self) -> GigaChatSecrets:
        return GigaChatSecrets()


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(override=False)
    path = Path(config_path or os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)).resolve()
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(root=path.parent, **data)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
