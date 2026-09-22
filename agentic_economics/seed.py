"""Пересоздание демо-базы из data/seed.yaml и agents/*.yaml (NFR-5)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

import yaml

from agentic_economics.agents.registry import SyncReport, sync_agents
from agentic_economics.agents.spec import ID_PATTERN
from agentic_economics.config import Settings
from agentic_economics.money import to_minor
from agentic_economics.storage.db import reset_db
from agentic_economics.storage.models import Provider, User
from agentic_economics.wallet.ledger import PLATFORM_OWNER, get_wallet, topup


class SeedProvider(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=ID_PATTERN)
    name: str
    inn: str | None = None
    balance: float = 0


class SeedClient(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=ID_PATTERN)
    name: str
    balance: float | None = None


class Seed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    providers: list[SeedProvider] = []
    clients: list[SeedClient] = []


def reset_demo(settings: Settings) -> SyncReport:
    seed = Seed.model_validate(yaml.safe_load(settings.path("seed_file").read_text(encoding="utf-8")) or {})
    currency = settings.wallet.currency
    make_session = reset_db(settings)
    with make_session() as db:
        get_wallet(db, *PLATFORM_OWNER, currency=currency)
        for p in seed.providers:
            db.add(Provider(id=p.id, name=p.name, inn=p.inn))
            wallet = get_wallet(db, "provider", p.id, currency)
            if p.balance:
                topup(db, wallet, to_minor(p.balance), note="Стартовый баланс")
        for c in seed.clients:
            db.add(User(id=c.id, name=c.name, role="client"))
            wallet = get_wallet(db, "client", c.id, currency)
            amount = settings.wallet.default_client_balance if c.balance is None else c.balance
            if amount:
                topup(db, wallet, to_minor(amount), note="Стартовый баланс")
        db.commit()
        return sync_agents(db, settings.path("agents_dir"))
