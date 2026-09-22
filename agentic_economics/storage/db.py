"""Подключение к SQLite в режиме WAL (NFR-4a)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import sessionmaker

from agentic_economics.config import Settings
from agentic_economics.storage.models import Base


def make_engine(settings: Settings) -> Engine:
    db_path = settings.path("db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 15})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


def make_sessionmaker(settings: Settings) -> sessionmaker:
    engine = make_engine(settings)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def reset_db(settings: Settings) -> sessionmaker:
    engine = make_engine(settings)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
