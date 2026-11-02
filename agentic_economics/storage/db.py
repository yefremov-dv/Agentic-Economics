"""Подключение к SQLite в режиме WAL (NFR-4a)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event, inspect
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


class SchemaOutdated(RuntimeError):
    """База создана старой версией программы — нужен reset-demo."""


def check_schema(engine: Engine) -> None:
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        missing = [c.name for c in table.columns if c.name not in existing]
        if missing:
            raise SchemaOutdated(
                f"в таблице {table.name} нет столбцов {', '.join(missing)}. "
                "Пересоздайте базу: python -m agentic_economics --admin reset-demo -y"
            )


def make_sessionmaker(settings: Settings) -> sessionmaker:
    engine = make_engine(settings)
    check_schema(engine)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def reset_db(settings: Settings) -> sessionmaker:
    engine = make_engine(settings)
    with engine.connect() as conn:
        # tasks.winner_bid_id и bids.task_id ссылаются друг на друга — без отключения проверки
        # внешних ключей SQLite не даст удалить заполненные таблицы
        conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        Base.metadata.drop_all(conn)
        conn.commit()
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
