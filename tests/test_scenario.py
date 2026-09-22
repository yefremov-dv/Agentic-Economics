"""Сценарный тест полного цикла на фейковой LLM (NFR-6, критерии приёмки раздела 10 ТЗ)."""

import re

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from agentic_economics.cli import app
from agentic_economics.orchestrator import taskfile
from agentic_economics.storage.db import make_sessionmaker
from agentic_economics.storage.models import Bid, Rating, Task, Transaction, Wallet
from agentic_economics.wallet.ledger import SYSTEM_OWNER, balance
from tests.fakes import TURKEY_PLAN, FakeLLM, analysis

REQUEST = ("Хочу через неделю полететь в Турцию на 6 ночей в отель Maxx Royal Belek, бюджет 1 000 000 рублей. "
           "Хочу, чтобы на мне были розовые кроссовки Nike.")
runner = CliRunner()


@pytest.fixture
def chat(seeded, config_file, monkeypatch):
    monkeypatch.setattr("agentic_economics.cli.console.width", 220)

    def _chat(llm, user, lines):
        monkeypatch.setattr("agentic_economics.cli.get_llm", lambda settings, model=None: llm)
        result = runner.invoke(app, ["--config", str(config_file), "chat", "--user", user],
                               input="".join(f"{line}\n" for line in lines))
        assert result.exit_code == 0, result.output
        return result.output

    return _chat


def balances(db) -> dict[str, int]:
    db.commit()
    return {f"{w.owner_type}:{w.owner_id}": balance(db, w) for w in db.scalars(select(Wallet))
            if (w.owner_type, w.owner_id) != SYSTEM_OWNER}


def test_full_cycle(chat, seeded, config_file):
    db = make_sessionmaker(seeded)()
    before = balances(db)
    tx_before = db.query(Transaction).count()
    llm = FakeLLM(analysis(ready=False, questions=["Из какого города вылет?", "Какой размер обуви?"], budget=1_000_000),
                  analysis(budget=1_000_000), TURKEY_PLAN)

    out = chat(llm, "ivanov", [REQUEST, "Из Москвы, лечу один, размер 42", "/plan"])
    session_id = re.search(r"Сессия (\S+)\. Опишите задачу", out).group(1)

    # 2а. Уточняющие вопросы
    assert "Из какого города вылет?" in out
    # 2б. Не менее 4 файлов task-N.md с корректными категориями и зависимостями
    folder = seeded.path("tasks_dir") / session_id
    metas = [taskfile.read_meta(p) for p in sorted(folder.glob("task-*.md"))]
    assert len(metas) >= 4
    assert {m["category"] for m in metas} >= {"flights", "hotels", "transfers", "retail"}
    transfer = next(m for m in metas if m["category"] == "transfers")
    dep_categories = {next(x["category"] for x in metas if x["id"] == d) for d in transfer["depends_on"]}
    assert dep_categories == {"flights", "hotels"}
    # 2в. Аукцион по каждой подзадаче, не менее двух ставок
    tasks = db.scalars(select(Task).where(Task.session_id == session_id)).all()
    for t in tasks:
        assert len([b for b in db.scalars(select(Bid).where(Bid.task_id == t.id)) if b.status in ("winner", "lost")]) >= 2
    # 2г. План показан, до подтверждения ни одного списания
    assert "План на подтверждение" in out and "К оплате" in out
    assert db.query(Transaction).count() == tx_before

    # Включаем рекомендованную страховку, подтверждаем, оцениваем двух исполнителей
    insurance = next(m for m in metas if m["category"] == "insurance")
    result = runner.invoke(app, ["--config", str(config_file), "chat", "--user", "ivanov", "--session", session_id],
                           input=f"/enable {insurance['id']}\n/confirm\n5\n\n4\n\n\n/result\n/exit\n")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Выполнение" in out and "выполнена" in out

    # 2д. Выполнение, списания и выплаты в журнале, result.md
    db.expire_all()
    tasks = db.scalars(select(Task).where(Task.session_id == session_id)).all()
    assert all(t.status == "done" for t in tasks)
    kinds = {tx.kind for tx in db.scalars(select(Transaction).where(Transaction.task_id.in_([t.id for t in tasks])))}
    assert kinds == {"hold", "payout", "fee"}
    report = (folder / "result.md").read_text(encoding="utf-8")
    assert "Итоги сессии" in report and "Оплачено исполнителям" in report
    assert "Итоги сессии" in out  # /result

    # 3. Итог в пределах бюджета
    after = balances(db)
    spent = before["client:ivanov"] - after["client:ivanov"]
    assert 0 < spent <= 1_000_000_00
    # 4. Деньги не создаются и не исчезают: сумма всех балансов не изменилась
    assert sum(after.values()) == sum(before.values())
    assert after["platform:platform"] == sum(
        tx.amount for tx in db.scalars(select(Transaction).where(Transaction.kind == "fee")))
    # Оценки (FR-32)
    assert db.query(Rating).count() == 2
    db.close()


def test_decline_creates_no_transactions(chat, seeded):
    db = make_sessionmaker(seeded)()
    before = db.query(Transaction).count()
    out = chat(FakeLLM(analysis(), TURKEY_PLAN), "petrova", [REQUEST, "/cancel", "y"])
    assert "Сессия отменена. Деньги не списывались." in out
    db.expire_all()
    assert db.query(Transaction).count() == before  # критерий приёмки 5
    db.close()


def test_confirm_word_and_insufficient_funds(chat, seeded, config_file):
    runner.invoke(app, ["--config", str(config_file), "--admin", "user", "add", "poor", "--name", "Бедняк",
                        "--balance", "1000"])
    out = chat(FakeLLM(analysis(), TURKEY_PLAN), "poor", [REQUEST, "подтверждаю"])
    assert "Недостаточно средств" in out
    assert "не хватает" in out  # предупреждение ещё в плане
