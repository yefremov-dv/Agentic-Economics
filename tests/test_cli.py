import re

import pytest
from typer.testing import CliRunner

from agentic_economics.cli import app
from tests.fakes import TURKEY_PLAN, FakeLLM, analysis

runner = CliRunner()


@pytest.fixture
def run(seeded, config_file, monkeypatch):
    monkeypatch.setattr("agentic_economics.cli.console.width", 200)

    def _run(*args, input=None):
        return runner.invoke(app, ["--config", str(config_file), *args], input=input)

    return _run


@pytest.mark.parametrize(
    "args",
    [["agent", "list"], ["user", "add", "x1", "--name", "X"], ["wallet", "balance", "ivanov"], ["ledger"],
     ["reset-demo", "-y"], ["provider", "list"], ["auction", "show", "1"]],
)
def test_admin_commands_require_flag(run, args):
    result = run(*args)
    assert result.exit_code == 1
    assert "--admin" in result.output


def test_user_add_and_balance(run):
    assert run("--admin", "user", "add", "kuznetsova", "--name", "Анна", "--balance", "1000").exit_code == 0
    result = run("--admin", "wallet", "balance", "client:kuznetsova")
    assert "1 000.00 VRUB" in result.output
    assert run("--admin", "user", "add", "kuznetsova", "--name", "Анна").exit_code == 1
    assert run("--admin", "user", "add", "Bad_Id", "--name", "X").exit_code == 1


def test_topup_writes_ledger(run):
    assert run("--admin", "wallet", "topup", "petrova", "500").exit_code == 0
    result = run("--admin", "ledger", "--owner", "client:petrova")
    assert "Пополнение администратором" in result.output


def test_provider_add_then_agent_disable(run):
    assert run("--admin", "provider", "add", "new-co", "--name", "ООО «Новая»").exit_code == 0
    assert run("--admin", "agent", "disable", "retail-sportline").exit_code == 0
    assert "retail-sportline" not in run("--admin", "agent", "list").output
    assert "retail-sportline" in run("--admin", "agent", "list", "--all").output


def use_llm(monkeypatch, *responses):
    llm = FakeLLM(*responses)
    monkeypatch.setattr("agentic_economics.cli.get_llm", lambda settings, model=None: llm)
    return llm


def test_chat_questions_plan_and_commands(run, monkeypatch, seeded):
    use_llm(monkeypatch, analysis(ready=False, questions=["Город вылета?"], budget=1_000_000),
            analysis(budget=1_000_000), TURKEY_PLAN)
    result = run("chat", "--user", "ivanov",
                 input="/balance\nХочу в Турцию\nИз Москвы\n/plan\n/history\n/bids\n/exit\n")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "1 500 000.00 VRUB" in out
    assert "Город вылета?" in out
    assert "План готов, аукцион проведён." in out
    assert "Отель Maxx Royal Belek, 6 ночей" in out
    assert "рекомендовано, выключено" in out
    assert "победитель" in out  # таблица /bids
    assert "К оплате:" in out and "комиссия платформы" in out
    session_id = re.search(r"Сессия (\S+)\. Опишите задачу", out).group(1)
    assert len(list((seeded.path("tasks_dir") / session_id).glob("task-*.md"))) == 5


def test_chat_resume_session(run, monkeypatch):
    use_llm(monkeypatch, analysis(), TURKEY_PLAN)
    out = run("chat", "--user", "ivanov", input="Хочу в Турцию\n/exit\n").output
    session_id = re.search(r"Сессия (\S+)\. Опишите задачу", out).group(1)

    result = run("chat", "--user", "ivanov", "--session", session_id, input="/exit\n")
    assert result.exit_code == 0
    assert f"Продолжаем сессию {session_id}" in result.output
    assert "Хочу в Турцию" in result.output  # история
    assert "Трансфер аэропорт ↔ отель" in result.output  # план

    assert run("chat", "--user", "petrova", "--session", session_id).exit_code == 1
    assert "Незавершённые сессии" in run("chat", "--user", "ivanov", input="/exit\n").output


def test_chat_cancel(run, monkeypatch):
    use_llm(monkeypatch, analysis(), TURKEY_PLAN)
    out = run("chat", "--user", "ivanov", input="Хочу в Турцию\n/cancel\ny\n").output
    assert "Сессия отменена." in out
    session_id = re.search(r"Сессия (\S+)\. Опишите задачу", out).group(1)
    assert "отменена" in run("chat", "--user", "ivanov", "--session", session_id).output


def test_chat_requires_llm(run):
    result = run("chat", "--user", "ivanov")
    assert result.exit_code == 1
    assert "GIGACHAT_CREDENTIALS" in result.output


def test_outdated_schema_asks_for_reset(run, seeded):
    import sqlite3

    con = sqlite3.connect(seeded.path("db"))
    con.execute("ALTER TABLE tasks DROP COLUMN criteria")
    con.commit()
    con.close()
    result = run("--admin", "agent", "list")
    assert result.exit_code == 1
    assert "reset-demo" in result.output


def test_chat_unknown_user(run):
    assert run("chat", "--user", "nobody").exit_code == 1


def test_check_llm_without_credentials(run):
    result = run("check-llm")
    assert result.exit_code == 1
    assert "GIGACHAT_CREDENTIALS" in result.output


def test_wallet_list_reconciles(run):
    result = run("--admin", "wallet", "list")
    assert result.exit_code == 0
    assert "client:ivanov" in result.output and "system:emission" in result.output
    assert "Выпущено (эмиссия): 4 500 000.00 VRUB" in result.output
    assert "сходится" in result.output


def test_balance_shows_escrow(run, monkeypatch, seeded):
    from agentic_economics.storage.db import make_sessionmaker
    from agentic_economics.storage.models import Task
    from agentic_economics.wallet.escrow import hold

    use_llm(monkeypatch, analysis(), TURKEY_PLAN)
    run("chat", "--user", "ivanov", input="Хочу в Турцию\n/exit\n")
    with make_sessionmaker(seeded)() as db:
        task = db.query(Task).filter(Task.number == 1).one()
        hold(db, "ivanov", [(task, 100_000_00)])
    out = run("chat", "--user", "ivanov", input="/balance\n/exit\n").output
    assert "Доступно: 1 400 000.00 VRUB, заблокировано в эскроу: 100 000.00 VRUB" in out
    listing = run("--admin", "wallet", "list").output
    assert "system:escrow" in listing and "сходится" in listing


def test_trace_shows_orchestrator_work(seeded, config_file, monkeypatch):
    import logging

    use_llm(monkeypatch, analysis(), TURKEY_PLAN)
    result = CliRunner().invoke(app, ["--config", str(config_file), "--trace", "chat", "--user", "ivanov"],
                                input="Хочу в Турцию\n/exit\n")
    assert result.exit_code == 0
    handlers = logging.getLogger().handlers
    assert any(type(h).__name__ == "RichHandler" for h in handlers)
    log_text = (seeded.path("logs_dir") / "agentic.log").read_text(encoding="utf-8")
    assert "Ставка" in log_text and "план из 5 подзадач" in log_text
