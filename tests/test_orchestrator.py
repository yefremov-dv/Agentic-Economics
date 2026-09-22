from datetime import date

import pytest

from agentic_economics.orchestrator import Orchestrator, OrchestratorError
from agentic_economics.orchestrator import taskfile
from agentic_economics.orchestrator.planner import normalize
from agentic_economics.orchestrator.schemas import Decomposition
from agentic_economics.storage.models import Message, Task
from tests.fakes import TURKEY_PLAN, FakeLLM, analysis, task

TODAY = date(2026, 9, 22)
REQUEST = ("Хочу через неделю полететь в Турцию на 6 ночей в отель Maxx Royal Belek, бюджет 1 000 000 рублей. "
           "Хочу, чтобы на мне были розовые кроссовки Nike.")


def start(seeded, db, *responses):
    llm = FakeLLM(*responses)
    return Orchestrator.start(seeded, db, llm, "ivanov", today=TODAY), llm


def test_clarification_then_plan(seeded, db):
    orch, llm = start(seeded, db,
                      analysis(ready=False, questions=["Город вылета?", "Размер обуви?"], budget=1_000_000),
                      analysis(ready=True, budget=1_000_000),
                      TURKEY_PLAN)
    assert orch.session.id.startswith("2026-09-22-")

    reply = orch.handle(REQUEST)
    assert reply.kind == "questions"
    assert reply.questions == ["Город вылета?", "Размер обуви?"]
    assert orch.tasks() == []
    assert orch.session.budget == 100_000_000  # в копейках

    reply = orch.handle("Из Москвы, один, размер 42")
    assert reply.kind == "plan"
    assert [m.role for m in orch.messages()] == ["user", "assistant", "user", "plan"]

    # Модель видит весь диалог и дату
    schema, messages = llm.calls[-1]
    assert schema is Decomposition
    assert "22.09.2026" in messages[0][1]
    assert messages[1] == ("user", REQUEST)
    assert "1 000 000.00 VRUB" in messages[0][1]


def test_plan_files_and_rows(seeded, db):
    orch, _ = start(seeded, db, analysis(budget=1_000_000), TURKEY_PLAN)
    orch.handle(REQUEST)
    tasks = orch.tasks()
    by_title = {t.title: t for t in tasks}

    # Трансфер переставлен после рейса и отеля, нумерация сквозная с 1
    assert [t.number for t in tasks] == [1, 2, 3, 4, 5]
    transfer = by_title["Трансфер аэропорт ↔ отель"]
    flights, hotel = by_title["Авиабилеты Москва — Анталья и обратно"], by_title["Отель Maxx Royal Belek, 6 ночей"]
    assert sorted(transfer.depends_on) == sorted([flights.number, hotel.number])
    assert all(d < t.number for t in tasks for d in t.depends_on)

    insurance = by_title["Страховка для выезда за рубеж"]
    assert insurance.recommended and not insurance.enabled  # FR-9a
    assert all(t.status == "awaiting_approval" for t in tasks)
    assert orch.session.status == "awaiting_approval"

    folder = seeded.path("tasks_dir") / orch.session.id
    assert sorted(p.name for p in folder.iterdir()) == [f"task-{n}.md" for n in range(1, 6)]
    meta = taskfile.read_meta(folder / f"task-{transfer.number}.md")
    assert meta["id"] == f"task-{transfer.number}"
    assert meta["session"] == orch.session.id
    assert meta["category"] == "transfers"
    assert sorted(meta["depends_on"]) == sorted([f"task-{flights.number}", f"task-{hotel.number}"])
    assert meta["budget_hint"] == 10000
    text = (folder / f"task-{transfer.number}.md").read_text(encoding="utf-8")
    assert "# Трансфер аэропорт ↔ отель" in text
    assert "## Критерии выполнения\n- Критерий" in text


def test_clarification_rounds_are_limited(seeded, db):
    seeded.orchestrator.max_clarification_rounds = 1
    orch, llm = start(seeded, db,
                      analysis(ready=False, questions=["Город?"]),
                      analysis(ready=False, questions=["Ещё вопрос?"]),  # лимит исчерпан — строим план
                      TURKEY_PLAN)
    assert orch.handle(REQUEST).kind == "questions"
    assert orch.handle("Москва").kind == "plan"
    assert "Больше не спрашивай" in llm.calls[1][1][0][1]


def test_change_requirements_rebuilds_plan(seeded, db):
    small = {"tasks": [task(1, "Авиабилеты", "flights"), task(2, "Отель", "hotels")]}
    orch, _ = start(seeded, db, analysis(), TURKEY_PLAN, analysis(), small)
    orch.handle(REQUEST)
    reply = orch.handle("Кроссовки не нужны, трансфер тоже")
    assert reply.kind == "plan" and reply.text.startswith("План обновлён")
    assert [t.title for t in orch.tasks()] == ["Авиабилеты", "Отель"]
    folder = seeded.path("tasks_dir") / orch.session.id
    assert sorted(p.name for p in folder.iterdir()) == ["task-1.md", "task-2.md"]
    assert db.query(Task).count() == 2


def test_invalid_llm_answer_does_not_break_session(seeded, db):
    orch, llm = start(seeded, db, ValueError("битый JSON"), ValueError("битый JSON"),
                      analysis(), TURKEY_PLAN)
    reply = orch.handle(REQUEST)
    assert reply.kind == "error"
    assert len(llm.calls) == 2  # одна повторная попытка
    assert orch.tasks() == []
    # Сессия жива: следующее сообщение обрабатывается
    assert orch.handle("Попробуй ещё раз").kind == "plan"


def test_one_retry_recovers(seeded, db):
    orch, _ = start(seeded, db, analysis(), ValueError("битый JSON"), TURKEY_PLAN)
    assert orch.handle(REQUEST).kind == "plan"


def test_resume_and_cancel(seeded, db):
    orch, _ = start(seeded, db, analysis(), TURKEY_PLAN)
    orch.handle(REQUEST)
    same = Orchestrator.resume(seeded, db, FakeLLM(), "ivanov", orch.session.id)
    assert len(same.tasks()) == 5

    with pytest.raises(OrchestratorError):
        Orchestrator.resume(seeded, db, FakeLLM(), "petrova", orch.session.id)  # чужая сессия

    same.cancel()
    assert all(t.status == "cancelled" for t in same.tasks())
    meta = taskfile.read_meta(seeded.path("tasks_dir") / orch.session.id / "task-1.md")
    assert meta["status"] == "cancelled"
    assert same.handle("ещё").kind == "error"
    with pytest.raises(OrchestratorError):
        Orchestrator.resume(seeded, db, FakeLLM(), "ivanov", orch.session.id)


def test_disabled_category_is_skipped(seeded, db):
    from agentic_economics.storage.models import Agent

    for a in db.query(Agent).filter(Agent.category == "insurance"):
        a.active = False
    db.commit()
    orch, llm = start(seeded, db, analysis(), TURKEY_PLAN)
    reply = orch.handle(REQUEST)
    assert "insurance" not in llm.calls[0][1][0][1]
    assert [t.category for t in reply.tasks].count("insurance") == 0
    assert any("insurance" in w for w in reply.warnings)


def test_normalize_drops_bad_dependencies_and_cycles():
    plan = Decomposition.model_validate({"tasks": [
        task(1, "A", "flights", depends_on=[2]),
        task(2, "B", "hotels", depends_on=[1, 99, 2]),
        task(3, "C", "transfers", depends_on=[1, 2, 2]),
    ]})
    result = normalize(plan, {"flights", "hotels", "transfers"})
    by_title = {t.title: t for t in result.tasks}
    assert [t.number for t in result.tasks] == [1, 2, 3]
    assert all(d < t.number for t in result.tasks for d in t.depends_on)
    assert by_title["C"].depends_on == [1, 2]
    assert any("Циклическая" in w for w in result.warnings)


def test_messages_persist(seeded, db):
    orch, _ = start(seeded, db, analysis(ready=False, questions=["Город?"]))
    orch.handle(REQUEST)
    assert db.query(Message).filter(Message.session_id == orch.session.id).count() == 2
