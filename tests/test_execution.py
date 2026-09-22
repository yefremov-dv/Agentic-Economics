from datetime import date

import pytest
from sqlalchemy import select

from agentic_economics.agents.executor import make_reference
from agentic_economics.orchestrator import Orchestrator, OrchestratorError, taskfile
from agentic_economics.storage.models import Agent, Bid, Rating, Transaction, Wallet
from agentic_economics.wallet.escrow import held
from agentic_economics.wallet.ledger import balance, find_wallet, topup
from tests.fakes import TURKEY_PLAN, FakeLLM, analysis, task

TODAY = date(2026, 9, 22)
START = 1_500_000_00


def total_money(db) -> int:
    return sum(balance(db, w) for w in db.scalars(select(Wallet)))


def client_balance(db, user="ivanov") -> int:
    return balance(db, find_wallet(db, "client", user))


def planned(seeded, db, llm, user="ivanov"):
    orch = Orchestrator.start(seeded, db, llm, user, today=TODAY)
    orch.handle("Хочу в Турцию")
    return orch


def by_category(orch, category):
    return next(t for t in orch.tasks() if t.category == category)


# ------------------------------------------------------------ подтверждение (FR-15 – FR-17)


def test_approval_view(seeded, db):
    orch = planned(seeded, db, FakeLLM(analysis(budget=1_000_000), TURKEY_PLAN))
    a = orch.approval()
    assert len(a.lines) == 5
    enabled = [l for l in a.lines if l.task.enabled]
    assert a.total == sum(l.winner.price for l in enabled)
    assert a.fee == sum(round(l.winner.price * 0.05) for l in enabled)
    assert a.budget_left == 1_000_000_00 - a.total
    assert a.balance == START
    flights = next(l for l in a.lines if l.task.category == "flights")
    assert 1 <= len(flights.alternatives) <= 2 and all(b.status == "lost" for b in flights.alternatives)


def test_nothing_is_charged_before_confirm(seeded, db):
    orch = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    before = db.query(Transaction).count()
    orch.remove("task-4")
    orch.set_enabled(taskfile.task_key(by_category(orch, "insurance").number), True)
    orch.cancel()  # критерий приёмки 5: отказ не создаёт транзакций
    assert db.query(Transaction).count() == before
    assert client_balance(db) == START


def test_plan_edits(seeded, db):
    orch = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    hotel = by_category(orch, "hotels")
    key = taskfile.task_key(hotel.number)
    other = next(b for b in orch.bids(hotel) if b.status == "lost")
    assert other.agent_id in orch.replace(key, other.agent_id)
    assert orch.winner(hotel).id == other.id
    assert sorted(b.status for b in orch.bids(hotel)) == ["lost", "winner"]
    assert taskfile.read_meta(seeded.path("tasks_dir") / orch.session.id / f"{key}.md")["winner"] == other.agent_id

    with pytest.raises(OrchestratorError, match="нет действующей ставки"):
        orch.replace(key, "flights-skylines")
    with pytest.raises(OrchestratorError, match="не рекомендованная"):
        orch.set_enabled(key, False)
    assert "убрана" in orch.remove(key)
    with pytest.raises(OrchestratorError, match="уже убрана"):
        orch.remove(key)
    with pytest.raises(OrchestratorError, match="не найдена"):
        orch.remove("task-99")


def test_removing_dependency_warns(seeded, db):
    orch = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    flights = by_category(orch, "flights")
    assert "зависят" in orch.remove(taskfile.task_key(flights.number))


def test_confirm_requires_money(seeded, db):
    orch = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN), user="sidorov")
    wallet = find_wallet(db, "client", "sidorov")
    from agentic_economics.wallet.ledger import get_wallet, transfer

    transfer(db, wallet, get_wallet(db, "provider", "sky-lines"), START - 1000, "payout")
    db.commit()
    before = db.query(Transaction).count()
    with pytest.raises(OrchestratorError, match="Недостаточно средств"):  # FR-25
        orch.confirm()
    assert db.query(Transaction).count() == before
    assert orch.session.status == "awaiting_approval"
    topup(db, wallet, START)
    db.commit()
    assert orch.confirm().outcomes


def test_confirm_blocks_task_without_bids(seeded, db):
    decline = {"bid": {"participate": False, "offer": "нет", "eta": "", "confidence": 0}}
    llm = FakeLLM(analysis(), TURKEY_PLAN, agents={"КроссМаркет": decline, "СпортЛайн": decline})
    orch = planned(seeded, db, llm)
    with pytest.raises(OrchestratorError, match="нет исполнителя"):
        orch.confirm()
    orch.remove(taskfile.task_key(by_category(orch, "retail").number))
    assert orch.confirm()


# ------------------------------------------------------------ выполнение (FR-18 – FR-20, FR-22)


def test_confirm_executes_and_pays(seeded, db):
    llm = FakeLLM(analysis(budget=1_000_000), TURKEY_PLAN)
    orch = planned(seeded, db, llm)
    approval = orch.approval()
    execution = orch.confirm()

    statuses = {o.task.category: o.status for o in execution.outcomes}
    assert statuses == {"flights": "done", "hotels": "done", "transfers": "done", "retail": "done"}
    insurance = by_category(orch, "insurance")
    assert insurance.status == "cancelled" and held(db, insurance.id) == 0  # рекомендованная не включена — не платим

    assert client_balance(db) == START - approval.total
    assert balance(db, find_wallet(db, "platform", "platform")) == approval.fee
    for o in execution.outcomes:
        assert held(db, o.task.id) == 0
        assert o.settlement.payout + o.settlement.fee == orch.winner(o.task).price
        assert o.reference and o.reference in o.task.result_json["summary"]
    assert total_money(db) == 0
    assert orch.session.status == "done"

    # Порядок: трансфер выполняется после рейса и отеля и видит их результаты
    transfer_call = next(m for m in llm.executions if "Трансфер" in m[1].content)
    assert "Связанные подзадачи" in transfer_call[1].content and "Номер подтверждения" in transfer_call[1].content

    kinds = {tx.kind for tx in db.scalars(select(Transaction).where(Transaction.task_id.isnot(None)))}
    assert kinds == {"hold", "payout", "fee"}

    report = execution.report_path.read_text(encoding="utf-8")
    assert "Итоги сессии" in report and "выполнена" in report
    assert "в том числе комиссия платформы" in report and "Бюджет: 1 000 000.00 VRUB" in report
    assert orch.handle("ещё").kind == "error"  # после подтверждения требования не меняются


def test_retry_after_rejected_result(seeded, db):
    bad = {"success": True, "summary": "Забронирован другой отель", "details": []}
    good = {"success": True, "summary": "Забронирован Maxx Royal Belek", "details": []}
    verify = lambda prompt: {"ok": "другой" not in prompt, "reason": "не тот отель"}
    llm = FakeLLM(analysis(), TURKEY_PLAN, verify=verify,
                  agents={"Санрайз": {"execute": [bad, good]}, "Босфор": {"execute": [bad, good]}})
    orch = planned(seeded, db, llm)
    outcome = next(o for o in orch.confirm().outcomes if o.task.category == "hotels")
    assert outcome.status == "done"
    assert [a.ok for a in outcome.attempts] == [False, True]
    assert outcome.attempts[0].reason == "не тот отель"
    feedback = [m for m in llm.executions if "не принял прошлый отчёт: не тот отель" in m[1].content]
    assert feedback


def test_runner_up_takes_over(seeded, db):
    fail = {"success": False, "summary": "Мест нет", "details": []}
    llm = FakeLLM(analysis(), TURKEY_PLAN)
    orch = planned(seeded, db, llm)
    hotel = by_category(orch, "hotels")
    winner, runner_up = orch.winner(hotel), next(b for b in orch.bids(hotel) if b.status == "lost")
    # Второе место дешевле победителя — переход без доплаты возможен
    runner_up.price = winner.price - 1000_00
    db.commit()
    winner_agent = db.get(Agent, winner.agent_id).name.split()[0]
    llm.agents = {winner_agent: {"execute": fail}}
    before = client_balance(db)
    outcome = next(o for o in orch.confirm().outcomes if o.task.category == "hotels")

    assert outcome.status == "done" and outcome.agent_id == runner_up.agent_id
    assert [a.agent_id for a in outcome.attempts] == [winner.agent_id, winner.agent_id, runner_up.agent_id]
    assert db.get(Bid, winner.id).status == "failed" and db.get(Bid, runner_up.id).status == "winner"
    refunds = db.scalars(select(Transaction).where(Transaction.task_id == hotel.id, Transaction.kind == "refund")).all()
    assert [r.amount for r in refunds] == [1000_00]  # разница цены вернулась клиенту
    assert held(db, hotel.id) == 0
    assert total_money(db) == 0
    # С клиента списано ровно то, что получили провайдеры и платформа
    paid = sum(tx.amount for tx in db.scalars(select(Transaction).where(Transaction.kind.in_(("payout", "fee")))))
    assert before - client_balance(db) == paid


def test_failed_task_is_refunded_and_dependents_cancelled(seeded, db):
    fail = {"success": False, "summary": "Рейсов нет", "details": []}
    llm = FakeLLM(analysis(), TURKEY_PLAN, agents={"Небесные": {"execute": fail}, "ТурбоАвиа": {"execute": fail},
                                                   "Санрайз Тревел», отдел перелётов": {"execute": fail}})
    orch = planned(seeded, db, llm)
    flights = by_category(orch, "flights")
    # Второе место дороже победителя: без подтверждения клиента доплату не берём (FR-17)
    for b in orch.bids(flights):
        if b.status == "lost":
            b.price = orch.winner(flights).price + 1
    db.commit()
    before = client_balance(db)
    outcomes = {o.task.category: o for o in orch.confirm().outcomes}

    assert outcomes["flights"].status == "failed"
    assert "второго места" in outcomes["flights"].note
    assert outcomes["transfers"].status == "cancelled" and "task-" in outcomes["transfers"].note
    assert outcomes["hotels"].status == "done"
    paid = sum(o.settlement.payout + o.settlement.fee for o in outcomes.values() if o.settlement)
    assert before - client_balance(db) == paid
    kinds = {tx.kind for tx in db.scalars(select(Transaction).where(Transaction.task_id == outcomes["transfers"].task.id))}
    assert kinds == {"hold", "release"}
    assert total_money(db) == 0
    report = (seeded.path("tasks_dir") / orch.session.id / "result.md").read_text(encoding="utf-8")
    assert "не выполнена" in report and "отменена" in report and "Возвращено из эскроу" in report


def test_executor_error_and_verify_failure(seeded, db):
    llm = FakeLLM(analysis(), {"tasks": [task(1, "Отель Maxx Royal Belek", "hotels")]},
                  agents={"Санрайз": {"execute_error": RuntimeError("503")}, "Босфор": {"execute_error": RuntimeError("503")}})
    orch = planned(seeded, db, llm)
    outcome = orch.confirm().outcomes[0]
    assert outcome.status == "failed"
    assert all("ошибка исполнителя" in a.reason for a in outcome.attempts)
    assert client_balance(db) == START

    # Сбой самой проверки не наказывает исполнителя
    def broken(prompt):
        raise ValueError("битый JSON")

    llm2 = FakeLLM(analysis(), {"tasks": [task(1, "Отель Maxx Royal Belek", "hotels")]}, verify=broken)
    import agentic_economics.auction.runner as runner

    runner.RETRY_DELAYS, saved = (0.0,), runner.RETRY_DELAYS
    try:
        orch2 = planned(seeded, db, llm2, user="petrova")
        assert orch2.confirm().outcomes[0].status == "done"
    finally:
        runner.RETRY_DELAYS = saved


def test_references_format():
    assert len(make_reference("flights")) == 6 and make_reference("flights").isupper()
    assert make_reference("hotels").startswith("HB-")
    assert make_reference("other").startswith("REF-")


# ------------------------------------------------------------ оценки (FR-32 – FR-34)


def test_ratings(seeded, db):
    orch = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    orch.confirm()
    hotel = by_category(orch, "hotels")
    key = taskfile.task_key(hotel.number)
    agent_id = orch.winner(hotel).agent_id
    agent = orch.rate(key, 3, "Номер не тот")
    assert (agent.rating, agent.ratings_count) == (3.0, 1)
    with pytest.raises(OrchestratorError, match="уже оценена"):
        orch.rate(key, 5)
    with pytest.raises(OrchestratorError, match="от 1 до 5"):
        orch.rate(taskfile.task_key(by_category(orch, "retail").number), 6)
    with pytest.raises(OrchestratorError, match="не выполнена"):
        orch.rate(taskfile.task_key(by_category(orch, "insurance").number), 5)

    # Вторая сессия того же агента: рейтинг — среднее, округлённое до 0,1 (FR-33)
    orch2 = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN), user="petrova")
    hotel2 = by_category(orch2, "hotels")
    orch2.replace(taskfile.task_key(hotel2.number), agent_id) if orch2.winner(hotel2).agent_id != agent_id else None
    orch2.confirm()
    agent = orch2.rate(taskfile.task_key(hotel2.number), 4)
    assert (agent.rating, agent.ratings_count) == (3.5, 2)
    assert db.query(Rating).count() == 2

    # Оценка чужой подзадачи невозможна
    from agentic_economics.agents.rating import RatingError, rate

    with pytest.raises(RatingError, match="не найдена"):
        rate(db, hotel2, "ivanov", 5)


def test_rating_survives_reload_and_affects_auction(seeded, db):
    from agentic_economics.agents.registry import sync_agents

    orch = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    orch.confirm()
    hotel = by_category(orch, "hotels")
    agent = orch.rate(taskfile.task_key(hotel.number), 1)
    sync_agents(db, seeded.path("agents_dir"))  # FR-34: reload не сбрасывает оценки
    assert db.get(Agent, agent.id).rating == 1.0

    from agentic_economics.auction.scoring import Offer, score_offers

    orch2 = planned(seeded, db, FakeLLM(analysis(), TURKEY_PLAN), user="petrova")
    bids = [b for b in orch2.bids(by_category(orch2, "hotels")) if b.score is not None]
    # Следующий аукцион считает оценку ставки уже с новым рейтингом 1.0 (FR-33)
    expected = score_offers([Offer(b.price, b.confidence, db.get(Agent, b.agent_id).rating) for b in bids],
                            seeded.auction.weights)
    assert [b.score for b in bids] == pytest.approx(expected)
    assert db.get(Agent, agent.id).rating == 1.0


def test_single_event_loop_for_all_async_steps(seeded, db):
    # Клиент GigaChat привязан к циклу событий: аукцион и все волны выполнения должны идти в одном
    llm = FakeLLM(analysis(), TURKEY_PLAN)
    orch = planned(seeded, db, llm)
    orch.confirm()
    assert len(llm.loops) == 1
