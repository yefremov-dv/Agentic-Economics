import asyncio
import shutil
from datetime import date
from pathlib import Path

import pytest

from agentic_economics.agents import catalog
from agentic_economics.agents.executor import BidDraft, ExecutorAgent, TaskBrief
from agentic_economics.agents.registry import sync_agents
from agentic_economics.auction.candidates import select_candidates
from agentic_economics.auction.scoring import Offer, pick_winner, rating_norm, score_offers
from agentic_economics.config import AuctionWeights
from agentic_economics.orchestrator import Orchestrator, taskfile
from agentic_economics.storage.models import Agent, Bid, Task
from tests.fakes import TURKEY_PLAN, FakeLLM, analysis, task

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 9, 22)
W = AuctionWeights(price=0.5, confidence=0.3, rating=0.2)
NO_ETA = {"eta": "", "confidence": 0.5}


# ------------------------------------------------------------ выбор победителя (FR-12, FR-12a)


def test_score_formula():
    scores = score_offers([Offer(100_00, 0.9, 5.0), Offer(200_00, 0.5, 3.0)], W)
    assert scores[0] == pytest.approx(0.5 * 1 + 0.3 * 0.9 + 0.2 * 1.0)
    assert scores[1] == pytest.approx(0.5 * 0 + 0.3 * 0.5 + 0.2 * 0.5)
    assert pick_winner(scores) == 0


def test_price_norm_is_relative_to_bids():
    scores = score_offers([Offer(100_00, 0.5, 4.0), Offer(150_00, 0.5, 4.0), Offer(200_00, 0.5, 4.0)], W)
    base = 0.3 * 0.5 + 0.2 * rating_norm(4.0)
    assert scores == pytest.approx([base + 0.5, base + 0.25, base])


def test_single_bid_and_equal_prices_get_full_price_score():
    assert score_offers([Offer(500_00, 1.0, 5.0)], W) == pytest.approx([1.0])
    a, b = score_offers([Offer(10, 0.5, 4.0), Offer(10, 0.5, 4.0)], W)
    assert a == b


def test_tie_goes_to_earlier_bid():
    assert pick_winner([0.7, 0.9, 0.9]) == 1
    assert pick_winner([0.5, 0.5]) == 0
    assert pick_winner([]) is None


def test_confidence_and_rating_are_clamped():
    assert rating_norm(0) == 0 and rating_norm(9) == 1
    assert score_offers([Offer(1, 5.0, 5.0)], W) == pytest.approx([1.0])
    assert score_offers([], W) == []


def test_price_beats_confidence_with_default_weights():
    # Дешевле вдвое, но уверенность ниже — при весе цены 0.5 выигрывает дешёвая ставка
    scores = score_offers([Offer(300_00, 0.95, 4.5), Offer(150_00, 0.7, 4.5)], W)
    assert pick_winner(scores) == 1


# ------------------------------------------------------------ каталог


def test_catalog_search_and_sellers():
    items = catalog.load_items(ROOT / "data/mock/hotels.json")
    bosphorus = catalog.for_provider(items, "bosphorus")
    assert all("bosphorus" in i["sellers"] for i in bosphorus)
    found = catalog.search(bosphorus, "отель Maxx Royal Belek")
    assert found and all(i["hotel"] == "Maxx Royal Belek" for i in found)
    flights = catalog.for_provider(catalog.load_items(ROOT / "data/mock/flights.json"), "turboavia")
    # «Анталию», «Анталия» и «Анталья» находят одно и то же
    assert catalog.search(flights, "Москва — Анталию")[0]["from_city"] == "Москва"
    assert catalog.search(flights, "xyz qwerty") == []
    assert len(catalog.search(flights, "", limit=3)) == 3


def test_catalog_size_is_matched_as_whole_number():
    items = catalog.load_items(ROOT / "data/mock/products.json")
    found = catalog.search(items, "розовые Nike 36")
    assert 36 in found[0]["sizes_in_stock"]


def test_mock_catalogs_cover_all_agents():
    from agentic_economics.agents.spec import load_spec

    for path in (ROOT / "agents").glob("*.yaml"):
        spec = load_spec(path)
        items = catalog.for_provider(catalog.load_items(ROOT / spec.mock_data), spec.provider)
        assert items, f"у агента {spec.id} пустой каталог"


# ------------------------------------------------------------ агент-исполнитель


def executor(db, agent_id, llm, root=ROOT):
    agent = db.get(Agent, agent_id)
    return ExecutorAgent(agent.id, agent.config_json, llm, root, TODAY)


BRIEF = TaskBrief(1, "task-2", "Отель Maxx Royal Belek, 6 ночей", "hotels", "29.09–05.10, 1 взрослый", ["Бронь"])


def test_bid_price_is_computed_from_catalog(db):
    llm = FakeLLM(agents={"Санрайз": {"bid": {"participate": True, "items": [{"item_id": "mrb-deluxe-sr", "quantity": 6}],
                                               "eta": "2 часа", "confidence": 0.9, "offer": "Deluxe, 6 ночей"}}})
    agent = executor(db, "hotels-sunrise", llm)
    result = asyncio.run(agent.bid(BRIEF))
    assert result.status == "submitted"
    assert result.price == round(89000 * 1.07) * 6 * 100  # наценка 7 %, в копейках
    assert result.details["items"][0]["quantity"] == 6
    assert result.details["queries"]


def test_min_price_applies(db):
    llm = FakeLLM(agents={"Кара": {"bid": {"participate": True, "items": [{"item_id": "ka-belek-sedan", "quantity": 1}],
                                            **NO_ETA}}})
    agent = executor(db, "transfers-kara", llm)
    agent.min_price = 50_000
    assert asyncio.run(agent.bid(BRIEF)).price == 50_000_00


@pytest.mark.parametrize("draft,status", [
    ({"participate": False, "offer": "Нет такого отеля", **NO_ETA}, "declined"),
    ({"participate": True, "items": [{"item_id": "выдуманный", "quantity": 1}], **NO_ETA}, "declined"),
    ({"participate": True, "items": [{"item_id": "mrb-deluxe-bo", "quantity": 0}], **NO_ETA}, "declined"),
])
def test_invalid_or_declined_bids(db, draft, status):
    llm = FakeLLM(agents={"Босфор": {"bid": draft}})
    assert asyncio.run(executor(db, "hotels-bosphorus", llm).bid(BRIEF)).status == status


def test_competitor_items_are_not_accepted(db):
    # Позиция из каталога другого провайдера не считается
    llm = FakeLLM(agents={"Босфор": {"bid": {"participate": True, "items": [{"item_id": "mrb-deluxe-sr", "quantity": 6}], **NO_ETA}}})
    assert asyncio.run(executor(db, "hotels-bosphorus", llm).bid(BRIEF)).status == "declined"


def test_search_fallback_when_model_skips_tool(db):
    llm = FakeLLM(agents={"Босфор": {"no_tool": True}})
    result = asyncio.run(executor(db, "hotels-bosphorus", llm).bid(BRIEF))
    assert result.status == "submitted"
    assert result.details["queries"] == [BRIEF.title]


def test_tool_hides_sellers_and_applies_markup(db):
    agent = executor(db, "hotels-bosphorus", FakeLLM())
    found = agent.search("Maxx Royal")
    assert found and all("sellers" not in i for i in found)
    assert found[0]["price"] == round(84500 * 1.04)


# ------------------------------------------------------------ отбор кандидатов (FR-10)


def test_candidates_by_category_and_limit(db):
    assert {a.id for a in select_candidates(db, "flights", 5)} == {"flights-skylines", "flights-sunrise", "flights-turboavia"}
    ranked = FakeLLM({"agent_ids": ["flights-turboavia", "nope", "flights-sunrise"]})
    assert [a.id for a in select_candidates(db, "flights", 2, "рейс", ranked)] == ["flights-turboavia", "flights-sunrise"]
    broken = FakeLLM(ValueError("битый JSON"))
    assert len(select_candidates(db, "flights", 2, "рейс", broken)) == 2  # запасной вариант — по рейтингу


# ------------------------------------------------------------ аукцион целиком


def plan_and_auction(seeded, db, llm):
    orch = Orchestrator.start(seeded, db, llm, "ivanov", today=TODAY)
    return orch, orch.handle("Хочу в Турцию")


def test_every_task_gets_at_least_two_bids(seeded, db):
    orch, reply = plan_and_auction(seeded, db, FakeLLM(analysis(budget=1_000_000), TURKEY_PLAN))
    assert reply.kind == "plan"
    for t in orch.tasks():
        valid = [b for b in orch.bids(t) if b.status in ("winner", "lost")]
        assert len(valid) >= 2, t.title  # критерий приёмки 2
        winner = orch.winner(t)
        assert winner.status == "winner"
        assert winner.score == max(b.score for b in valid)
        assert t.status == "awaiting_approval"
        meta = taskfile.read_meta(seeded.path("tasks_dir") / orch.session.id / f"task-{t.number}.md")
        assert meta["winner"] == winner.agent_id
        assert meta["price"] == winner.price / 100
    assert orch.session.status == "awaiting_approval"
    assert "исполнитель" in orch.messages()[-1].content


def test_timeout_and_error_do_not_break_auction(seeded, db):
    seeded.auction.bid_timeout = 0.2
    llm = FakeLLM(analysis(), TURKEY_PLAN, agents={"Кара": {"delay": 1}, "ТурбоАвиа": {"error": RuntimeError("503")}})
    orch, reply = plan_and_auction(seeded, db, llm)
    statuses = {b.agent_id: b.status for t in orch.tasks() for b in orch.bids(t)}
    assert statuses["transfers-kara"] == "timeout"
    assert statuses["flights-turboavia"] == "error"
    transfer = next(t for t in orch.tasks() if t.category == "transfers")
    assert orch.winner(transfer).agent_id == "transfers-sunrise"


def test_task_without_bids(seeded, db):
    decline = {"bid": {"participate": False, "offer": "нет", **NO_ETA}}
    llm = FakeLLM(analysis(), {"tasks": [task(1, "Кроссовки", "retail")]},
                  agents={"КроссМаркет": decline, "СпортЛайн": decline})
    orch, reply = plan_and_auction(seeded, db, llm)
    t = orch.tasks()[0]
    assert t.winner_bid_id is None and t.status == "awaiting_approval"
    assert any("ни одной ставки" in w for w in reply.warnings)
    assert "winner" not in taskfile.read_meta(seeded.path("tasks_dir") / orch.session.id / "task-1.md")


def test_budget_overrun_is_reported(seeded, db):
    orch, reply = plan_and_auction(seeded, db, FakeLLM(analysis(budget=1000), TURKEY_PLAN))
    assert any("превышает бюджет" in w for w in reply.warnings)
    orch2, reply2 = plan_and_auction(seeded, db, FakeLLM(analysis(budget=10_000_000), TURKEY_PLAN))
    assert not any("бюджет" in w for w in reply2.warnings)


def test_recommended_task_is_auctioned_but_not_counted(seeded, db):
    orch, _ = plan_and_auction(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    insurance = next(t for t in orch.tasks() if t.category == "insurance")
    assert insurance.recommended and not insurance.enabled
    assert orch.winner(insurance) is not None


def test_replan_removes_old_bids(seeded, db):
    llm = FakeLLM(analysis(), TURKEY_PLAN, analysis(), {"tasks": [task(1, "Авиабилеты Москва Анталья", "flights")]})
    orch, _ = plan_and_auction(seeded, db, llm)
    orch.handle("Только билеты")
    assert db.query(Task).count() == 1
    assert {b.task_id for b in db.query(Bid)} == {orch.tasks()[0].id}


def test_new_agent_joins_next_auction(seeded, db):
    agents_dir = seeded.path("agents_dir")
    text = (agents_dir / "retail-sportline.yaml").read_text(encoding="utf-8")
    (agents_dir / "retail-newshop.yaml").write_text(
        text.replace("id: retail-sportline", "id: retail-newshop").replace("СпортЛайн", "НьюШоп"), encoding="utf-8")
    sync_agents(db, agents_dir)
    llm = FakeLLM(analysis(), {"tasks": [task(1, "Розовые кроссовки Nike", "retail")]})
    orch, _ = plan_and_auction(seeded, db, llm)
    assert "retail-newshop" in {b.agent_id for b in orch.bids(orch.tasks()[0])}  # критерий приёмки 6


def test_dependencies_are_passed_as_context(seeded, db):
    llm = FakeLLM(analysis(), TURKEY_PLAN)
    plan_and_auction(seeded, db, llm)
    transfer_prompts = [m for m in llm.bid_calls if "Трансфер" in m[1].content]
    assert transfer_prompts and "Связанные подзадачи" in transfer_prompts[0][1].content


@pytest.mark.parametrize("size,status", [("42", "submitted"), ("39", "declined"), (None, "declined")])
def test_size_must_be_in_stock(db, size, status):
    # У Nike Revolution 7 в «СпортЛайне» нет 42-го размера, у Air Force 1 — есть
    item = "sp-af1-pink" if status == "submitted" else "sp-revolution-pink"
    if size is None:
        item = "sp-af1-pink"
    llm = FakeLLM(agents={"СпортЛайн": {"bid": {"participate": True, "items": [{"item_id": item, "quantity": 1, "size": size}],
                                                **NO_ETA}}})
    result = asyncio.run(executor(db, "retail-sportline", llm).bid(BRIEF))
    assert result.status == status
    if status == "declined":
        assert "нет в наличии" in result.offer
    if status == "submitted":
        assert "размер 42" in result.offer or result.details["items"][0]["size"] == "42"


def test_rate_limit_is_retried(seeded, db, monkeypatch):
    monkeypatch.setattr("agentic_economics.auction.runner.RETRY_DELAYS", (0.01, 0.01))

    class RateLimitError(Exception):
        pass

    calls = {"n": 0}
    real = ExecutorAgent.bid

    async def flaky(self, brief):
        if self.id == "hotels-sunrise" and calls["n"] < 2:
            calls["n"] += 1
            raise RateLimitError("429")
        return await real(self, brief)

    monkeypatch.setattr(ExecutorAgent, "bid", flaky)
    orch, _ = plan_and_auction(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    hotel = next(t for t in orch.tasks() if t.category == "hotels")
    assert {b.agent_id: b.status for b in orch.bids(hotel)}["hotels-sunrise"] in ("winner", "lost")
    assert calls["n"] == 2


def test_missing_confidence_is_neutral(db):
    llm = FakeLLM(agents={"Босфор": {"bid": {"participate": True, "items": [{"item_id": "mrb-deluxe-bo", "quantity": 6}]}}})
    result = asyncio.run(executor(db, "hotels-bosphorus", llm).bid(BRIEF))
    assert result.status == "submitted" and result.confidence == 0.5 and result.eta == ""


def test_reset_demo_after_auction(seeded, db):
    from agentic_economics.seed import reset_demo

    plan_and_auction(seeded, db, FakeLLM(analysis(), TURKEY_PLAN))
    db.close()
    reset_demo(seeded)  # взаимные внешние ключи tasks ↔ bids не мешают пересозданию
    from agentic_economics.storage.db import make_sessionmaker

    with make_sessionmaker(seeded)() as fresh:
        assert fresh.query(Bid).count() == 0 and fresh.query(Task).count() == 0


def test_agent_gets_one_chance_to_fix_bid(db):
    wrong = {"participate": True, "items": [{"item_id": "sp-revolution-pink", "quantity": 1, "size": "42"}], **NO_ETA}
    right = {"participate": True, "items": [{"item_id": "sp-af1-pink", "quantity": 1, "size": "42"}], **NO_ETA}
    llm = FakeLLM(agents={"СпортЛайн": {"bids": [wrong, right]}})
    result = asyncio.run(executor(db, "retail-sportline", llm).bid(BRIEF))
    assert result.status == "submitted"
    assert result.details["items"][0]["id"] == "sp-af1-pink"
    assert "размера 42 нет в наличии" in llm.bid_calls[-1][-1].content

    llm = FakeLLM(agents={"СпортЛайн": {"bids": [wrong, wrong]}})
    assert asyncio.run(executor(db, "retail-sportline", llm).bid(BRIEF)).status == "declined"


def test_flight_legs_must_connect(db):
    brief = TaskBrief(1, "task-1", "Авиабилеты Москва — Анталья и обратно", "flights", "1 взрослый", [])
    broken = {"participate": True, "items": [{"item_id": "ta-7401", "quantity": 1}, {"item_id": "ta-7602", "quantity": 1}],
              **NO_ETA}
    fixed = {"participate": True, "items": [{"item_id": "ta-7401", "quantity": 1}, {"item_id": "ta-7402", "quantity": 1}],
             **NO_ETA}
    llm = FakeLLM(agents={"ТурбоАвиа": {"bids": [broken, fixed]}})
    result = asyncio.run(executor(db, "flights-turboavia", llm).bid(brief))
    assert result.status == "submitted"
    assert [i["id"] for i in result.details["items"]] == ["ta-7401", "ta-7402"]
    assert "не стыкуется" in llm.bid_calls[-1][-1].content
