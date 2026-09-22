"""Проведение аукционов по подзадачам: ставки собираются параллельно (NFR-3), с таймаутом (FR-13)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from agentic_economics.agents.executor import BidResult, ExecutorAgent, TaskBrief
from agentic_economics.auction.candidates import select_candidates
from agentic_economics.auction.scoring import Offer, pick_winner, score_offers
from agentic_economics.config import Settings
from agentic_economics.orchestrator import taskfile
from agentic_economics.storage.models import Agent, Bid, Task, utcnow

log = logging.getLogger(__name__)


@dataclass
class AuctionOutcome:
    task: Task
    bids: list[Bid]
    winner: Bid | None


@dataclass
class _Job:
    task: Task
    agent: ExecutorAgent
    brief: TaskBrief


def run_auctions(settings: Settings, db: Session, tasks: list[Task], agent_llm: BaseChatModel,
                 rank_llm: BaseChatModel | None = None, today: date | None = None,
                 on_progress: Callable[[str], None] | None = None) -> list[AuctionOutcome]:
    """Один раунд закрытых ставок по каждой подзадаче (FR-12a). Все аукционы идут одновременно."""
    cfg = settings.auction
    by_number = {t.number: t for t in tasks}
    jobs: list[_Job] = []
    agents: dict[str, Agent] = {}
    for task in tasks:
        brief = TaskBrief(task_id=task.id, key=taskfile.task_key(task.number), title=task.title,
                          category=task.category, description=task.description, criteria=list(task.criteria),
                          context=_context(task, by_number))
        for agent in select_candidates(db, task.category, cfg.max_candidates, brief.text(), rank_llm):
            agents[agent.id] = agent
            executor = ExecutorAgent(agent.id, agent.config_json, agent_llm, settings.root, today)
            jobs.append(_Job(task, executor, brief))
        task.status = "bidding"
    db.commit()

    if on_progress:
        on_progress(f"Аукцион: {len(jobs)} ставок по {len(tasks)} подзадачам…")
    received = asyncio.run(_collect(jobs, cfg.bid_timeout, cfg.max_parallel, on_progress))

    outcomes = []
    for task in tasks:
        results = [r for job, r, _ in received if job.task is task]
        stamps = [ts for job, _, ts in received if job.task is task]
        outcomes.append(_settle(db, settings, task, results, stamps, agents))
    db.commit()
    return outcomes


async def _collect(jobs: list[_Job], timeout: float, parallel: int,
                   on_progress: Callable[[str], None] | None) -> list[tuple[_Job, BidResult, object]]:
    semaphore = asyncio.Semaphore(parallel)
    received: list[tuple[_Job, BidResult, object]] = []  # в порядке поступления ставок

    async def one(job: _Job) -> None:
        async with semaphore:
            try:
                result = await asyncio.wait_for(job.agent.bid(job.brief), timeout)
            except asyncio.TimeoutError:
                log.warning("Агент %s не уложился в %s с (%s)", job.agent.id, timeout, job.brief.key)
                result = BidResult(job.agent.id, "timeout", offer=f"нет ответа за {timeout:g} с")
            except Exception as exc:  # NFR-2: сбой одного агента не срывает аукцион
                log.exception("Агент %s: ошибка ставки по %s", job.agent.id, job.brief.key)
                result = BidResult(job.agent.id, "error", offer=f"ошибка: {type(exc).__name__}")
        received.append((job, result, utcnow()))
        if on_progress:
            on_progress(f"Аукцион: получено {len(received)} из {len(jobs)} ставок…")

    await asyncio.gather(*(one(job) for job in jobs))
    return received


def _settle(db: Session, settings: Settings, task: Task, results: list[BidResult], stamps: list,
            agents: dict[str, Agent]) -> AuctionOutcome:
    valid = [r for r in results if r.status == "submitted"]
    scores = score_offers([Offer(r.price, r.confidence, agents[r.agent_id].rating) for r in valid],
                          settings.auction.weights)
    score_of = {id(r): s for r, s in zip(valid, scores)}
    best = pick_winner(scores)
    winner_result = valid[best] if best is not None else None

    bids = []
    for result, ts in zip(results, stamps):
        status = result.status
        if status == "submitted":
            status = "winner" if result is winner_result else "lost"
        bid = Bid(task_id=task.id, agent_id=result.agent_id, price=result.price, eta=result.eta,
                  confidence=result.confidence, offer=result.offer, score=score_of.get(id(result)),
                  status=status, details=result.details, ts=ts)
        db.add(bid)
        bids.append(bid)
    db.flush()
    winner = next((b for b in bids if b.status == "winner"), None)
    task.winner_bid_id = winner.id if winner else None
    task.status = "awaiting_approval"
    taskfile.write(task, settings.path("tasks_dir"), winner)
    log.info("%s: %d ставок, победитель %s", taskfile.task_key(task.number), len(valid),
             winner.agent_id if winner else "—")
    return AuctionOutcome(task, bids, winner)


def _context(task: Task, by_number: dict[int, Task]) -> str:
    lines = []
    for n in task.depends_on:
        dep = by_number.get(n)
        if dep:
            lines.append(f"- {taskfile.task_key(n)}: {dep.title}. {dep.description}")
    return "\n".join(lines)
