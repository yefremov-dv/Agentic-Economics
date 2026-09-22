"""Оценки агентов клиентами (FR-32 – FR-34)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentic_economics.storage.models import Agent, Bid, Rating, Task
from agentic_economics.storage.models import Session as ChatSession


class RatingError(ValueError):
    pass


def rate(db: Session, task: Task, user_id: str, score: int, comment: str | None = None) -> Agent:
    """Оценка исполнителя выполненной подзадачи. Рейтинг агента — среднее всех оценок, округлённое до 0,1."""
    if not 1 <= score <= 5:
        raise RatingError("оценка — целое число от 1 до 5")
    session = db.get(ChatSession, task.session_id)
    if session is None or session.user_id != user_id:
        raise RatingError("подзадача не найдена")
    if task.status != "done" or task.winner_bid_id is None:
        raise RatingError(f"task-{task.number} не выполнена — оценивать нечего")
    if db.scalar(select(Rating.id).where(Rating.task_id == task.id)):
        raise RatingError(f"task-{task.number} уже оценена")
    agent = db.get(Agent, db.get(Bid, task.winner_bid_id).agent_id)
    db.add(Rating(task_id=task.id, agent_id=agent.id, user_id=user_id, score=score, comment=comment or None))
    db.flush()
    count, total = db.execute(select(func.count(Rating.id), func.sum(Rating.score))
                              .where(Rating.agent_id == agent.id)).one()
    agent.ratings_count = count
    agent.rating = float((Decimal(total) / count).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    db.commit()
    return agent


def is_rated(db: Session, task: Task) -> bool:
    return db.scalar(select(Rating.id).where(Rating.task_id == task.id)) is not None
