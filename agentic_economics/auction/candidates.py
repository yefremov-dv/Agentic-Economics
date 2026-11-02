"""Отбор кандидатов (FR-10): активные агенты категории, при избытке — ранжирование LLM."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_economics.storage.models import Agent

log = logging.getLogger(__name__)

RANK = """\
Подзадача: {task}

Агенты-исполнители категории {category}:
{agents}

Упорядочи агентов по тому, насколько их описание подходит к подзадаче, от лучшего к худшему."""


class Ranking(BaseModel):
    """Агенты, упорядоченные по соответствию подзадаче."""

    agent_ids: list[str] = Field(description="id агентов от самого подходящего к наименее подходящему")


def select_candidates(db: Session, category: str, limit: int, task_text: str = "",
                      llm: BaseChatModel | None = None) -> list[Agent]:
    agents = list(db.scalars(select(Agent).where(Agent.category == category, Agent.active.is_(True))
                             .order_by(Agent.rating.desc(), Agent.id)))
    if len(agents) <= limit or llm is None:
        return agents[:limit]
    listing = "\n".join(f"- {a.id}: {a.description}" for a in agents)
    try:
        ranking = llm.with_structured_output(Ranking).invoke(
            RANK.format(task=task_text, category=category, agents=listing))
        order = [i for i in dict.fromkeys(ranking.agent_ids) if i in {a.id for a in agents}]
    except Exception as exc:  # ранжирование — улучшение, а не условие аукциона
        log.warning("Ранжирование кандидатов не удалось: %s", exc)
        order = []
    by_id = {a.id: a for a in agents}
    ranked = [by_id[i] for i in order] + [a for a in agents if a.id not in order]
    return ranked[:limit]
