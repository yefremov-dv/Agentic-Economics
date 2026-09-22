"""Проверка и нормализация плана от LLM: категории, зависимости, нумерация."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentic_economics.orchestrator.schemas import Decomposition, SubtaskDraft

log = logging.getLogger(__name__)


@dataclass
class PlannedTask:
    number: int
    title: str
    category: str
    description: str
    criteria: list[str]
    depends_on: list[int]
    recommended: bool
    budget_hint_rub: float | None


@dataclass
class NormalizedPlan:
    tasks: list[PlannedTask]
    warnings: list[str] = field(default_factory=list)


def normalize(plan: Decomposition, categories: set[str]) -> NormalizedPlan:
    """Отбрасывает подзадачи без исполнителей, битые и циклические зависимости,
    упорядочивает топологически и нумерует заново с 1, так что зависимости всегда ссылаются назад."""
    warnings: list[str] = []
    drafts: dict[int, SubtaskDraft] = {}
    for i, draft in enumerate(plan.tasks):
        key = draft.number if draft.number not in drafts else 10_000 + i
        if draft.category not in categories:
            warnings.append(f"«{draft.title}»: нет активных агентов категории {draft.category}, подзадача пропущена")
            continue
        if not draft.title.strip():
            continue
        drafts[key] = draft

    deps: dict[int, list[int]] = {}
    for key, draft in drafts.items():
        deps[key] = []
        for d in dict.fromkeys(draft.depends_on):
            if d == key or d not in drafts:
                log.debug("Подзадача %s: отброшена зависимость %s", key, d)
                continue
            deps[key].append(d)

    order: list[int] = []
    state: dict[int, int] = {}  # 1 — в обработке, 2 — готово

    def visit(key: int) -> None:
        state[key] = 1
        for d in list(deps[key]):
            if state.get(d) == 1:  # цикл — разрываем ребро
                deps[key].remove(d)
                warnings.append(f"Циклическая зависимость «{drafts[key].title}» → «{drafts[d].title}» убрана")
            elif d not in state:
                visit(d)
        state[key] = 2
        order.append(key)

    for key in drafts:  # исходный порядок LLM сохраняется, где его не меняют зависимости
        if key not in state:
            visit(key)

    new_number = {key: i for i, key in enumerate(order, start=1)}
    tasks = []
    for key in order:
        d = drafts[key]
        tasks.append(PlannedTask(
            number=new_number[key],
            title=d.title.strip(),
            category=d.category,
            description=d.description.strip(),
            criteria=[c.strip() for c in d.criteria if c.strip()],
            depends_on=sorted(new_number[x] for x in deps[key]),
            recommended=d.recommended,
            budget_hint_rub=d.budget_hint_rub if d.budget_hint_rub and d.budget_hint_rub > 0 else None,
        ))
    return NormalizedPlan(tasks=tasks, warnings=warnings)
