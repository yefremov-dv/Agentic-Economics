"""Схемы структурированного ответа LLM.

У GigaChat function calling требует описания у каждой схемы и поля — это docstring и Field(description=...).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_economics.agents.spec import Category


class Analysis(BaseModel):
    """Анализ диалога с клиентом: достаточно ли данных, чтобы разбить задачу на подзадачи."""

    ready: bool = Field(
        description="true — данных достаточно для декомпозиции; false — нужно задать клиенту уточняющие вопросы"
    )
    questions: list[str] = Field(
        default_factory=list,
        description="Уточняющие вопросы клиенту, только если ready=false. Не более 5, самые важные",
    )
    budget_rub: float | None = Field(
        None, description="Общий бюджет клиента в рублях, если он назван в диалоге, иначе null"
    )
    summary: str = Field(
        "", description="Краткое изложение задачи клиента со всеми известными деталями: даты, люди, места, пожелания"
    )


class SubtaskDraft(BaseModel):
    """Одна простая подзадача, которую выполнит один агент одной категории."""

    number: int = Field(description="Порядковый номер подзадачи, начиная с 1")
    title: str = Field(description="Короткое название подзадачи")
    category: Category = Field(description="Категория агента-исполнителя")
    description: str = Field(description="Что именно нужно сделать: даты, количество людей, места, параметры")
    criteria: list[str] = Field(
        default_factory=list, description="Критерии выполнения: по каким признакам проверить результат"
    )
    depends_on: list[int] = Field(
        default_factory=list, description="Номера подзадач, результат которых нужен для этой (например, трансфер зависит от рейса и отеля)"
    )
    recommended: bool = Field(
        False, description="true — клиент об этом не просил, но это логически нужно (например, страховка)"
    )
    budget_hint_rub: float | None = Field(
        None, description="Ориентировочная стоимость подзадачи в рублях в пределах общего бюджета, если он известен"
    )


class Decomposition(BaseModel):
    """План: задача клиента, разбитая на максимальное число простых подзадач."""

    tasks: list[SubtaskDraft] = Field(description="Подзадачи в порядке выполнения")
