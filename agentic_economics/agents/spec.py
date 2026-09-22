"""Описание агента-исполнителя в YAML (раздел 6.2 ТЗ)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Category = Literal["flights", "hotels", "transfers", "retail", "insurance"]
CATEGORIES: tuple[str, ...] = Category.__args__

ID_PATTERN = r"^[a-z0-9][a-z0-9-]{1,63}$"


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_markup: float = Field(0.0, ge=0, le=1, description="Наценка к цене из моковых данных")
    min_price: float = Field(0, ge=0)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    name: str
    provider: str = Field(pattern=ID_PATTERN)
    category: Category
    description: str
    system_prompt: str
    pricing: Pricing = Pricing()
    mock_data: str | None = None
    rating: float = Field(4.0, ge=1, le=5, description="Стартовый рейтинг до первых оценок клиентов")
    active: bool = True


def load_spec(path: Path) -> AgentSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AgentSpec.model_validate(data)


def set_active(path: Path, active: bool) -> AgentSpec:
    """Меняет только строку `active:` в файле, сохраняя остальное форматирование и комментарии."""
    text = path.read_text(encoding="utf-8")
    line = f"active: {'true' if active else 'false'}"
    new_text, n = re.subn(r"(?m)^active:.*$", line, text)
    if n == 0:
        new_text = text.rstrip("\n") + f"\n{line}\n"
    spec = AgentSpec.model_validate(yaml.safe_load(new_text))
    path.write_text(new_text, encoding="utf-8")
    return spec
