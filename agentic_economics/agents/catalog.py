"""Моковые каталоги data/mock/*.json (раздел 6.4 ТЗ) и поиск по ним."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

STOPWORDS = {"и", "в", "во", "на", "с", "со", "до", "из", "по", "для", "к", "от", "о", "а", "или", "the"}


@lru_cache(maxsize=32)
def _load(path: str, mtime: float) -> tuple[dict, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    return tuple(items)


def load_items(path: Path) -> list[dict]:
    return list(_load(str(path), path.stat().st_mtime))


def for_provider(items: list[dict], provider: str) -> list[dict]:
    """Позиции, которые продаёт провайдер. Без поля sellers позиция доступна всем."""
    return [i for i in items if not i.get("sellers") or provider in i["sellers"]]


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def _stems(query: str) -> list[str]:
    # Грубый стемминг: первые 5 букв слова, чтобы «Анталью», «Анталии» и «Анталья» совпадали
    stems = []
    for word in re.findall(r"\w+", _norm(query)):
        if word in STOPWORDS or (len(word) < 2 and not word.isdigit()):
            continue
        stems.append(word[:5] if len(word) > 5 else word)
    return list(dict.fromkeys(stems))


def _text(item: dict) -> str:
    parts = []
    for key, value in item.items():
        if key in ("sellers", "price"):
            continue
        parts.append(" ".join(map(str, value)) if isinstance(value, list) else str(value))
    return _norm(" ".join(parts))


def search(items: list[dict], query: str, limit: int = 8) -> list[dict]:
    """Поиск по словам запроса: чем больше слов совпало, тем выше позиция. Пустой запрос — весь каталог."""
    stems = _stems(query)
    if not stems:
        return items[:limit]
    scored = []
    for n, item in enumerate(items):
        text = _text(item)
        hits = sum(1 for s in stems if (re.search(rf"\b{re.escape(s)}\b", text) if s.isdigit() else s in text))
        if hits:
            scored.append((-hits, n, item))
    return [item for _, _, item in sorted(scored)[:limit]]
