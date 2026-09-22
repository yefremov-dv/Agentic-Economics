"""Файлы подзадач tasks/<session_id>/task-N.md (раздел 6.1 ТЗ)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml

from agentic_economics.storage.models import Task


def task_key(number: int) -> str:
    return f"task-{number}"


def render(task: Task) -> str:
    meta: dict = {
        "id": task_key(task.number),
        "session": task.session_id,
        "category": task.category,
        "depends_on": [task_key(n) for n in task.depends_on],
        "status": task.status,
        "recommended": task.recommended,
        "enabled": task.enabled,
    }
    if task.budget_hint is not None:
        meta["budget_hint"] = _rub(task.budget_hint)
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, default_flow_style=None).strip()
    lines = ["---", front, "---", f"# {task.title}", ""]
    if task.description:
        lines += [task.description, ""]
    if task.criteria:
        lines += ["## Критерии выполнения", *(f"- {c}" for c in task.criteria), ""]
    return "\n".join(lines)


def write(task: Task, tasks_dir: Path) -> Path:
    path = tasks_dir / task.session_id / f"{task_key(task.number)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(task), encoding="utf-8")
    return path


def read_meta(path: Path) -> dict:
    """Front matter файла подзадачи."""
    text = path.read_text(encoding="utf-8")
    _, front, _ = text.split("---\n", 2)
    return yaml.safe_load(front)


def _rub(minor: int) -> int | float:
    value = Decimal(minor) / 100
    return int(value) if value == value.to_integral_value() else float(value)
