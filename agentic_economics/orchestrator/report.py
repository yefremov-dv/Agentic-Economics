"""Итоговый отчёт tasks/<session_id>/result.md (FR-31). Строится по базе, поэтому его можно пересоздать."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_economics.config import Settings
from agentic_economics.money import fmt
from agentic_economics.orchestrator.taskfile import task_key
from agentic_economics.storage.models import Bid, Task, Transaction, User
from agentic_economics.storage.models import Session as ChatSession
from agentic_economics.wallet.ledger import balance, find_wallet

STATUS = {"done": "выполнена", "failed": "не выполнена", "cancelled": "отменена", "approved": "подтверждена",
          "in_progress": "выполняется", "awaiting_approval": "ждёт подтверждения", "new": "новая", "bidding": "аукцион"}


@dataclass
class Totals:
    paid: int = 0       # списано с клиента окончательно: hold − refund − release
    fee: int = 0        # из них комиссия платформы
    returned: int = 0   # возвращено клиенту


def totals(db: Session, tasks: list[Task]) -> Totals:
    t = Totals()
    ids = [task.id for task in tasks]
    for tx in db.scalars(select(Transaction).where(Transaction.task_id.in_(ids))):
        if tx.kind == "hold":
            t.paid += tx.amount
        elif tx.kind in ("refund", "release"):
            t.paid -= tx.amount
            t.returned += tx.amount
        elif tx.kind == "fee":
            t.fee += tx.amount
    return t


def render(db: Session, settings: Settings, session: ChatSession) -> str:
    tasks = list(db.scalars(select(Task).where(Task.session_id == session.id).order_by(Task.number)))
    client = db.get(User, session.user_id)
    cur = settings.wallet.currency
    t = totals(db, tasks)
    lines = [f"# Итоги сессии {session.id}", "",
             f"Клиент: {client.name if client else session.user_id}  ", f"Статус сессии: {session.status}", "",
             "| # | Подзадача | Исполнитель | Статус | Цена | Результат |", "|---|---|---|---|---|---|"]
    for task in tasks:
        bid = db.get(Bid, task.winner_bid_id) if task.winner_bid_id else None
        result = task.result_json or {}
        outcome = result.get("summary") or result.get("note") or ""
        if task.recommended and not task.enabled:
            outcome = outcome or "рекомендованная, не включена"
        price = fmt(bid.price, cur) if bid and task.status in ("done", "approved", "in_progress") else "—"
        lines.append(f"| {task_key(task.number)} | {_cell(task.title)} | {bid.agent_id if bid else '—'} | "
                     f"{STATUS.get(task.status, task.status)} | {price} | {_cell(outcome)} |")
    lines += ["", "## Деньги", "",
              f"- Оплачено исполнителям: **{fmt(t.paid, cur)}**, в том числе комиссия платформы {fmt(t.fee, cur)}"]
    if t.returned:
        lines.append(f"- Возвращено из эскроу: {fmt(t.returned, cur)}")
    if session.budget:
        diff = session.budget - t.paid
        lines.append(f"- Бюджет: {fmt(session.budget, cur)}, "
                     + (f"остаток {fmt(diff, cur)}" if diff >= 0 else f"превышение {fmt(-diff, cur)}"))
    wallet = find_wallet(db, "client", session.user_id)
    if wallet:
        lines.append(f"- Баланс кошелька после сессии: {fmt(balance(db, wallet), cur)}")

    details = [task for task in tasks if (task.result_json or {}).get("details")]
    if details:
        lines += ["", "## Детали"]
        for task in details:
            lines += ["", f"### {task_key(task.number)}. {task.title}", "",
                      *(f"- {d}" for d in task.result_json["details"])]
    return "\n".join(lines) + "\n"


def write(db: Session, settings: Settings, session: ChatSession) -> Path:
    path = settings.path("tasks_dir") / session.id / "result.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(db, settings, session), encoding="utf-8")
    return path


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
