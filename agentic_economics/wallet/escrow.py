"""Эскроу, выплаты и комиссия платформы (FR-21 – FR-25).

Деньги подзадачи проходят через системный кошелёк эскроу, каждая проводка помечена task_id:
  hold    клиент → эскроу          после подтверждения плана (FR-22)
  payout  эскроу → провайдер       после успешного выполнения, за вычетом комиссии
  fee     эскроу → платформа       комиссия, по умолчанию 5 % цены (FR-23)
  refund  эскроу → клиент          подзадача не выполнена
  release эскроу → клиент          подзадача отменена до начала выполнения

Каждая функция — отдельная атомарная операция: она сама открывает транзакцию с блокировкой записи
и сама делает commit или rollback. Так два терминала одного клиента не спишут деньги дважды (NFR-4a).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Sequence

from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from agentic_economics.money import fmt
from agentic_economics.storage.models import Agent, Bid, Task, Transaction, Wallet
from agentic_economics.storage.models import Session as ChatSession
from agentic_economics.wallet.ledger import PLATFORM_OWNER, InsufficientFunds, balance, get_wallet, transfer

log = logging.getLogger(__name__)

ESCROW_OWNER = ("system", "escrow")
OUT_OF_ESCROW = ("payout", "fee", "refund", "release")


class EscrowError(RuntimeError):
    """Операция невозможна в текущем состоянии эскроу подзадачи."""


@dataclass(frozen=True)
class Settlement:
    task_id: int
    provider_id: str
    price: int
    payout: int
    fee: int


def fee_of(amount: int, rate: float) -> int:
    """Комиссия в копейках, округление до копейки по правилам арифметики."""
    return int((Decimal(amount) * Decimal(str(rate))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def held(db: Session, task_id: int) -> int:
    """Сколько денег подзадачи сейчас лежит в эскроу."""
    amount = case((Transaction.kind == "hold", Transaction.amount), else_=-Transaction.amount)
    value = db.scalar(select(func.coalesce(func.sum(amount), 0)).where(
        Transaction.task_id == task_id, Transaction.kind.in_(("hold", *OUT_OF_ESCROW))))
    return int(value)


def reserved(db: Session, client_id: str) -> int:
    """Сумма клиента, заблокированная в эскроу по всем его подзадачам."""
    task_ids = select(Task.id).join(ChatSession, ChatSession.id == Task.session_id).where(ChatSession.user_id == client_id)
    return sum(held(db, t) for t in db.scalars(task_ids))


def hold(db: Session, client_id: str, items: Sequence[tuple[Task, int]], currency: str = "VRUB") -> list[Transaction]:
    """Блокирует суммы по подзадачам — всё или ничего. InsufficientFunds, если не хватает денег (FR-25)."""
    try:
        client = _lock(db, "client", client_id, currency)
        escrow = get_wallet(db, *ESCROW_OWNER, currency=currency)
        for task, amount in items:
            _check_owner(db, task, client_id)
            if amount <= 0:
                raise ValueError(f"task-{task.number}: сумма должна быть положительной")
            if held(db, task.id):
                raise EscrowError(f"task-{task.number}: сумма уже заблокирована")
        need = sum(amount for _, amount in items)
        available = balance(db, client)
        if available < need:
            raise InsufficientFunds(
                f"Недостаточно средств: нужно {fmt(need, currency)}, доступно {fmt(available, currency)}")
        txs = [transfer(db, client, escrow, amount, "hold", task_id=task.id, note=f"Эскроу task-{task.number}")
               for task, amount in items]
        db.commit()
    except Exception:
        db.rollback()
        raise
    log.info("Клиент %s: заблокировано %s по %d подзадачам", client_id, fmt(need, currency), len(items))
    return txs


def settle(db: Session, task: Task, fee_rate: float, currency: str = "VRUB") -> Settlement:
    """Выплата провайдеру победившей ставки за вычетом комиссии платформы (FR-22, FR-23)."""
    try:
        _lock(db, *ESCROW_OWNER, currency)
        amount = held(db, task.id)
        if amount <= 0:
            raise EscrowError(f"task-{task.number}: в эскроу нет денег для выплаты")
        bid = db.get(Bid, task.winner_bid_id) if task.winner_bid_id else None
        if bid is None:
            raise EscrowError(f"task-{task.number}: нет победителя аукциона")
        provider_id = db.get(Agent, bid.agent_id).provider_id
        escrow = get_wallet(db, *ESCROW_OWNER, currency=currency)
        fee = fee_of(amount, fee_rate)
        payout = amount - fee
        if payout:
            transfer(db, escrow, get_wallet(db, "provider", provider_id, currency), payout, "payout",
                     task_id=task.id, note=f"Выплата за task-{task.number} ({bid.agent_id})")
        if fee:
            transfer(db, escrow, get_wallet(db, *PLATFORM_OWNER, currency=currency), fee, "fee",
                     task_id=task.id, note=f"Комиссия {fee_rate:.0%} за task-{task.number}")
        db.commit()
    except Exception:
        db.rollback()
        raise
    log.info("task-%s: выплата %s провайдеру %s, комиссия %s", task.number, fmt(payout), provider_id, fmt(fee))
    return Settlement(task.id, provider_id, amount, payout, fee)


def refund(db: Session, task: Task, kind: str = "refund", note: str | None = None,
           currency: str = "VRUB") -> Transaction | None:
    """Возвращает клиенту всё, что лежит в эскроу по подзадаче. Если возвращать нечего — None."""
    if kind not in ("refund", "release"):
        raise ValueError("kind: refund или release")
    try:
        _lock(db, *ESCROW_OWNER, currency)
        amount = held(db, task.id)
        tx = None
        if amount > 0:
            client_id = db.get(ChatSession, task.session_id).user_id
            reason = "Отмена" if kind == "release" else "Возврат"
            tx = transfer(db, get_wallet(db, *ESCROW_OWNER, currency=currency),
                          get_wallet(db, "client", client_id, currency), amount, kind,
                          task_id=task.id, note=note or f"{reason} по task-{task.number}")
        db.commit()
    except Exception:
        db.rollback()
        raise
    if tx:
        log.info("task-%s: %s %s клиенту", task.number, kind, fmt(amount))
    return tx


def _lock(db: Session, owner_type: str, owner_id: str, currency: str) -> Wallet:
    """Начинает денежную операцию: завершает прежнюю (возможно, устаревшую) транзакцию чтения
    и первой же командой берёт блокировку записи SQLite. Второй терминал подождёт (busy timeout)
    и прочитает уже новый баланс."""
    db.commit()
    wallet = get_wallet(db, owner_type, owner_id, currency)
    db.execute(update(Wallet).where(Wallet.id == wallet.id).values(currency=Wallet.currency))
    return wallet


def _check_owner(db: Session, task: Task, client_id: str) -> None:
    session = db.get(ChatSession, task.session_id)
    if session is None or session.user_id != client_id:
        raise EscrowError(f"task-{task.number} не принадлежит клиенту {client_id}")
