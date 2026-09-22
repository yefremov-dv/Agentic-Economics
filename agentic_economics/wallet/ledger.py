"""Кошельки и журнал транзакций с двойной записью (раздел 5.6 ТЗ).

Баланс не хранится, а вычисляется из журнала: сумма поступлений минус сумма списаний.
Виртуальная валюта выпускается с системного кошелька эмиссии, поэтому сумма всех
балансов, включая эмиссию, всегда равна нулю.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentic_economics.storage.models import Transaction, Wallet

SYSTEM_OWNER = ("system", "emission")
PLATFORM_OWNER = ("platform", "platform")


class InsufficientFunds(RuntimeError):
    pass


def get_wallet(db: Session, owner_type: str, owner_id: str, currency: str = "VRUB") -> Wallet:
    wallet = db.scalar(select(Wallet).where(Wallet.owner_type == owner_type, Wallet.owner_id == owner_id))
    if wallet is None:
        wallet = Wallet(owner_type=owner_type, owner_id=owner_id, currency=currency)
        db.add(wallet)
        db.flush()
    return wallet


def find_wallet(db: Session, owner_type: str, owner_id: str) -> Wallet | None:
    return db.scalar(select(Wallet).where(Wallet.owner_type == owner_type, Wallet.owner_id == owner_id))


def balance(db: Session, wallet: Wallet) -> int:
    credit = db.scalar(select(func.coalesce(func.sum(Transaction.amount), 0)).where(Transaction.credit_wallet_id == wallet.id))
    debit = db.scalar(select(func.coalesce(func.sum(Transaction.amount), 0)).where(Transaction.debit_wallet_id == wallet.id))
    return int(credit) - int(debit)


def transfer(
    db: Session,
    source: Wallet,
    target: Wallet,
    amount: int,
    kind: str,
    *,
    task_id: int | None = None,
    note: str | None = None,
    allow_negative: bool = False,
) -> Transaction:
    """Перевод в копейках. Вызывающий код отвечает за commit, чтобы серия переводов была атомарной."""
    if amount <= 0:
        raise ValueError("Сумма перевода должна быть положительной")
    if source.currency != target.currency:
        raise ValueError("Валюты кошельков не совпадают")
    if not allow_negative and balance(db, source) < amount:
        raise InsufficientFunds(f"Недостаточно средств на кошельке {source.owner_type}:{source.owner_id}")
    tx = Transaction(
        debit_wallet_id=source.id, credit_wallet_id=target.id, amount=amount, kind=kind, task_id=task_id, note=note
    )
    db.add(tx)
    db.flush()
    return tx


def topup(db: Session, wallet: Wallet, amount: int, note: str | None = None) -> Transaction:
    emission = get_wallet(db, *SYSTEM_OWNER, currency=wallet.currency)
    return transfer(db, emission, wallet, amount, "topup", note=note, allow_negative=True)
