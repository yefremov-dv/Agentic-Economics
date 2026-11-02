import threading

import pytest
from sqlalchemy import select

from agentic_economics.storage.db import make_sessionmaker
from agentic_economics.storage.models import Bid, Task, Transaction, Wallet
from agentic_economics.storage.models import Session as ChatSession
from agentic_economics.wallet.escrow import (
    ESCROW_OWNER, EscrowError, fee_of, held, hold, refund, reserved, settle,
)
from agentic_economics.wallet.ledger import PLATFORM_OWNER, InsufficientFunds, balance, find_wallet, get_wallet

RUB = 100
START = 1_500_000 * RUB


def total(db) -> int:
    return sum(balance(db, w) for w in db.scalars(select(Wallet)))


def bal(db, owner_type, owner_id) -> int:
    w = find_wallet(db, owner_type, owner_id)
    return balance(db, w) if w else 0


def make_task(db, user="ivanov", number=1, agent="hotels-sunrise", price=100_000 * RUB, session_id=None):
    session_id = session_id or f"s-{user}"
    if db.get(ChatSession, session_id) is None:
        db.add(ChatSession(id=session_id, user_id=user, status="awaiting_approval"))
    task = Task(session_id=session_id, number=number, title=f"Подзадача {number}", category="hotels",
                status="awaiting_approval")
    db.add(task)
    db.flush()
    bid = Bid(task_id=task.id, agent_id=agent, price=price, confidence=0.9, offer="", status="winner")
    db.add(bid)
    db.flush()
    task.winner_bid_id = bid.id
    db.commit()
    return task, bid


# ------------------------------------------------------------ комиссия


@pytest.mark.parametrize("amount,expected", [(100_000, 5_000), (1, 0), (10, 1), (30, 2), (99_999, 5_000)])
def test_fee_rounding(amount, expected):
    assert fee_of(amount, 0.05) == expected


# ------------------------------------------------------------ полный цикл


def test_hold_then_settle(db):
    task, bid = make_task(db, price=540_000 * RUB)
    hold(db, "ivanov", [(task, bid.price)])
    assert bal(db, "client", "ivanov") == START - 540_000 * RUB
    assert held(db, task.id) == 540_000 * RUB
    assert reserved(db, "ivanov") == 540_000 * RUB
    assert bal(db, *ESCROW_OWNER) == 540_000 * RUB

    s = settle(db, task, 0.05)
    assert (s.provider_id, s.price, s.fee, s.payout) == ("sunrise-travel", 540_000 * RUB, 27_000 * RUB, 513_000 * RUB)
    assert bal(db, "provider", "sunrise-travel") == 513_000 * RUB
    assert bal(db, *PLATFORM_OWNER) == 27_000 * RUB
    assert held(db, task.id) == 0 and reserved(db, "ivanov") == 0
    assert bal(db, "client", "ivanov") == START - 540_000 * RUB
    assert total(db) == 0  # критерий приёмки 4: деньги не создаются и не исчезают

    kinds = [t.kind for t in db.scalars(select(Transaction).where(Transaction.task_id == task.id).order_by(Transaction.id))]
    assert kinds == ["hold", "payout", "fee"]


@pytest.mark.parametrize("kind", ["refund", "release"])
def test_refund_returns_everything(db, kind):
    task, bid = make_task(db)
    hold(db, "ivanov", [(task, bid.price)])
    tx = refund(db, task, kind)
    assert tx.kind == kind and tx.amount == bid.price
    assert bal(db, "client", "ivanov") == START
    assert held(db, task.id) == 0
    assert refund(db, task, kind) is None  # повторно возвращать нечего
    assert total(db) == 0


def test_refund_kind_is_checked(db):
    task, _ = make_task(db)
    with pytest.raises(ValueError):
        refund(db, task, "payout")


def test_hold_many_is_all_or_nothing(db):
    t1, _ = make_task(db, number=1)
    t2, _ = make_task(db, number=2)
    before = db.query(Transaction).count()
    with pytest.raises(InsufficientFunds, match="нужно 1 600 000.00 VRUB, доступно 1 500 000.00 VRUB"):
        hold(db, "ivanov", [(t1, 800_000 * RUB), (t2, 800_000 * RUB)])  # FR-25
    assert db.query(Transaction).count() == before  # ни одной проводки
    hold(db, "ivanov", [(t1, 700_000 * RUB), (t2, 800_000 * RUB)])
    assert bal(db, "client", "ivanov") == 0
    assert reserved(db, "ivanov") == START


def test_double_hold_is_rejected(db):
    task, bid = make_task(db)
    hold(db, "ivanov", [(task, bid.price)])
    with pytest.raises(EscrowError, match="уже заблокирована"):
        hold(db, "ivanov", [(task, bid.price)])
    assert bal(db, "client", "ivanov") == START - bid.price


def test_settle_requires_money_and_winner(db):
    task, bid = make_task(db)
    with pytest.raises(EscrowError, match="нет денег"):
        settle(db, task, 0.05)
    hold(db, "ivanov", [(task, bid.price)])
    settle(db, task, 0.05)
    with pytest.raises(EscrowError, match="нет денег"):
        settle(db, task, 0.05)  # второй выплаты не будет

    t2, bid2 = make_task(db, number=2)
    hold(db, "ivanov", [(t2, bid2.price)])
    t2.winner_bid_id = None
    db.commit()
    with pytest.raises(EscrowError, match="нет победителя"):
        settle(db, t2, 0.05)
    assert held(db, t2.id) == bid2.price  # неудачная операция ничего не списала


def test_foreign_task_and_bad_amount(db):
    task, bid = make_task(db, user="petrova")
    with pytest.raises(EscrowError, match="не принадлежит"):
        hold(db, "ivanov", [(task, bid.price)])
    with pytest.raises(ValueError):
        hold(db, "petrova", [(task, 0)])


def test_zero_fee_rate(db):
    task, bid = make_task(db)
    hold(db, "ivanov", [(task, bid.price)])
    s = settle(db, task, 0.0)
    assert s.fee == 0 and s.payout == bid.price
    assert bal(db, *PLATFORM_OWNER) == 0


def test_full_fee_rate(db):
    task, _ = make_task(db, price=10 * RUB)
    hold(db, "ivanov", [(task, 10 * RUB)])
    s = settle(db, task, 1.0)
    assert s.payout == 0 and s.fee == 10 * RUB  # выплаты в 0 копеек не пишутся в журнал


# ------------------------------------------------------------ параллельная работа (NFR-4a)


def test_parallel_terminals_cannot_double_spend(seeded, db):
    # Два терминала одного клиента одновременно блокируют по 1 000 000 при балансе 1 500 000
    t1, _ = make_task(db, number=1, session_id="s1")
    t2, _ = make_task(db, number=2, session_id="s2")
    maker = make_sessionmaker(seeded)
    barrier = threading.Barrier(2)
    results = {}

    def terminal(task_id):
        with maker() as s:
            task = s.get(Task, task_id)
            balance(s, find_wallet(s, "client", "ivanov"))  # у терминала открыт снимок с прошлым балансом
            barrier.wait()
            try:
                hold(s, "ivanov", [(task, 1_000_000 * RUB)])
                results[task_id] = "ok"
            except InsufficientFunds:
                results[task_id] = "insufficient"

    threads = [threading.Thread(target=terminal, args=(t.id,)) for t in (t1, t2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results.values()) == ["insufficient", "ok"]
    db.expire_all()
    db.commit()
    assert bal(db, "client", "ivanov") == 500_000 * RUB
    assert total(db) == 0


def test_escrow_wallet_is_created_on_demand(db):
    assert find_wallet(db, *ESCROW_OWNER) is None
    task, bid = make_task(db)
    hold(db, "ivanov", [(task, bid.price)])
    assert get_wallet(db, *ESCROW_OWNER).owner_type == "system"
