import pytest
from sqlalchemy import select

from agentic_economics.storage.models import Wallet
from agentic_economics.wallet.ledger import InsufficientFunds, balance, find_wallet, get_wallet, topup, transfer


def total(db) -> int:
    return sum(balance(db, w) for w in db.scalars(select(Wallet)))


def test_seed_balances(db):
    assert balance(db, find_wallet(db, "client", "ivanov")) == 1_500_000_00
    assert balance(db, find_wallet(db, "provider", "sky-lines")) == 0
    assert balance(db, find_wallet(db, "platform", "platform")) == 0


def test_money_is_conserved(db):
    client = find_wallet(db, "client", "ivanov")
    provider = find_wallet(db, "provider", "sky-lines")
    transfer(db, client, provider, 12_345_67, "payout")
    topup(db, provider, 100)
    db.commit()
    assert total(db) == 0  # эмиссия уравновешивает все балансы
    assert balance(db, client) == 1_500_000_00 - 12_345_67


def test_insufficient_funds(db):
    provider = find_wallet(db, "provider", "sky-lines")
    client = find_wallet(db, "client", "ivanov")
    with pytest.raises(InsufficientFunds):
        transfer(db, provider, client, 1, "refund")


@pytest.mark.parametrize("amount", [0, -1])
def test_non_positive_amount(db, amount):
    client = find_wallet(db, "client", "ivanov")
    provider = find_wallet(db, "provider", "sky-lines")
    with pytest.raises(ValueError):
        transfer(db, client, provider, amount, "payout")


def test_get_wallet_is_idempotent(db):
    first = get_wallet(db, "client", "new-client")
    assert get_wallet(db, "client", "new-client").id == first.id
