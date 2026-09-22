"""Денежные суммы хранятся в копейках (целые числа), чтобы избежать ошибок округления."""

from decimal import ROUND_HALF_UP, Decimal


def to_minor(amount: float | int | str | Decimal) -> int:
    return int((Decimal(str(amount)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fmt(minor: int, currency: str = "VRUB") -> str:
    value = Decimal(minor) / 100
    text = f"{value:,.2f}".replace(",", " ")
    return f"{text} {currency}"
