"""Выбор победителя (FR-12, FR-12a).

score = w_price · (1 − price_norm) + w_conf · confidence + w_rating · rating_norm
price_norm — цена, нормированная между минимальной и максимальной ставкой аукциона (0 — самая дешёвая);
rating_norm — рейтинг агента по шкале 1–5, приведённый к 0–1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentic_economics.config import AuctionWeights


@dataclass(frozen=True)
class Offer:
    price: int
    confidence: float
    rating: float


def rating_norm(rating: float) -> float:
    return min(max((rating - 1) / 4, 0.0), 1.0)


def score_offers(offers: Sequence[Offer], weights: AuctionWeights) -> list[float]:
    if not offers:
        return []
    low, high = min(o.price for o in offers), max(o.price for o in offers)
    scores = []
    for o in offers:
        price_norm = (o.price - low) / (high - low) if high > low else 0.0
        score = (weights.price * (1 - price_norm)
                 + weights.confidence * min(max(o.confidence, 0.0), 1.0)
                 + weights.rating * rating_norm(o.rating))
        scores.append(round(score, 6))
    return scores


def pick_winner(scores: Sequence[float]) -> int | None:
    """Индекс лучшей ставки. Ставки идут в порядке поступления, поэтому при равенстве побеждает более ранняя."""
    best: int | None = None
    for i, s in enumerate(scores):
        if best is None or s > scores[best]:
            best = i
    return best
