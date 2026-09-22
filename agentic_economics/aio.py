"""Общий event loop для асинхронных шагов (сбор ставок, выполнение).

Асинхронный клиент GigaChat привязывается к циклу, в котором его впервые использовали. Если каждый шаг
запускать через asyncio.run (каждый раз новый цикл), клиент следующего шага падает с «Event loop is closed».
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None


def run(coro: Awaitable[T]) -> T:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)
