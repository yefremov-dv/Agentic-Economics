"""Фейковая LLM для тестов: отдаёт заранее заданные структурированные ответы по очереди."""

from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from agentic_economics.agents.executor import BidDraft, ExecutionReport
from agentic_economics.orchestrator.execution import Verification


class FakeLLM:
    """Оркестратор получает ответы из очереди responses по порядку.

    Агенты-исполнители (bind_tools + BidDraft) по умолчанию ищут по названию подзадачи и ставят на первую
    найденную позицию × 1. Поведение отдельных агентов задаётся в agents: {подстрока системного промпта: правило},
    правило — dict с ключами query, bid (dict BidDraft), delay (с), error (исключение),
    execute (dict ExecutionReport или список — по попыткам), execute_error (исключение при выполнении).
    Проверка результата оркестратором по умолчанию принимает отчёт; verify(текст проверки) -> dict меняет это.
    """

    def __init__(self, *responses, agents: dict | None = None, verify=None):
        self.responses = list(responses)
        self.agents = agents or {}
        self.verify = verify
        self.executions: list[list] = []
        self.loops: set[int] = set()  # event loop'ы, в которых вызывалась модель
        self.verifications: list[str] = []
        self.calls: list[tuple[type[BaseModel], list]] = []
        self.bid_calls: list[list] = []

    def with_structured_output(self, schema, **_):
        return _Structured(self, schema)

    def bind_tools(self, tools, **_):
        return _Tools(self, tools)

    def rule(self, messages) -> dict:
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        return next((r for key, r in self.agents.items() if key in system), {})


class _Tools:
    def __init__(self, llm: FakeLLM, tools):
        self.llm, self.tools = llm, tools

    async def ainvoke(self, messages):
        rule = self.llm.rule(messages)
        if rule.get("delay"):
            await asyncio.sleep(rule["delay"])
        if rule.get("error"):
            raise rule["error"]
        if rule.get("no_tool"):
            return AIMessage(content="Сейчас посмотрю.")
        query = rule.get("query") or messages[-1].content.splitlines()[0].split(": ", 1)[-1]
        return AIMessage(content="", tool_calls=[{"name": self.tools[0].name, "args": {"query": query},
                                                  "id": "call-1", "type": "tool_call"}])


class _Structured:
    def __init__(self, llm: FakeLLM, schema):
        self.llm, self.schema = llm, schema

    async def ainvoke(self, messages):
        self.llm.loops.add(id(asyncio.get_running_loop()))
        if self.schema is Verification:
            self.llm.verifications.append(messages)
            return Verification.model_validate(self.llm.verify(messages) if self.llm.verify else {"ok": True})
        if self.schema is ExecutionReport:
            return self._execute(messages)
        assert self.schema is BidDraft, self.schema
        self.llm.bid_calls.append(messages)
        rule = self.llm.rule(messages)
        if "bids" in rule:  # последовательность ставок: первая попытка, исправление
            attempt = sum(1 for m in messages if "Ставка не принята" in str(m.content))
            return BidDraft.model_validate(rule["bids"][attempt])
        if "bid" in rule:
            return BidDraft.model_validate(rule["bid"])
        found = []
        for m in messages:
            if isinstance(m, ToolMessage) and m.content.startswith("["):
                found = json.loads(m.content)
            elif "Результаты поиска" in str(m.content) and "[" in m.content:
                found = json.loads(m.content[m.content.index("["):m.content.rindex("]") + 1])
        if not found:
            return BidDraft(participate=False, offer="нет подходящих позиций")
        sizes = found[0].get("sizes_in_stock") or [None]
        size = str(sizes[0]) if sizes[0] is not None else None
        return BidDraft(participate=True, items=[{"item_id": found[0]["id"], "quantity": 1, "size": size}],
                        eta="1 день", confidence=0.8, offer=found[0]["title"])

    def _execute(self, messages):
        self.llm.executions.append(messages)
        rule = self.llm.rule(messages)
        if rule.get("execute_error"):
            raise rule["execute_error"]
        plan = rule.get("execute")
        if isinstance(plan, list):
            attempt = sum(1 for m in messages if "не принял прошлый отчёт" in str(m.content))
            plan = plan[min(attempt, len(plan) - 1)]
        if plan is not None:
            return ExecutionReport.model_validate(plan)
        reference = messages[0].content.split("Номер подтверждения: ")[1].split(" ")[0]
        title = messages[1].content.splitlines()[0]
        return ExecutionReport(success=True, summary=f"Выполнено: {title}. Номер подтверждения: {reference}",
                               details=[f"Номер: {reference}"])

    def invoke(self, messages):
        self.llm.calls.append((self.schema, messages))
        if not self.llm.responses:
            raise AssertionError(f"FakeLLM: неожиданный запрос {self.schema.__name__}")
        response = self.llm.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return self.schema.model_validate(response)


def analysis(ready=True, questions=(), budget=None, summary="Поездка в Турцию"):
    return {"ready": ready, "questions": list(questions), "budget_rub": budget, "summary": summary}


def task(number, title, category, depends_on=(), recommended=False, budget=None, description="Описание"):
    return {"number": number, "title": title, "category": category, "description": description,
            "criteria": ["Критерий"], "depends_on": list(depends_on), "recommended": recommended,
            "budget_hint_rub": budget}


TURKEY_PLAN = {"tasks": [
    task(1, "Трансфер аэропорт ↔ отель", "transfers", depends_on=[2, 3], budget=10000),
    task(2, "Авиабилеты Москва — Анталья и обратно", "flights", budget=150000),
    task(3, "Отель Maxx Royal Belek, 6 ночей", "hotels", budget=600000),
    task(4, "Розовые кроссовки Nike, 42", "retail", budget=20000),
    task(5, "Страховка для выезда за рубеж", "insurance", recommended=True, budget=5000),
]}
