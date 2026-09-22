"""Фейковая LLM для тестов: отдаёт заранее заданные структурированные ответы по очереди."""

from __future__ import annotations

from pydantic import BaseModel


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[type[BaseModel], list]] = []

    def with_structured_output(self, schema, **_):
        return _Structured(self, schema)


class _Structured:
    def __init__(self, llm: FakeLLM, schema):
        self.llm, self.schema = llm, schema

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
