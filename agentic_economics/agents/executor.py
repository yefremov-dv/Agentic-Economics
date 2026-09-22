"""Агент-исполнитель: один класс, параметризованный YAML-описанием (раздел 7 ТЗ).

Ставка (FR-11): агент ищет позиции в своём каталоге инструментом LangChain, выбирает позиции и количество,
а цену считает код — по каталогу и ценовой политике агента, поэтому модель не может её выдумать.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic_economics.agents import catalog
from agentic_economics.money import to_minor

log = logging.getLogger(__name__)

TOOLS = {
    "flights": ("search_flights", "Поиск рейсов в каталоге компании: города, аэропорты, авиакомпания, класс."),
    "hotels": ("search_hotels", "Поиск отелей в каталоге компании: название отеля, курорт, страна, тип номера."),
    "transfers": ("search_transfers", "Поиск трансферов в каталоге компании: аэропорт, курорт, тип трансфера."),
    "retail": ("search_products", "Поиск товаров в каталоге магазина: бренд, модель, цвет, размер."),
    "insurance": ("search_insurance", "Поиск страховых программ компании: страна, покрытие, опции."),
}

SYSTEM = """\
{system_prompt}
Сегодня {today}. Ты участвуешь в закрытом аукционе за подзадачу клиента: одна ставка, ставок конкурентов ты не видишь.
Порядок работы:
1. Найди подходящие позиции в каталоге своей компании инструментом {tool}. Запрос — ключевые слова: города, названия, цвет, размер.
2. Сделай ставку: выбери позиции (item_id из результатов поиска) и количество каждой в единицах из поля unit
   (например, 6 ночей; 1 билет туда и 1 обратно на каждого пассажира; 2 поездки трансфера; дни × путешественники для страховки).
Цену не считай и не пиши — она рассчитывается автоматически по каталогу.
Если в каталоге нет подходящего (нет направления, отеля, нужного размера) — откажись: participate=false.
confidence — насколько твоё предложение соответствует подзадаче и критериям, от 0 до 1."""


class SearchArgs(BaseModel):
    query: str = Field(description="Ключевые слова для поиска по каталогу")


class BidLine(BaseModel):
    """Позиция каталога в ставке."""

    item_id: str = Field(description="id позиции из результатов поиска")
    quantity: int = Field(description="Количество в единицах позиции (поле unit)")


class BidDraft(BaseModel):
    """Ставка агента на подзадачу."""

    participate: bool = Field(description="false — отказаться от участия, если подходящих позиций нет")
    items: list[BidLine] = Field(default_factory=list, description="Выбранные позиции каталога и количество")
    eta: str = Field("", description="Срок выполнения, например «подтверждение брони за 2 часа»")
    confidence: float = Field(0.0, description="Уверенность от 0 до 1, что предложение соответствует подзадаче")
    offer: str = Field("", description="Что конкретно предлагается, 1–3 предложения, без цены")


@dataclass
class TaskBrief:
    """Подзадача в том виде, в каком её видит агент (без объектов БД — ставки собираются параллельно)."""

    task_id: int
    key: str
    title: str
    category: str
    description: str
    criteria: list[str]
    context: str = ""

    def text(self) -> str:
        lines = [f"Подзадача {self.key}: {self.title}", self.description]
        if self.criteria:
            lines += ["Критерии выполнения:", *(f"- {c}" for c in self.criteria)]
        if self.context:
            lines += ["Связанные подзадачи (выполняют другие исполнители):", self.context]
        return "\n".join(line for line in lines if line)


@dataclass
class BidResult:
    agent_id: str
    status: str  # submitted | declined | error | timeout
    price: int = 0  # в копейках
    eta: str = ""
    confidence: float = 0.0
    offer: str = ""
    details: dict = field(default_factory=dict)


class ExecutorAgent:
    def __init__(self, agent_id: str, config: dict, llm: BaseChatModel, root: Path, today: date | None = None):
        self.id = agent_id
        self.config = config
        self.llm = llm
        self.today = today or date.today()
        self.markup = config.get("pricing", {}).get("base_markup", 0.0)
        self.min_price = config.get("pricing", {}).get("min_price", 0)
        mock = config.get("mock_data")
        path = Path(mock) if mock else None
        if path is not None and not path.is_absolute():
            path = root / path
        self.items = catalog.for_provider(catalog.load_items(path), config["provider"]) if path and path.exists() else []
        self.by_id = {i["id"]: i for i in self.items}
        self.tool_name, self.tool_description = TOOLS[config["category"]]

    # ------------------------------------------------------------ каталог

    def unit_price(self, item: dict) -> float:
        return round(item["price"] * (1 + self.markup))

    def search(self, query: str) -> list[dict]:
        found = catalog.search(self.items, query)
        return [{k: v for k, v in i.items() if k != "sellers"} | {"price": self.unit_price(i)} for i in found]

    def tool(self) -> StructuredTool:
        def run(query: str) -> str:
            found = self.search(query)
            return json.dumps(found, ensure_ascii=False) if found else "Ничего не найдено. Попробуйте другие слова."

        return StructuredTool.from_function(func=run, name=self.tool_name, description=self.tool_description,
                                            args_schema=SearchArgs)

    # ------------------------------------------------------------ ставка

    async def bid(self, task: TaskBrief) -> BidResult:
        start = time.monotonic()
        tool = self.tool()
        system = SYSTEM.format(system_prompt=self.config["system_prompt"].strip(), today=f"{self.today:%d.%m.%Y}",
                               tool=self.tool_name)
        messages: list = [SystemMessage(system), HumanMessage(task.text())]
        queries: list[str] = []

        ai = await self.llm.bind_tools([tool]).ainvoke(messages)
        calls = [c for c in getattr(ai, "tool_calls", None) or [] if c["name"] == self.tool_name]
        if calls:
            messages.append(ai)
            for call in calls:
                query = str(call["args"].get("query", ""))
                queries.append(query)
                messages.append(ToolMessage(tool.invoke({"query": query}), tool_call_id=call["id"]))
        else:
            # Модель не вызвала инструмент — ищем сами по названию подзадачи
            queries.append(task.title)
            if isinstance(ai, AIMessage) and ai.content:
                messages.append(ai)
            messages.append(HumanMessage(f"Результаты поиска по каталогу ({self.tool_name}):\n{tool.invoke({'query': task.title})}"
                                         "\nТеперь сделай ставку."))

        draft = await self.llm.with_structured_output(BidDraft).ainvoke(messages)
        if isinstance(draft, dict):
            draft = BidDraft.model_validate(draft)
        result = self._price(draft, queries)
        log.info("Агент %s, %s: %s %s за %.1f с", self.id, task.key, result.status, result.price,
                 time.monotonic() - start)
        return result

    def _price(self, draft: BidDraft | None, queries: list[str]) -> BidResult:
        if draft is None:
            return BidResult(self.id, "error", offer="модель не вернула ставку")
        if not draft.participate:
            return BidResult(self.id, "declined", offer=draft.offer.strip() or "нет подходящих позиций")
        lines = []
        for line in draft.items:
            item = self.by_id.get(line.item_id)
            if item is None or line.quantity < 1:
                log.debug("Агент %s: отброшена позиция %s × %s", self.id, line.item_id, line.quantity)
                continue
            lines.append({"id": item["id"], "title": item["title"], "quantity": line.quantity,
                          "unit": item.get("unit", ""), "unit_price": self.unit_price(item)})
        if not lines:
            return BidResult(self.id, "declined", offer="не выбрано ни одной позиции из каталога")
        total = max(sum(line["unit_price"] * line["quantity"] for line in lines), self.min_price)
        offer = draft.offer.strip() or "; ".join(f"{line['title']} × {line['quantity']}" for line in lines)
        return BidResult(
            self.id, "submitted", price=to_minor(total), eta=draft.eta.strip()[:100],
            confidence=min(max(draft.confidence, 0.0), 1.0), offer=offer,
            details={"items": lines, "queries": queries},
        )
