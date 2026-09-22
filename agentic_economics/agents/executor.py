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
   Количество должно покрывать подзадачу целиком: все ночи проживания, всех путешественников, оба направления,
   если нужна дорога туда и обратно. Ставка на часть подзадачи — ошибка.
   Бери по одной позиции на каждую нужную услугу, без лишнего. Если у позиции есть sizes_in_stock, укажи размер —
   он должен быть в наличии, иначе выбери другую позицию или откажись.
Цену не считай и не пиши — она рассчитывается автоматически по каталогу.
Если в каталоге нет подходящего (нет направления, отеля, нужного размера) — откажись: participate=false.
Обязательно заполни eta (срок выполнения) и confidence — насколько предложение соответствует подзадаче и критериям, от 0 до 1."""


# Уверенность ставки, если модель её не указала: ноль несправедливо топил бы ставку
NEUTRAL_CONFIDENCE = 0.5


class SearchArgs(BaseModel):
    query: str = Field(description="Ключевые слова для поиска по каталогу")


class BidLine(BaseModel):
    """Позиция каталога в ставке."""

    item_id: str = Field(description="id позиции из результатов поиска")
    quantity: int = Field(description="Количество в единицах позиции (поле unit)")
    size: str | None = Field(None, description="Размер, если у позиции есть sizes_in_stock, иначе null")


class BidDraft(BaseModel):
    """Ставка агента на подзадачу."""

    participate: bool = Field(description="false — отказаться от участия, если подходящих позиций нет")
    items: list[BidLine] = Field(default_factory=list, description="Выбранные позиции каталога и количество")
    eta: str | None = Field(None, description="Срок выполнения, например «подтверждение брони за 2 часа»")
    confidence: float | None = Field(None, description="Уверенность от 0 до 1, что предложение соответствует подзадаче")
    offer: str | None = Field(None, description="Что конкретно предлагается, 1–3 предложения, без цены")


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

        structured = self.llm.with_structured_output(BidDraft)
        draft = _as_draft(await structured.ainvoke(messages))
        result = self._price(draft, queries)
        if result.status == "declined" and result.details.get("rejected"):
            # Модель хотела участвовать, но выбрала неподходящее — одна попытка исправиться
            messages.append(HumanMessage("Ставка не принята: " + "; ".join(result.details["rejected"])
                                         + ". Выбери другие позиции из результатов поиска или откажись."))
            draft = _as_draft(await structured.ainvoke(messages))
            result = self._price(draft, queries)
        log.info("Агент %s, %s: %s %s за %.1f с", self.id, task.key, result.status, result.price,
                 time.monotonic() - start)
        return result

    def _price(self, draft: BidDraft | None, queries: list[str]) -> BidResult:
        if draft is None:
            return BidResult(self.id, "error", offer="модель не вернула ставку")
        if not draft.participate:
            return BidResult(self.id, "declined", offer=(draft.offer or "").strip() or "нет подходящих позиций")
        lines, rejected = [], []
        for line in draft.items:
            item = self.by_id.get(line.item_id)
            problem = _line_problem(item, line)
            if problem:
                rejected.append(problem)
                log.debug("Агент %s: отброшено — %s", self.id, problem)
                continue
            entry = {"id": item["id"], "title": item["title"], "quantity": line.quantity,
                     "unit": item.get("unit", ""), "unit_price": self.unit_price(item)}
            if "sizes_in_stock" in item:
                entry["size"] = line.size
            lines.append(entry)
        route = _route_problem([self.by_id[line["id"]] for line in lines])
        if route:
            rejected.append(route)
            lines = []
        if not lines:
            reason = "; ".join(rejected) or "не выбрано ни одной позиции из каталога"
            return BidResult(self.id, "declined", offer=f"ставка отклонена: {reason}"[:500],
                             details={"rejected": rejected, "queries": queries})
        total = max(sum(line["unit_price"] * line["quantity"] for line in lines), self.min_price)
        offer = (draft.offer or "").strip() or "; ".join(
            f"{line['title']}{', размер ' + line['size'] if line.get('size') else ''} × {line['quantity']}"
            for line in lines)
        if draft.confidence is None:
            log.info("Агент %s не указал уверенность — принимается нейтральная %.1f", self.id, NEUTRAL_CONFIDENCE)
        confidence = NEUTRAL_CONFIDENCE if draft.confidence is None else min(max(draft.confidence, 0.0), 1.0)
        return BidResult(
            self.id, "submitted", price=to_minor(total), eta=(draft.eta or "").strip()[:100],
            confidence=confidence, offer=offer,
            details={"items": lines, "queries": queries, **({"rejected": rejected} if rejected else {})},
        )


def _as_draft(value) -> BidDraft | None:
    return BidDraft.model_validate(value) if isinstance(value, dict) else value


def _line_problem(item: dict | None, line: BidLine) -> str | None:
    """Почему позицию ставки нельзя принять, или None, если можно."""
    if item is None:
        return f"позиции {line.item_id} нет в каталоге"
    if line.quantity < 1:
        return f"{item['title']}: количество {line.quantity}"
    if not _size_ok(item, line.size):
        return f"{item['title']}: размера {line.size or '(не указан)'} нет в наличии"
    return None


def _route_problem(items: list[dict]) -> str | None:
    """Перелёты в ставке должны стыковаться: следующий сегмент вылетает оттуда, куда прилетел предыдущий."""
    legs = [i for i in items if "from_city" in i and "to_city" in i]
    for prev, nxt in zip(legs, legs[1:]):
        if prev["to_city"] != nxt["from_city"]:
            return (f"маршрут не стыкуется: {prev['title']} прилетает в {prev['to_city']}, "
                    f"а {nxt['title']} вылетает из {nxt['from_city']}")
    return None


def _size_ok(item: dict, size: str | None) -> bool:
    """Товар с размерной сеткой принимается, только если выбранный размер есть в наличии."""
    if "sizes_in_stock" not in item or not item["sizes_in_stock"]:
        return True
    return size is not None and size.strip() in {str(x) for x in item["sizes_in_stock"]}
