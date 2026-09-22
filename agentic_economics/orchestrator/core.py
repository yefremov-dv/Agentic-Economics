"""Диалог с клиентом: уточнение → декомпозиция → task-N.md (FR-5 – FR-9a)."""

from __future__ import annotations

import logging
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from agentic_economics.config import Settings
from agentic_economics.money import fmt, to_minor
from agentic_economics.orchestrator import prompts, taskfile
from agentic_economics.orchestrator.planner import normalize
from agentic_economics.orchestrator.schemas import Analysis, Decomposition
from agentic_economics.storage.models import Agent, Bid, Message, Task
from agentic_economics.storage.models import Session as ChatSession

log = logging.getLogger(__name__)

S = TypeVar("S", bound=BaseModel)

# Роли сообщений: user — клиент, assistant — вопросы оркестратора, plan — предложенный план
LLM_ROLES = {"user": "user", "assistant": "assistant", "plan": "assistant"}


class OrchestratorError(RuntimeError):
    """Ошибка, которую нужно показать клиенту (сессия не найдена, отменена и т. п.)."""


class LLMFailure(RuntimeError):
    pass


@dataclass
class Reply:
    kind: Literal["questions", "plan", "error"]
    text: str
    questions: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, settings: Settings, db: Session, llm: BaseChatModel, session: ChatSession,
                 today: date | None = None, agent_llm: BaseChatModel | None = None) -> None:
        self.settings = settings
        self.db = db
        self.llm = llm
        self.agent_llm = agent_llm or llm
        self.session = session
        self.today = today or date.today()
        # Сообщения о ходе долгих шагов (аукциона) для интерфейса
        self.on_progress: Callable[[str], None] | None = None

    # ------------------------------------------------------------ сессии

    @classmethod
    def start(cls, settings: Settings, db: Session, llm: BaseChatModel, user_id: str,
              today: date | None = None, agent_llm: BaseChatModel | None = None) -> Orchestrator:
        today = today or date.today()
        while True:
            session_id = f"{today:%Y-%m-%d}-{secrets.token_hex(2)}"
            if db.get(ChatSession, session_id) is None:
                break
        session = ChatSession(id=session_id, user_id=user_id, status="active")
        db.add(session)
        db.commit()
        log.info("Новая сессия %s клиента %s", session_id, user_id)
        return cls(settings, db, llm, session, today, agent_llm)

    @classmethod
    def resume(cls, settings: Settings, db: Session, llm: BaseChatModel, user_id: str, session_id: str,
               today: date | None = None, agent_llm: BaseChatModel | None = None) -> Orchestrator:
        session = db.get(ChatSession, session_id)
        # Чужая сессия неотличима от несуществующей: клиент видит только свои (FR-27a)
        if session is None or session.user_id != user_id:
            raise OrchestratorError(f"сессия '{session_id}' не найдена")
        if session.status == "cancelled":
            raise OrchestratorError(f"сессия '{session_id}' отменена")
        return cls(settings, db, llm, session, today, agent_llm)

    def tasks(self) -> list[Task]:
        return list(self.db.scalars(
            select(Task).where(Task.session_id == self.session.id).order_by(Task.number)))

    def bids(self, task: Task) -> list[Bid]:
        return list(self.db.scalars(select(Bid).where(Bid.task_id == task.id).order_by(Bid.id)))

    def winner(self, task: Task) -> Bid | None:
        return self.db.get(Bid, task.winner_bid_id) if task.winner_bid_id else None

    def messages(self) -> list[Message]:
        return list(self.db.scalars(
            select(Message).where(Message.session_id == self.session.id).order_by(Message.id)))

    def cancel(self) -> None:
        for task in self.tasks():
            task.status = "cancelled"
            taskfile.write(task, self.tasks_dir, self.winner(task))
        self.session.status = "cancelled"
        self.db.commit()
        log.info("Сессия %s отменена", self.session.id)

    @property
    def tasks_dir(self):
        return self.settings.path("tasks_dir")

    # ------------------------------------------------------------ диалог

    def handle(self, text: str) -> Reply:
        """Обрабатывает сообщение клиента: задаёт вопросы или строит (перестраивает) план."""
        if self.session.status == "cancelled":
            return Reply("error", "Сессия отменена. Начните новую: перезапустите chat без --session.")
        self._save("user", text)

        categories = self._categories()
        if not categories:
            return Reply("error", "Нет ни одного активного агента — декомпозировать задачу не на что.")
        categories_text = "\n".join(f"- {c}: {d}" for c, d in categories.items())

        rounds = self._clarification_rounds()
        limit = self.settings.orchestrator.max_clarification_rounds
        force = prompts.FORCE_READY.format(rounds=rounds) if rounds >= limit else ""
        system = prompts.ANALYZE.format(today=self._today(), categories=categories_text, force=force)
        try:
            analysis = self._structured(Analysis, system)
        except LLMFailure:
            return self._llm_error()

        if analysis.budget_rub and analysis.budget_rub > 0:
            self.session.budget = to_minor(analysis.budget_rub)
        questions = [q.strip() for q in analysis.questions if q.strip()]
        if not analysis.ready and questions and rounds < limit:
            self._save("assistant", "\n".join(questions))
            self.db.commit()
            return Reply("questions", "Уточните, пожалуйста:", questions=questions)

        budget = fmt(self.session.budget) if self.session.budget else "не указан"
        system = prompts.DECOMPOSE.format(today=self._today(), categories=categories_text,
                                          summary=analysis.summary or "см. диалог", budget=budget)
        try:
            plan = normalize(self._structured(Decomposition, system), set(categories))
        except LLMFailure:
            return self._llm_error()
        for w in plan.warnings:
            log.warning("Сессия %s: %s", self.session.id, w)
        if not plan.tasks:
            self.db.commit()
            return Reply("error", "Не получилось разбить задачу на подзадачи для доступных агентов. "
                                  "Опишите её подробнее.", warnings=plan.warnings)

        replanned = self.session.status in ("planned", "awaiting_approval")
        self._clear_tasks()
        tasks = []
        for p in plan.tasks:
            task = Task(
                session_id=self.session.id, number=p.number, title=p.title, description=p.description,
                criteria=p.criteria, category=p.category, depends_on=p.depends_on,
                recommended=p.recommended, enabled=not p.recommended,  # FR-9a: рекомендованные выключены
                status="new", budget_hint=to_minor(p.budget_hint_rub) if p.budget_hint_rub else None,
            )
            self.db.add(task)
            tasks.append(task)
        self.db.flush()
        for task in tasks:
            taskfile.write(task, self.tasks_dir)
        self.session.status = "planned"
        self.db.commit()
        log.info("Сессия %s: план из %d подзадач", self.session.id, len(tasks))

        # Аукцион по каждой подзадаче, включая рекомендованные (FR-9a)
        from agentic_economics.auction.runner import run_auctions

        run_auctions(self.settings, self.db, tasks, self.agent_llm, self.llm, self.today, self.on_progress)
        self.session.status = "awaiting_approval"
        warnings = plan.warnings + self._auction_warnings(tasks)
        self._save("plan", plan_summary(tasks, self.winner))
        self.db.commit()
        text = "План обновлён, аукцион проведён." if replanned else "План готов, аукцион проведён."
        return Reply("plan", text, tasks=tasks, warnings=warnings)

    # ------------------------------------------------------------ служебное

    def _structured(self, schema: type[S], system: str) -> S:
        """Структурированный ответ LLM. Невалидный ответ — ещё одна попытка, затем LLMFailure (NFR-2)."""
        messages = [("system", system)] + [(LLM_ROLES[m.role], m.content) for m in self.messages()]
        runnable = self.llm.with_structured_output(schema)
        error: Exception | None = None
        for attempt in (1, 2):
            try:
                result = runnable.invoke(messages)
                if isinstance(result, dict):
                    result = schema.model_validate(result)
                if result is None:
                    raise ValueError("модель не вернула структурированный ответ")
                log.debug("%s: %s", schema.__name__, result)
                return result
            except Exception as exc:
                error = exc
                log.warning("%s, попытка %d: %s: %s", schema.__name__, attempt, type(exc).__name__, exc)
        raise LLMFailure(str(error)) from error

    def _llm_error(self) -> Reply:
        self.db.commit()
        return Reply("error", "Модель вернула некорректный ответ или недоступна. "
                              "Попробуйте ещё раз или переформулируйте сообщение.")

    def _save(self, role: str, content: str) -> None:
        self.db.add(Message(session_id=self.session.id, role=role, content=content))
        self.db.flush()

    def _clarification_rounds(self) -> int:
        """Сколько раз клиенту задавали вопросы после последнего плана."""
        rounds = 0
        for m in self.messages():
            if m.role == "plan":
                rounds = 0
            elif m.role == "assistant":
                rounds += 1
        return rounds

    def _categories(self) -> dict[str, str]:
        """Категории, в которых есть активные агенты, с описаниями агентов."""
        result: dict[str, list[str]] = {}
        for a in self.db.scalars(select(Agent).where(Agent.active.is_(True)).order_by(Agent.category, Agent.id)):
            result.setdefault(a.category, []).append(a.description)
        return {c: "; ".join(d) for c, d in result.items()}

    def _auction_warnings(self, tasks: list[Task]) -> list[str]:
        """Подзадачи без ставок и контроль бюджета (FR-9)."""
        warnings = []
        total = cheapest = 0
        for task in tasks:
            winner = self.winner(task)
            if winner is None:
                warnings.append(f"{taskfile.task_key(task.number)}: ни одной ставки — подзадачу некому выполнить")
                continue
            if task.enabled:
                total += winner.price
                cheapest += min(b.price for b in self.bids(task) if b.status in ("winner", "lost"))
        budget = self.session.budget
        if budget and total > budget:
            text = f"Сумма лучших ставок {fmt(total)} превышает бюджет {fmt(budget)} на {fmt(total - budget)}."
            if cheapest < total:
                text += f" Самые дешёвые ставки дают {fmt(cheapest)}."
            text += " Можно выбрать более дешёвых исполнителей, отказаться от части подзадач или изменить требования."
            warnings.append(text)
        return warnings

    def _clear_tasks(self) -> None:
        """Удаляет прежний план сессии вместе со ставками (до подтверждения денег по ним нет)."""
        ids = select(Task.id).where(Task.session_id == self.session.id)
        self.db.execute(update(Task).where(Task.session_id == self.session.id).values(winner_bid_id=None))
        self.db.execute(delete(Bid).where(Bid.task_id.in_(ids)))
        self.db.execute(delete(Task).where(Task.session_id == self.session.id))
        folder = self.tasks_dir / self.session.id
        if folder.exists():
            shutil.rmtree(folder)

    def _today(self) -> str:
        return f"{self.today:%d.%m.%Y}"


def plan_summary(tasks: list[Task], winner_of: Callable[[Task], Bid | None] = lambda t: None) -> str:
    lines = ["План:"]
    for t in tasks:
        extra = []
        winner = winner_of(t)
        if winner:
            extra.append(f"исполнитель {winner.agent_id}, {fmt(winner.price)}")
        if t.depends_on:
            extra.append("после " + ", ".join(taskfile.task_key(n) for n in t.depends_on))
        if t.recommended:
            extra.append("рекомендовано")
        suffix = f" ({'; '.join(extra)})" if extra else ""
        lines.append(f"{taskfile.task_key(t.number)} [{t.category}] {t.title}{suffix}")
    return "\n".join(lines)
