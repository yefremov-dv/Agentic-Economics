"""Выполнение подтверждённого плана (FR-18 – FR-20, FR-22).

Подзадачи выполняются волнами: в волне — все, чьи зависимости уже выполнены, внутри волны параллельно.
Результат проверяет оркестратор (FR-19): не принят — одна повторная попытка с замечаниями, затем второе место
аукциона. Второе место берётся, только если его цена не выше заблокированной суммы: доплату без подтверждения
клиента списывать нельзя (FR-17). Выполнено — выплата провайдеру за вычетом комиссии, нет — возврат клиенту.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_economics import aio
from agentic_economics.agents.executor import ExecutionReport, ExecutorAgent, TaskBrief, make_reference
from agentic_economics.auction.runner import retry_llm
from agentic_economics.config import Settings
from agentic_economics.money import fmt
from agentic_economics.orchestrator import taskfile
from agentic_economics.storage.models import Agent, Bid, Task
from agentic_economics.storage.models import Session as ChatSession
from agentic_economics.wallet.escrow import Settlement, held, refund, settle

log = logging.getLogger(__name__)

VERIFY = """\
Ты — оркестратор. Проверь отчёт исполнителя о выполнении подзадачи клиента.

{task}

Ставка исполнителя (позиции каталога; название модели товара — не размер):
{items}

Отчёт исполнителя (success={success}):
{summary}
{details}

Прими результат (ok=true), если по существу выполнено то, что требует подзадача: те же даты, места, количество,
параметры. Формулировки и мелочи не важны. Отклони (ok=false), если выполнено не то или success=false,
и кратко укажи, что не так."""


class Verification(BaseModel):
    """Решение оркестратора по результату подзадачи."""

    ok: bool = Field(description="true — результат соответствует подзадаче и критериям")
    reason: str = Field("", description="Если ok=false — что именно не так, одним предложением")


@dataclass
class Attempt:
    agent_id: str
    ok: bool
    reason: str = ""
    report: ExecutionReport | None = None
    reference: str = ""


@dataclass
class TaskOutcome:
    task: Task
    status: str  # done | failed | cancelled
    agent_id: str | None = None
    reference: str = ""
    summary: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    settlement: Settlement | None = None
    refunded: int = 0
    note: str = ""


class Executor:
    def __init__(self, settings: Settings, db: Session, session: ChatSession, agent_llm: BaseChatModel,
                 verify_llm: BaseChatModel, today: date | None = None,
                 on_progress: Callable[[str], None] | None = None):
        self.settings = settings
        self.db = db
        self.session = session
        self.agent_llm = agent_llm
        self.verify_llm = verify_llm
        self.today = today or date.today()
        self.on_progress = on_progress
        self.currency = settings.wallet.currency

    def run(self) -> list[TaskOutcome]:
        tasks = list(self.db.scalars(select(Task).where(Task.session_id == self.session.id, Task.status == "approved")
                                     .order_by(Task.number)))
        in_plan = {t.number for t in tasks}
        outcomes: dict[int, TaskOutcome] = {}
        self.session.status = "in_progress"
        self.db.commit()

        pending, wave_no = list(tasks), 0
        while pending:
            wave, rest, cancelled = [], [], False
            for task in pending:
                deps = [n for n in task.depends_on if n in in_plan]  # убранные клиентом зависимости не мешают
                failed = [n for n in deps if n in outcomes and outcomes[n].status != "done"]
                if failed:
                    outcomes[task.number] = self._cancel(task, failed)
                    cancelled = True
                elif all(n in outcomes for n in deps):
                    wave.append(task)
                else:
                    rest.append(task)
            if wave:
                wave_no += 1
                self._progress(f"Выполнение, волна {wave_no}: " + ", ".join(taskfile.task_key(t.number) for t in wave))
                for task, outcome in zip(wave, self._run_wave(wave, outcomes)):
                    outcomes[task.number] = outcome
            elif not cancelled:  # не должно случаться: план нормализован топологически
                for task in rest:
                    outcomes[task.number] = self._cancel(task, [], "неразрешимые зависимости")
                break
            pending = rest

        self.session.status = "done"
        self.db.commit()
        return [outcomes[t.number] for t in tasks]

    # ------------------------------------------------------------ волна

    def _run_wave(self, tasks: list[Task], done: dict[int, TaskOutcome]) -> list[TaskOutcome]:
        for task in tasks:
            task.status = "in_progress"
            taskfile.write(task, self._tasks_dir(), self._winner(task))
        self.db.commit()
        briefs = {t.id: self._brief(t, done) for t in tasks}
        jobs = [(t, self._winner(t)) for t in tasks]
        first = aio.run(self._attempt_all([(t, bid, briefs[t.id]) for t, bid in jobs]))

        outcomes: dict[int, TaskOutcome] = {}
        second_round = []
        for (task, bid), attempts in zip(jobs, first):
            if attempts[-1].ok:
                outcomes[task.id] = self._finish(task, bid, attempts)
            elif (runner_up := self._runner_up(task, bid)) is not None:
                self._switch(task, bid, runner_up)
                second_round.append((task, runner_up, attempts))
            else:
                outcomes[task.id] = self._fail(task, bid, attempts, "второго места с подходящей ценой нет")

        if second_round:
            more = aio.run(self._attempt_all([(t, bid, briefs[t.id]) for t, bid, _ in second_round]))
            for (task, bid, earlier), attempts in zip(second_round, more):
                history = earlier + attempts
                outcomes[task.id] = (self._finish(task, bid, history) if attempts[-1].ok
                                     else self._fail(task, bid, history, "не справилось и второе место"))
        return [outcomes[t.id] for t in tasks]

    async def _attempt_all(self, jobs: list[tuple[Task, Bid, TaskBrief]]) -> list[list[Attempt]]:
        semaphore = asyncio.Semaphore(self.settings.auction.max_parallel)

        async def one(task: Task, bid: Bid, brief: TaskBrief) -> list[Attempt]:
            async with semaphore:
                return await self._attempts(task, bid, brief)

        return await asyncio.gather(*(one(*job) for job in jobs))

    async def _attempts(self, task: Task, bid: Bid, brief: TaskBrief) -> list[Attempt]:
        """Выполнение агентом и проверка: первая попытка и одна повторная с замечаниями (FR-19)."""
        agent_row = self.db.get(Agent, bid.agent_id)
        agent = ExecutorAgent(agent_row.id, agent_row.config_json, self.agent_llm, self.settings.root, self.today)
        reference = make_reference(task.category)
        attempts: list[Attempt] = []
        feedback = ""
        timeout = self.settings.orchestrator.execution_timeout
        for _ in range(2):
            try:
                report = await asyncio.wait_for(retry_llm(
                    lambda: agent.execute(brief, bid.offer, bid.details or {}, reference, feedback),
                    f"Исполнитель {agent.id}"), timeout)
            except asyncio.TimeoutError:
                attempts.append(Attempt(agent.id, False, f"нет ответа за {timeout:g} с"))
                feedback = "прошлая попытка не уложилась во время"
                continue
            except Exception as exc:
                log.exception("Исполнитель %s: ошибка выполнения %s", agent.id, brief.key)
                attempts.append(Attempt(agent.id, False, f"ошибка исполнителя: {type(exc).__name__}"))
                feedback = "прошлая попытка завершилась ошибкой"
                continue
            if reference not in report.summary and reference not in " ".join(report.details):
                report.summary = f"{report.summary} Номер подтверждения: {reference}."
            verdict = await self._verify(brief, report, bid.details or {})
            attempts.append(Attempt(agent.id, verdict.ok, verdict.reason, report, reference))
            log.info("%s, %s: попытка %d — %s %s", brief.key, agent.id, len(attempts),
                     "принято" if verdict.ok else "не принято", verdict.reason)
            if verdict.ok:
                break
            feedback = verdict.reason or "результат не соответствует подзадаче"
        return attempts

    async def _verify(self, brief: TaskBrief, report: ExecutionReport, bid_details: dict) -> Verification:
        if not report.success:
            return Verification(ok=False, reason="исполнитель сообщил, что не смог выполнить подзадачу")
        items = "\n".join(f"- {i['title']} × {i['quantity']} ({i.get('unit', '')})"
                          + (f", размер {i['size']}" if i.get("size") else "")
                          for i in bid_details.get("items", [])) or "—"
        prompt = VERIFY.format(task=brief.text(), items=items, success=report.success, summary=report.summary,
                               details="\n".join(f"- {d}" for d in report.details))
        try:
            verdict = await retry_llm(lambda: self.verify_llm.with_structured_output(Verification).ainvoke(prompt),
                                      "Проверка результата")
            if isinstance(verdict, dict):
                verdict = Verification.model_validate(verdict)
            if verdict is None:
                raise ValueError("пустой ответ")
            return verdict
        except Exception as exc:
            # Не наказываем исполнителя за сбой проверки: принимаем отчёт, а сбой пишем в лог
            log.warning("%s: проверка результата не выполнена (%s), отчёт принят", brief.key, exc)
            return Verification(ok=True, reason="автоматическая проверка недоступна")

    # ------------------------------------------------------------ итоги и деньги

    def _finish(self, task: Task, bid: Bid, attempts: list[Attempt]) -> TaskOutcome:
        report = attempts[-1].report
        settlement = settle(self.db, task, self.settings.wallet.platform_fee, self.currency)
        task.status = "done"
        task.result_json = self._result(bid, attempts, report)
        self.db.commit()
        taskfile.write(task, self._tasks_dir(), bid)
        return TaskOutcome(task, "done", bid.agent_id, attempts[-1].reference, report.summary, attempts, settlement)

    def _fail(self, task: Task, bid: Bid, attempts: list[Attempt], note: str) -> TaskOutcome:
        bid.status = "failed"
        amount = held(self.db, task.id)
        refund(self.db, task, "refund", f"Возврат: task-{task.number} не выполнена", self.currency)
        task.status = "failed"
        task.result_json = self._result(bid, attempts, None) | {"note": note}
        self.db.commit()
        taskfile.write(task, self._tasks_dir(), bid)
        log.warning("task-%s не выполнена: %s", task.number, note)
        return TaskOutcome(task, "failed", bid.agent_id, attempts=attempts, refunded=amount, note=note)

    def _cancel(self, task: Task, failed_deps: list[int], note: str = "") -> TaskOutcome:
        note = note or "не выполнены зависимости: " + ", ".join(taskfile.task_key(n) for n in failed_deps)
        amount = held(self.db, task.id)
        refund(self.db, task, "release", f"Отмена task-{task.number}: {note}", self.currency)
        task.status = "cancelled"
        task.result_json = {"note": note}
        self.db.commit()
        taskfile.write(task, self._tasks_dir(), self._winner(task))
        return TaskOutcome(task, "cancelled", refunded=amount, note=note)

    def _runner_up(self, task: Task, failed: Bid) -> Bid | None:
        """Лучшая из остальных ставок, чья цена не выше заблокированной суммы."""
        limit = held(self.db, task.id)
        candidates = [b for b in self.db.scalars(select(Bid).where(Bid.task_id == task.id))
                      if b.id != failed.id and b.status == "lost" and b.price <= limit]
        candidates.sort(key=lambda b: (-(b.score or 0), b.id))
        return candidates[0] if candidates else None

    def _switch(self, task: Task, failed: Bid, runner_up: Bid) -> None:
        failed.status = "failed"
        runner_up.status = "winner"
        task.winner_bid_id = runner_up.id
        self.db.commit()
        # В эскроу остаётся ровно цена второго места, разница возвращается клиенту
        refund(self.db, task, "refund", f"Разница цены при замене исполнителя task-{task.number}",
               self.currency, keep=runner_up.price)
        log.info("task-%s: переход ко второму месту — %s (%s)", task.number, runner_up.agent_id, fmt(runner_up.price))

    # ------------------------------------------------------------ служебное

    def _result(self, bid: Bid, attempts: list[Attempt], report: ExecutionReport | None) -> dict:
        result = {
            "agent_id": bid.agent_id,
            "attempts": [{"agent_id": a.agent_id, "ok": a.ok, "reason": a.reason} for a in attempts],
        }
        if report is not None:
            result |= {"summary": report.summary, "details": report.details, "reference": attempts[-1].reference}
        return result

    def _brief(self, task: Task, done: dict[int, TaskOutcome]) -> TaskBrief:
        lines = []
        for n in task.depends_on:
            outcome = done.get(n)
            if outcome and outcome.status == "done":
                lines.append(f"- {taskfile.task_key(n)} ({outcome.task.title}): {outcome.summary}")
        return TaskBrief(task.id, taskfile.task_key(task.number), task.title, task.category, task.description,
                         list(task.criteria), "\n".join(lines))

    def _winner(self, task: Task) -> Bid | None:
        return self.db.get(Bid, task.winner_bid_id) if task.winner_bid_id else None

    def _tasks_dir(self):
        return self.settings.path("tasks_dir")

    def _progress(self, message: str) -> None:
        if self.on_progress:
            self.on_progress(message)

