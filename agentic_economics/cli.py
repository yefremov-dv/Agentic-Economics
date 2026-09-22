"""Интерфейс командной строки (разделы 5.1 и 5.7 ТЗ)."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Column, Table
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from agentic_economics.agents.registry import SyncReport, sync_agents
from agentic_economics.agents.spec import ID_PATTERN, load_spec, set_active
from agentic_economics.config import Settings, load_settings
from agentic_economics.llm import LLMNotConfigured, get_llm
from agentic_economics.logging_setup import setup_logging
from agentic_economics.money import fmt, to_minor
from agentic_economics.orchestrator import Orchestrator, OrchestratorError, Reply
from agentic_economics.orchestrator.taskfile import task_key
from agentic_economics.storage.db import SchemaOutdated, make_sessionmaker
from agentic_economics.storage.models import Agent, Bid, Provider, Task, Transaction, User, Wallet
from agentic_economics.storage.models import Session as ChatSession
from agentic_economics.wallet.ledger import balance, find_wallet, get_wallet, topup

log = logging.getLogger(__name__)
console = Console()
err_console = Console(stderr=True)

app = typer.Typer(help="Agentic Economics — демо агентной экономики.", no_args_is_help=True)
agent_app = typer.Typer(help="Агенты-исполнители (администратор).", no_args_is_help=True)
provider_app = typer.Typer(help="Провайдеры — юрлица (администратор).", no_args_is_help=True)
user_app = typer.Typer(help="Клиенты — физлица (администратор).", no_args_is_help=True)
wallet_app = typer.Typer(help="Кошельки (администратор).", no_args_is_help=True)
auction_app = typer.Typer(help="История аукционов (администратор).", no_args_is_help=True)
app.add_typer(agent_app, name="agent")
app.add_typer(provider_app, name="provider")
app.add_typer(user_app, name="user")
app.add_typer(wallet_app, name="wallet")
app.add_typer(auction_app, name="auction")


@dataclass
class AppState:
    settings: Settings
    admin: bool

    @cached_property
    def sessions(self) -> sessionmaker:
        try:
            return make_sessionmaker(self.settings)
        except SchemaOutdated as exc:
            raise fail(f"база устарела: {exc}")

    def db(self) -> Session:
        return self.sessions()


def fail(message: str) -> typer.Exit:
    err_console.print(f"[bold red]Ошибка:[/] {message}")
    return typer.Exit(code=1)


def state(ctx: typer.Context) -> AppState:
    return ctx.find_root().obj


def require_admin(ctx: typer.Context) -> AppState:
    st = state(ctx)
    if not st.admin:
        raise fail("команда доступна только администратору. Запустите с флагом --admin, например: "
                   "python -m agentic_economics --admin agent list")
    return st


@app.callback()
def main(
    ctx: typer.Context,
    admin: Annotated[bool, typer.Option("--admin", help="Режим администратора (без пароля, FR-30).")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Подробные логи в logs/agentic.log.")] = False,
    config: Annotated[Optional[Path], typer.Option("--config", help="Путь к config.yaml.")] = None,
) -> None:
    try:
        settings = load_settings(config)
    except ValidationError as exc:
        raise fail(f"некорректный config.yaml:\n{exc}")
    setup_logging(settings, verbose)
    ctx.obj = AppState(settings=settings, admin=admin)


for _sub in (agent_app, provider_app, user_app, wallet_app, auction_app):
    _sub.callback()(require_admin)


# ---------------------------------------------------------------- клиент


CHAT_HELP = """\
[bold]Команды чата[/]
  /plan     — план: подзадачи, зависимости, победители аукционов
  /bids     — все ставки; /bids task-N — ставки по одной подзадаче
  /history  — история диалога
  /balance  — баланс кошелька
  /cancel   — отменить сессию
  /help     — эта справка
  /exit     — выход (сессию можно продолжить: chat --session <id>)
Любой другой текст — сообщение оркестратору. Если план уже готов, сообщение меняет требования,
и план строится заново.
[dim]Подтверждение плана и выполнение появятся на этапе 5.[/]"""


@app.command()
def chat(
    ctx: typer.Context,
    user: Annotated[str, typer.Option("--user", "-u", help="ID клиента.")],
    session: Annotated[Optional[str], typer.Option("--session", help="Продолжить сессию.")] = None,
) -> None:
    """Чат с оркестратором."""
    st = state(ctx)
    with st.db() as db:
        client = db.get(User, user)
        if client is None or client.role != "client":
            raise fail(f"клиент '{user}' не найден. Список клиентов: python -m agentic_economics --admin user list")
        wallet = get_wallet(db, "client", client.id, st.settings.wallet.currency)
        db.commit()
        try:
            llm = get_llm(st.settings, st.settings.orchestrator.model)
            agent_llm = get_llm(st.settings, st.settings.auction.agent_model)
        except LLMNotConfigured as exc:
            raise fail(str(exc))

        try:
            if session:
                orch = Orchestrator.resume(st.settings, db, llm, client.id, session, agent_llm=agent_llm)
            else:
                orch = Orchestrator.start(st.settings, db, llm, client.id, agent_llm=agent_llm)
        except OrchestratorError as exc:
            raise fail(str(exc))

        console.print(f"[bold]Agentic Economics[/] — здравствуйте, {client.name}! Введите /help для справки.")
        if session:
            console.print(f"Продолжаем сессию [bold]{orch.session.id}[/].")
            print_history(orch, last=6)
            if orch.tasks():
                print_plan(orch)
        else:
            console.print(f"Сессия [bold]{orch.session.id}[/]. Опишите задачу.")
            others = db.scalars(select(ChatSession.id).where(
                ChatSession.user_id == client.id, ChatSession.status.not_in(("cancelled", "done")),
                ChatSession.id != orch.session.id)).all()
            if others:
                console.print(f"[dim]Незавершённые сессии: {', '.join(others)} (продолжить: --session <id>)[/]")

        while True:
            try:
                text = console.input("[bold cyan]вы>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/help":
                console.print(CHAT_HELP)
            elif text == "/balance":
                console.print(f"Баланс: [bold]{fmt(balance(db, wallet), wallet.currency)}[/]")
            elif text == "/plan":
                if orch.tasks():
                    print_plan(orch)
                else:
                    console.print("Плана пока нет — опишите задачу.")
            elif text == "/history":
                print_history(orch)
            elif text == "/bids" or text.startswith("/bids "):
                print_bids(orch, text.removeprefix("/bids").strip())
            elif text == "/cancel":
                if typer.confirm("Отменить сессию?", default=False):
                    orch.cancel()
                    console.print("Сессия отменена.")
                    break
            elif text.startswith("/"):
                console.print(f"Неизвестная команда {text}. /help — список команд.")
            else:
                with console.status("Оркестратор думает…") as status:
                    orch.on_progress = lambda message: status.update(message)
                    reply = orch.handle(text)
                print_reply(orch, reply)
    console.print("До встречи!")


def print_reply(orch: Orchestrator, reply: Reply) -> None:
    if reply.kind == "questions":
        console.print(f"[bold magenta]оркестратор>[/] {reply.text}")
        for q in reply.questions:
            console.print(f"  • {q}")
    elif reply.kind == "plan":
        console.print(f"[bold magenta]оркестратор>[/] {reply.text}")
        print_plan(orch)
    else:
        console.print(f"[bold magenta]оркестратор>[/] [yellow]{reply.text}[/]")
    for w in reply.warnings:
        console.print(f"[dim]  ! {w}[/]")


def print_plan(orch: Orchestrator) -> None:
    tasks = orch.tasks()
    table = Table("#", "Подзадача", "Категория", "Зависит от", "Победитель", "Цена", "Ставок", "Статус", title="План")
    total = 0
    for t in tasks:
        title = t.title + ("\n[dim]рекомендовано, выключено[/]" if t.recommended and not t.enabled else "")
        winner = orch.winner(t)
        bids = [b for b in orch.bids(t) if b.status in ("winner", "lost")]
        if winner and t.enabled:
            total += winner.price
        table.add_row(task_key(t.number), title, t.category,
                      ", ".join(task_key(n) for n in t.depends_on) or "—",
                      winner.agent_id if winner else "—", fmt(winner.price) if winner else "—",
                      str(len(bids)) if t.status != "new" else "—", t.status)
    console.print(table)
    if total:
        line = f"Итого по включённым подзадачам: [bold]{fmt(total)}[/]"
        if orch.session.budget:
            rest = orch.session.budget - total
            line += f", бюджет {fmt(orch.session.budget)}, " + (
                f"остаток {fmt(rest)}" if rest >= 0 else f"[red]превышение {fmt(-rest)}[/]")
        console.print(line)
    console.print(f"[dim]Файлы: {orch.tasks_dir / orch.session.id}/[/]")


BID_STATUS = {"winner": "[green]победитель[/]", "lost": "проиграл", "declined": "[dim]отказ[/]",
              "timeout": "[red]таймаут[/]", "error": "[red]ошибка[/]"}


def print_bids(orch: Orchestrator, which: str = "") -> None:
    tasks = orch.tasks()
    if which:
        tasks = [t for t in tasks if task_key(t.number) == which or str(t.number) == which]
        if not tasks:
            console.print(f"Подзадача {which} не найдена.")
            return
    shown = False
    for t in tasks:
        bids = orch.bids(t)
        if not bids:
            continue
        shown = True
        agents = {a.id: a for a in orch.db.scalars(select(Agent).where(Agent.id.in_([b.agent_id for b in bids])))}
        table = Table("Агент", "Цена", "Срок", "Уверен.", "Рейтинг", "Оценка", "Статус", "Предложение",
                      title=f"{task_key(t.number)}: {t.title}", title_justify="left")
        ordered = sorted(bids, key=lambda b: (b.score is None, -(b.score or 0), b.id))
        for b in ordered:
            a = agents.get(b.agent_id)
            valid = b.status in ("winner", "lost")
            table.add_row(b.agent_id, fmt(b.price) if valid else "—", b.eta or "—",
                          f"{b.confidence:.2f}" if valid else "—",
                          f"{a.rating:.1f} ({a.ratings_count})" if a else "—",
                          f"{b.score:.3f}" if b.score is not None else "—",
                          BID_STATUS.get(b.status, b.status), b.offer)
        console.print(table)
    if not shown:
        console.print("Ставок пока нет — сначала опишите задачу.")


def print_history(orch: Orchestrator, last: int | None = None) -> None:
    messages = orch.messages()
    if last:
        messages = messages[-last:]
    if not messages:
        console.print("[dim]История пуста.[/]")
    names = {"user": "[bold cyan]вы[/]", "assistant": "[bold magenta]оркестратор[/]", "plan": "[bold magenta]оркестратор[/]"}
    for m in messages:
        console.print(f"{names.get(m.role, m.role)}> {m.content}")


@app.command("check-llm")
def check_llm(ctx: typer.Context) -> None:
    """Проверить подключение к GigaChat."""
    st = state(ctx)
    try:
        llm = get_llm(st.settings)
        with console.status("Запрос к GigaChat…"):
            reply = llm.invoke("Ответь одним словом: работает ли связь?")
    except LLMNotConfigured as exc:
        raise fail(str(exc))
    except Exception as exc:  # сетевые ошибки, авторизация, TLS
        log.exception("Ошибка подключения к GigaChat")
        raise fail(f"GigaChat недоступен: {type(exc).__name__}: {exc}")
    model = (reply.response_metadata or {}).get("model_name", st.settings.llm.model)
    console.print(f"[green]GigaChat отвечает[/] (модель {model}): {reply.content}")


# ---------------------------------------------------------------- администрирование


@app.command("reset-demo")
def reset_demo_cmd(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Не спрашивать подтверждение.")] = False,
) -> None:
    """Пересоздать базу из data/seed.yaml и agents/*.yaml (удаляет все данные)."""
    from agentic_economics.seed import reset_demo

    st = require_admin(ctx)
    if not yes and not typer.confirm(f"Все данные в {st.settings.path('db')} будут удалены. Продолжить?"):
        raise typer.Exit()
    try:
        report = reset_demo(st.settings)
    except (FileNotFoundError, ValidationError) as exc:
        raise fail(f"не удалось загрузить seed: {exc}")
    console.print("[green]Демо-база пересоздана.[/]")
    print_sync_report(report)


@app.command()
def ledger(
    ctx: typer.Context,
    owner: Annotated[Optional[str], typer.Option("--owner", help="Владелец кошелька: [<тип>:]<id>, например client:ivanov или sky-lines.")] = None,
    limit: Annotated[int, typer.Option(help="Сколько последних записей показать.")] = 50,
) -> None:
    """Журнал транзакций."""
    st = require_admin(ctx)
    with st.db() as db:
        wallets = {w.id: w for w in db.scalars(select(Wallet))}
        query = select(Transaction).order_by(Transaction.id.desc()).limit(limit)
        if owner:
            wallet = resolve_wallet(db, owner)
            if wallet is None:
                raise fail(f"кошелёк {owner} не найден")
            query = query.where(or_(Transaction.debit_wallet_id == wallet.id, Transaction.credit_wallet_id == wallet.id))
        table = Table("#", "Время (UTC)", "Тип", "Откуда", "Куда", "Сумма", "Задача", "Комментарий")
        for tx in reversed(db.scalars(query).all()):
            src, dst = wallets[tx.debit_wallet_id], wallets[tx.credit_wallet_id]
            table.add_row(
                str(tx.id), f"{tx.ts:%Y-%m-%d %H:%M:%S}", tx.kind,
                f"{src.owner_type}:{src.owner_id}", f"{dst.owner_type}:{dst.owner_id}",
                fmt(tx.amount, dst.currency), str(tx.task_id or ""), tx.note or "",
            )
        console.print(table)


def resolve_wallet(db: Session, owner: str) -> Wallet | None:
    """Кошелёк по строке "<тип>:<id>". Без типа ищется клиент, затем провайдер."""
    kind, sep, owner_id = owner.partition(":")
    if sep:
        return find_wallet(db, kind, owner_id)
    return find_wallet(db, "client", owner) or find_wallet(db, "provider", owner)


def print_sync_report(report: SyncReport) -> None:
    if report.added:
        console.print(f"Добавлены агенты: {', '.join(report.added)}")
    if report.updated:
        console.print(f"Обновлены агенты: {', '.join(report.updated)}")
    if report.deactivated:
        console.print(f"[yellow]Отключены (нет YAML-файла):[/] {', '.join(report.deactivated)}")
    for name, error in report.errors.items():
        console.print(f"[red]Пропущен {name}:[/] {error}")


# --- agent


@agent_app.command("list")
def agent_list(
    ctx: typer.Context,
    category: Annotated[Optional[str], typer.Option(help="Фильтр по категории.")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Показывать отключённых.")] = False,
) -> None:
    """Список агентов."""
    st = state(ctx)
    with st.db() as db:
        query = select(Agent).order_by(Agent.category, Agent.id)
        if category:
            query = query.where(Agent.category == category)
        if not all_:
            query = query.where(Agent.active.is_(True))
        table = Table(Column("ID", no_wrap=True), "Название", "Провайдер", "Категория", "Рейтинг", "Оценок", "Активен")
        for a in db.scalars(query):
            table.add_row(a.id, a.name, a.provider_id, a.category, f"{a.rating:.1f}", str(a.ratings_count),
                          "да" if a.active else "нет")
        console.print(table)


@agent_app.command("show")
def agent_show(ctx: typer.Context, agent_id: str) -> None:
    """Карточка агента."""
    st = state(ctx)
    with st.db() as db:
        a = db.get(Agent, agent_id)
        if a is None:
            raise fail(f"агент '{agent_id}' не найден")
        provider = db.get(Provider, a.provider_id)
        cfg = a.config_json
        console.print(f"[bold]{a.name}[/] ({a.id})")
        console.print(f"Провайдер: {provider.name if provider else a.provider_id} ({a.provider_id})")
        console.print(f"Категория: {a.category}   Активен: {'да' if a.active else 'нет'}")
        console.print(f"Рейтинг: {a.rating:.1f} (оценок: {a.ratings_count}, стартовый: {a.initial_rating:.1f})")
        console.print(f"Описание: {a.description}")
        console.print(f"Ценообразование: наценка {cfg['pricing']['base_markup']:.0%}, "
                      f"минимальная цена {cfg['pricing']['min_price']:,.0f}".replace(",", " "))
        console.print(f"Моковые данные: {cfg.get('mock_data') or '—'}")
        console.print(f"Файл: {a.source_file}")
        console.print("[bold]Системный промпт:[/]")
        console.print(cfg["system_prompt"].strip())


@agent_app.command("reload")
def agent_reload(ctx: typer.Context) -> None:
    """Перечитать agents/*.yaml и синхронизировать с базой."""
    st = state(ctx)
    with st.db() as db:
        report = sync_agents(db, st.settings.path("agents_dir"))
    print_sync_report(report)


@agent_app.command("add")
def agent_add(ctx: typer.Context, file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Добавить агента из YAML-файла (копируется в agents/)."""
    st = state(ctx)
    try:
        spec = load_spec(file)
    except ValidationError as exc:
        raise fail(f"некорректное описание агента:\n{exc}")
    agents_dir = st.settings.path("agents_dir")
    target = agents_dir / f"{spec.id}.yaml"
    with st.db() as db:
        if db.get(Provider, spec.provider) is None:
            raise fail(f"провайдер '{spec.provider}' не найден. Сначала: provider add {spec.provider} --name ...")
        if target.exists() and target.resolve() != file.resolve():
            raise fail(f"файл {target} уже существует")
        agents_dir.mkdir(parents=True, exist_ok=True)
        if target.resolve() != file.resolve():
            shutil.copyfile(file, target)
        report = sync_agents(db, agents_dir)
    print_sync_report(report)


@agent_app.command("disable")
def agent_disable(ctx: typer.Context, agent_id: str) -> None:
    """Отключить агента (active: false в его YAML-файле)."""
    set_agent_active(ctx, agent_id, False)


@agent_app.command("enable")
def agent_enable(ctx: typer.Context, agent_id: str) -> None:
    """Включить агента (active: true в его YAML-файле)."""
    set_agent_active(ctx, agent_id, True)


def set_agent_active(ctx: typer.Context, agent_id: str, active: bool) -> None:
    st = state(ctx)
    with st.db() as db:
        a = db.get(Agent, agent_id)
        if a is None or not a.source_file or not Path(a.source_file).exists():
            raise fail(f"агент '{agent_id}' не найден или у него нет YAML-файла")
        set_active(Path(a.source_file), active)
        sync_agents(db, st.settings.path("agents_dir"))
    console.print(f"Агент {agent_id} {'включён' if active else 'отключён'}.")


# --- provider


@provider_app.command("list")
def provider_list(ctx: typer.Context) -> None:
    """Список провайдеров."""
    st = state(ctx)
    with st.db() as db:
        table = Table(Column("ID", no_wrap=True), "Название", "ИНН", "Агентов", "Баланс")
        for p in db.scalars(select(Provider).order_by(Provider.id)):
            n = len(db.scalars(select(Agent.id).where(Agent.provider_id == p.id, Agent.active.is_(True))).all())
            w = find_wallet(db, "provider", p.id)
            table.add_row(p.id, p.name, p.inn or "", str(n), fmt(balance(db, w), w.currency) if w else "—")
        console.print(table)


@provider_app.command("add")
def provider_add(
    ctx: typer.Context,
    provider_id: str,
    name: Annotated[str, typer.Option("--name", help="Название юрлица.")],
    inn: Annotated[Optional[str], typer.Option(help="ИНН (вымышленный).")] = None,
) -> None:
    """Добавить провайдера и открыть ему кошелёк."""
    st = state(ctx)
    check_id(provider_id)
    with st.db() as db:
        if db.get(Provider, provider_id):
            raise fail(f"провайдер '{provider_id}' уже существует")
        db.add(Provider(id=provider_id, name=name, inn=inn))
        get_wallet(db, "provider", provider_id, st.settings.wallet.currency)
        db.commit()
    console.print(f"Провайдер {provider_id} добавлен.")


# --- user


@user_app.command("list")
def user_list(ctx: typer.Context) -> None:
    """Список клиентов."""
    st = state(ctx)
    with st.db() as db:
        table = Table(Column("ID", no_wrap=True), "Имя", "Баланс")
        for u in db.scalars(select(User).where(User.role == "client").order_by(User.id)):
            w = find_wallet(db, "client", u.id)
            table.add_row(u.id, u.name, fmt(balance(db, w), w.currency) if w else "—")
        console.print(table)


@user_app.command("add")
def user_add(
    ctx: typer.Context,
    user_id: str,
    name: Annotated[str, typer.Option("--name", help="Имя клиента.")],
    balance_: Annotated[Optional[float], typer.Option("--balance", help="Стартовый баланс, VRUB.")] = None,
) -> None:
    """Добавить клиента (FR-27a)."""
    st = state(ctx)
    check_id(user_id)
    amount = st.settings.wallet.default_client_balance if balance_ is None else balance_
    if amount < 0:
        raise fail("баланс не может быть отрицательным")
    with st.db() as db:
        if db.get(User, user_id):
            raise fail(f"клиент '{user_id}' уже существует")
        db.add(User(id=user_id, name=name, role="client"))
        wallet = get_wallet(db, "client", user_id, st.settings.wallet.currency)
        if amount:
            topup(db, wallet, to_minor(amount), note="Стартовый баланс")
        db.commit()
        console.print(f"Клиент {user_id} добавлен, баланс {fmt(balance(db, wallet), wallet.currency)}.")


@user_app.command("show")
def user_show(ctx: typer.Context, user_id: str) -> None:
    """Карточка клиента."""
    st = state(ctx)
    with st.db() as db:
        u = db.get(User, user_id)
        if u is None:
            raise fail(f"клиент '{user_id}' не найден")
        w = find_wallet(db, "client", u.id)
        console.print(f"[bold]{u.name}[/] ({u.id}), роль: {u.role}")
        console.print(f"Баланс: {fmt(balance(db, w), w.currency) if w else '—'}")


def check_id(value: str) -> None:
    if not re.match(ID_PATTERN, value):
        raise fail("ID: латиница в нижнем регистре, цифры и дефис, 2–64 символа")


# --- wallet


@wallet_app.command("balance")
def wallet_balance(ctx: typer.Context, owner: Annotated[str, typer.Argument(help="<тип>:<id>, например client:ivanov")]) -> None:
    """Баланс кошелька."""
    st = state(ctx)
    with st.db() as db:
        w = resolve_wallet(db, owner)
        if w is None:
            raise fail(f"кошелёк {owner} не найден")
        console.print(f"{w.owner_type}:{w.owner_id} — [bold]{fmt(balance(db, w), w.currency)}[/]")


@wallet_app.command("topup")
def wallet_topup(
    ctx: typer.Context,
    owner: Annotated[str, typer.Argument(help="<тип>:<id>, например client:ivanov")],
    amount: Annotated[float, typer.Argument(help="Сумма, VRUB.")],
) -> None:
    """Пополнить кошелёк."""
    st = state(ctx)
    if amount <= 0:
        raise fail("сумма должна быть положительной")
    with st.db() as db:
        w = resolve_wallet(db, owner)
        if w is None or w.owner_type == "system":
            raise fail(f"кошелёк {owner} не найден")
        topup(db, w, to_minor(amount), note="Пополнение администратором")
        db.commit()
        console.print(f"Пополнено. Баланс {w.owner_type}:{w.owner_id} — {fmt(balance(db, w), w.currency)}")


# --- auction


@auction_app.command("show")
def auction_show(ctx: typer.Context, task_id: int) -> None:
    """Ставки аукциона по подзадаче."""
    st = state(ctx)
    with st.db() as db:
        task = db.get(Task, task_id)
        if task is None:
            raise fail(f"подзадача {task_id} не найдена (аукционы появятся на этапе 3)")
        table = Table("#", "Время (UTC)", "Агент", "Цена", "Срок", "Уверенность", "Оценка", "Статус", "Предложение")
        for b in db.scalars(select(Bid).where(Bid.task_id == task_id).order_by(Bid.id)):
            valid = b.status in ("winner", "lost")
            table.add_row(str(b.id), f"{b.ts:%H:%M:%S}", b.agent_id, fmt(b.price) if valid else "—", b.eta or "",
                          f"{b.confidence:.2f}" if valid else "—",
                          f"{b.score:.3f}" if b.score is not None else "", b.status, b.offer)
        console.print(f"[bold]task-{task.number}[/] ({task.session_id}): {task.title}")
        console.print(table)
