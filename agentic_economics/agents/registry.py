"""Синхронизация YAML-описаний агентов с таблицей agents. Источник истины — YAML."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_economics.agents.spec import AgentSpec, load_spec
from agentic_economics.storage.models import Agent, Provider

log = logging.getLogger(__name__)


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deactivated: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def read_specs(agents_dir: Path) -> tuple[dict[str, tuple[AgentSpec, Path]], dict[str, str]]:
    specs: dict[str, tuple[AgentSpec, Path]] = {}
    errors: dict[str, str] = {}
    for path in sorted(agents_dir.glob("*.y*ml")):
        try:
            spec = load_spec(path)
        except ValidationError as exc:
            errors[path.name] = _short(exc)
            continue
        except yaml.YAMLError as exc:
            errors[path.name] = f"некорректный YAML: {str(exc).splitlines()[0]}"
            continue
        if spec.id in specs:
            errors[path.name] = f"id '{spec.id}' уже объявлен в {specs[spec.id][1].name}"
            continue
        specs[spec.id] = (spec, path)
    return specs, errors


def _short(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())


def sync_agents(db: Session, agents_dir: Path) -> SyncReport:
    report = SyncReport()
    specs, report.errors = read_specs(agents_dir)
    providers = set(db.scalars(select(Provider.id)))
    existing = {a.id: a for a in db.scalars(select(Agent))}

    for agent_id, (spec, path) in specs.items():
        if spec.provider not in providers:
            report.errors[path.name] = f"провайдер '{spec.provider}' не найден"
            continue
        agent = existing.get(agent_id)
        if agent is None:
            agent = Agent(id=agent_id, rating=spec.rating, ratings_count=0)
            db.add(agent)
            report.added.append(agent_id)
        else:
            report.updated.append(agent_id)
        agent.provider_id = spec.provider
        agent.name = spec.name
        agent.category = spec.category
        agent.description = spec.description
        agent.config_json = spec.model_dump(mode="json")
        agent.initial_rating = spec.rating
        # накопленные оценки клиентов не сбрасываются (FR-34)
        if not agent.ratings_count:
            agent.rating = spec.rating
        agent.active = spec.active
        agent.source_file = str(path)

    for agent_id, agent in existing.items():
        if agent_id not in specs and agent.active:
            agent.active = False
            report.deactivated.append(agent_id)

    db.commit()
    for name, err in report.errors.items():
        log.warning("Агент %s пропущен: %s", name, err)
    return report
