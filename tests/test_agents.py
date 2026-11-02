from pathlib import Path

import pytest
from sqlalchemy import select

from agentic_economics.agents.registry import sync_agents
from agentic_economics.agents.spec import CATEGORIES, load_spec, set_active
from agentic_economics.storage.models import Agent

ROOT = Path(__file__).resolve().parent.parent


def test_repository_agents_are_valid():
    specs = [load_spec(p) for p in sorted((ROOT / "agents").glob("*.yaml"))]
    assert 10 <= len(specs) <= 12
    by_category = {c: {s.provider for s in specs if s.category == c} for c in CATEGORIES}
    # для содержательного аукциона — минимум два провайдера в каждой категории (п. 6.5)
    assert all(len(providers) >= 2 for providers in by_category.values()), by_category
    for path in (ROOT / "agents").glob("*.yaml"):
        assert load_spec(path).id == path.stem


def test_seed_loads_all_agents(db):
    assert len(db.scalars(select(Agent).where(Agent.active.is_(True))).all()) == 11


def test_invalid_files_are_reported_not_fatal(db, seeded):
    agents_dir = seeded.path("agents_dir")
    (agents_dir / "broken.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    (agents_dir / "bad-category.yaml").write_text(
        "id: bad-category\nname: x\nprovider: shield\ncategory: taxi\ndescription: x\nsystem_prompt: x\n",
        encoding="utf-8",
    )
    (agents_dir / "orphan.yaml").write_text(
        "id: orphan\nname: x\nprovider: nobody\ncategory: retail\ndescription: x\nsystem_prompt: x\n",
        encoding="utf-8",
    )
    report = sync_agents(db, agents_dir)
    assert set(report.errors) == {"broken.yaml", "bad-category.yaml", "orphan.yaml"}
    assert "category" in report.errors["bad-category.yaml"]
    assert "nobody" in report.errors["orphan.yaml"]
    assert len(report.updated) == 11


def test_removed_file_deactivates_agent(db, seeded):
    (seeded.path("agents_dir") / "retail-sportline.yaml").unlink()
    report = sync_agents(db, seeded.path("agents_dir"))
    assert report.deactivated == ["retail-sportline"]
    assert db.get(Agent, "retail-sportline").active is False


def test_reload_keeps_client_ratings(db, seeded):
    agent = db.get(Agent, "hotels-sunrise")
    agent.rating, agent.ratings_count = 3.0, 2
    db.commit()
    sync_agents(db, seeded.path("agents_dir"))
    db.refresh(agent)
    assert (agent.rating, agent.ratings_count, agent.initial_rating) == (3.0, 2, 4.6)


def test_new_yaml_agent_joins_catalog(db, seeded):
    agents_dir = seeded.path("agents_dir")
    text = (agents_dir / "retail-sportline.yaml").read_text(encoding="utf-8")
    (agents_dir / "retail-sneakerhub.yaml").write_text(
        text.replace("id: retail-sportline", "id: retail-sneakerhub"), encoding="utf-8"
    )
    assert sync_agents(db, agents_dir).added == ["retail-sneakerhub"]
    assert db.get(Agent, "retail-sneakerhub").category == "retail"


def test_set_active_preserves_formatting(tmp_path):
    source = ROOT / "agents" / "transfers-kara.yaml"
    path = tmp_path / source.name
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    assert set_active(path, False).active is False
    assert path.read_text(encoding="utf-8") == source.read_text(encoding="utf-8").replace(
        "active: true", "active: false"
    )


def test_set_active_rejects_invalid_file(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("id: x\n", encoding="utf-8")
    with pytest.raises(Exception):
        set_active(path, True)
    assert path.read_text(encoding="utf-8") == "id: x\n"
