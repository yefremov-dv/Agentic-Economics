import pytest
from typer.testing import CliRunner

from agentic_economics.cli import app

runner = CliRunner()


@pytest.fixture
def run(seeded, config_file, monkeypatch):
    monkeypatch.setattr("agentic_economics.cli.console.width", 200)

    def _run(*args, input=None):
        return runner.invoke(app, ["--config", str(config_file), *args], input=input)

    return _run


@pytest.mark.parametrize(
    "args",
    [["agent", "list"], ["user", "add", "x1", "--name", "X"], ["wallet", "balance", "ivanov"], ["ledger"],
     ["reset-demo", "-y"], ["provider", "list"], ["auction", "show", "1"]],
)
def test_admin_commands_require_flag(run, args):
    result = run(*args)
    assert result.exit_code == 1
    assert "--admin" in result.output


def test_user_add_and_balance(run):
    assert run("--admin", "user", "add", "kuznetsova", "--name", "Анна", "--balance", "1000").exit_code == 0
    result = run("--admin", "wallet", "balance", "client:kuznetsova")
    assert "1 000.00 VRUB" in result.output
    assert run("--admin", "user", "add", "kuznetsova", "--name", "Анна").exit_code == 1
    assert run("--admin", "user", "add", "Bad_Id", "--name", "X").exit_code == 1


def test_topup_writes_ledger(run):
    assert run("--admin", "wallet", "topup", "petrova", "500").exit_code == 0
    result = run("--admin", "ledger", "--owner", "client:petrova")
    assert "Пополнение администратором" in result.output


def test_provider_add_then_agent_disable(run):
    assert run("--admin", "provider", "add", "new-co", "--name", "ООО «Новая»").exit_code == 0
    assert run("--admin", "agent", "disable", "retail-sportline").exit_code == 0
    assert "retail-sportline" not in run("--admin", "agent", "list").output
    assert "retail-sportline" in run("--admin", "agent", "list", "--all").output


def test_chat_stub(run):
    result = run("chat", "--user", "ivanov", input="/balance\nпривет\n/exit\n")
    assert result.exit_code == 0
    assert "1 500 000.00 VRUB" in result.output
    assert "этапе 2" in result.output


def test_chat_unknown_user(run):
    assert run("chat", "--user", "nobody").exit_code == 1


def test_check_llm_without_credentials(run):
    result = run("check-llm")
    assert result.exit_code == 1
    assert "GIGACHAT_CREDENTIALS" in result.output
