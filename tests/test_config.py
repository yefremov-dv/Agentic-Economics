from agentic_economics.config import load_settings
from agentic_economics.money import fmt, to_minor


def test_project_config_matches_spec():
    settings = load_settings("config.yaml")
    assert settings.wallet.platform_fee == 0.05  # FR-23
    assert settings.wallet.default_client_balance == 1_500_000
    assert settings.auction.bid_timeout == 30  # FR-13
    assert settings.auction.max_candidates == 5  # FR-10
    assert settings.path("db").is_absolute()


def test_defaults_without_config_file(tmp_path):
    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.path("agents_dir") == tmp_path / "agents"


def test_gigachat_secrets_from_env(monkeypatch, settings):
    assert settings.gigachat.credentials is None
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "secret-key")
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "false")
    secrets = settings.gigachat
    assert secrets.credentials.get_secret_value() == "secret-key"
    assert "secret-key" not in repr(secrets)
    assert secrets.verify_ssl_certs is False
    assert secrets.scope == "GIGACHAT_API_PERS"


def test_money():
    assert to_minor(1000.5) == 100050
    assert to_minor("0.015") == 2
    assert fmt(150000000) == "1 500 000.00 VRUB"
    assert fmt(-5) == "-0.05 VRUB"
