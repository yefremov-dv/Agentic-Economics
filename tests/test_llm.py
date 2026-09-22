import pytest

from agentic_economics.llm import LLMNotConfigured, get_llm


def test_get_llm_requires_credentials(settings):
    with pytest.raises(LLMNotConfigured):
        get_llm(settings)


def test_get_llm_builds_gigachat(monkeypatch, settings):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "secret-key")
    llm = get_llm(settings)
    assert type(llm).__name__ == "GigaChat"
    assert llm.model == settings.llm.model
    assert llm.scope == "GIGACHAT_API_PERS"
