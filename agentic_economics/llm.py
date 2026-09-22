"""Фабрика LLM GigaChat."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_economics.config import Settings


class LLMNotConfigured(RuntimeError):
    pass


def get_llm(settings: Settings) -> BaseChatModel:
    from langchain_gigachat import GigaChat

    secrets = settings.gigachat
    credentials = secrets.credentials.get_secret_value() if secrets.credentials else ""
    access_token = secrets.access_token.get_secret_value() if secrets.access_token else ""
    if not credentials and not access_token:
        raise LLMNotConfigured(
            "Не задан GIGACHAT_CREDENTIALS. Скопируйте .env.example в .env и укажите ключ авторизации."
        )
    return GigaChat(
        credentials=credentials or None,
        access_token=access_token or None,
        scope=secrets.scope,
        verify_ssl_certs=secrets.verify_ssl_certs,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )
