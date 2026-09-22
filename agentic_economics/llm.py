"""Фабрика LLM GigaChat."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_economics.config import Settings


class LLMNotConfigured(RuntimeError):
    pass


def get_llm(settings: Settings) -> BaseChatModel:
    from langchain_gigachat import GigaChat

    secrets = settings.gigachat
    if secrets.credentials is None or not secrets.credentials.get_secret_value():
        raise LLMNotConfigured(
            "Не задан GIGACHAT_CREDENTIALS. Скопируйте .env.example в .env и укажите ключ авторизации."
        )
    return GigaChat(
        credentials=secrets.credentials.get_secret_value(),
        scope=secrets.scope,
        verify_ssl_certs=secrets.verify_ssl_certs,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )
