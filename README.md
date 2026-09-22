# Agentic-Economics

Демо агентной экономики: клиент ставит задачу в терминальном чате, оркестратор разбивает её на подзадачи, агенты компаний-провайдеров соревнуются за них в аукционе, расчёты идут в виртуальной валюте. Работает на тестовых данных, LLM — GigaChat.

Техническое задание: [docs/TZ.md](docs/TZ.md).

## Статус

| Этап | Состояние |
|---|---|
| 1. Каркас: конфиг, GigaChat, БД, seed, CLI | готов |
| 2. Оркестратор: диалог и декомпозиция в `task-N.md` | — |
| 3. Аукцион | — |
| 4. Экономика: эскроу, комиссия | — |
| 5. Полный цикл | — |

## Установка

Нужен Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # и впишите GIGACHAT_CREDENTIALS
python -m agentic_economics check-llm
```

`.env` не попадает в git. Ключ авторизации берётся в личном кабинете GigaChat API (developers.sber.ru). Если запросы падают с ошибкой TLS, установите корневой сертификат Минцифры или временно укажите `GIGACHAT_VERIFY_SSL_CERTS=false`.

Если OAuth-эндпоинт `ngw.devices.sberbank.ru:9443` недоступен (например, прокси пропускает только порт 443), получите токен доступа на другой машине и укажите его в `GIGACHAT_ACCESS_TOKEN` — тогда запрос за токеном не делается. Токен живёт 30 минут.

## Запуск

```bash
python -m agentic_economics --admin reset-demo -y   # пересоздать демо-базу
python -m agentic_economics chat --user ivanov       # чат клиента
```

Демо-клиенты: `ivanov`, `petrova`, `sidorov` (по 1 500 000 VRUB).

## Команды администратора

Все команды ниже требуют флага `--admin` (пароля в демо нет):

```bash
python -m agentic_economics --admin agent list [--category hotels] [--all]
python -m agentic_economics --admin agent show hotels-sunrise
python -m agentic_economics --admin agent add path/to/agent.yaml
python -m agentic_economics --admin agent disable|enable <id>
python -m agentic_economics --admin agent reload

python -m agentic_economics --admin provider list
python -m agentic_economics --admin provider add <id> --name "ООО «…»"

python -m agentic_economics --admin user list
python -m agentic_economics --admin user add <id> --name "Имя" [--balance 500000]

python -m agentic_economics --admin wallet balance client:ivanov
python -m agentic_economics --admin wallet topup ivanov 100000
python -m agentic_economics --admin ledger [--owner sky-lines]
```

## Структура

```
agentic_economics/   код: cli, config, llm, storage, agents, wallet, seed
agents/              YAML-описания агентов-исполнителей (источник истины)
data/seed.yaml       провайдеры и клиенты для reset-demo
config.yaml          настройки (без секретов)
docs/TZ.md           техническое задание
tests/               pytest
```

## Тесты

```bash
pytest
```
