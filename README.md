# Agentic-Economics

Демо агентной экономики: клиент ставит задачу в терминальном чате, оркестратор разбивает её на подзадачи, агенты компаний-провайдеров соревнуются за них в аукционе, расчёты идут в виртуальной валюте. Работает на тестовых данных, LLM — GigaChat.

Техническое задание: [docs/TZ.md](docs/TZ.md).

## Статус

| Этап | Состояние |
|---|---|
| 1. Каркас: конфиг, GigaChat, БД, seed, CLI | готов |
| 2. Оркестратор: диалог и декомпозиция в `task-N.md` | готов |
| 3. Аукцион: агенты-исполнители, ставки, победитель | готов |
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

Модель задаётся в `config.yaml` (`llm.model`, по умолчанию `GigaChat-2`). Для оркестратора отдельно — `orchestrator.model` (по умолчанию `GigaChat-2-Max`: на младших моделях декомпозиция заметно хуже). Доступны также `GigaChat-2-Pro`, `GigaChat-2-Max`, `GigaChat-3-Lightning` и другие — список зависит от scope.

## Запуск

```bash
python -m agentic_economics --admin reset-demo -y   # пересоздать демо-базу
python -m agentic_economics chat --user ivanov       # чат клиента
```

Демо-клиенты: `ivanov`, `petrova`, `sidorov` (по 1 500 000 VRUB).

В чате опишите задачу свободным текстом. Оркестратор задаст уточняющие вопросы (не более `orchestrator.max_clarification_rounds` раз, затем достроит план с допущениями), разобьёт задачу на подзадачи и сохранит их в `tasks/<session>/task-N.md`. Рекомендованные подзадачи (например, страховка) создаются выключенными. Новое сообщение после готового плана меняет требования — план строится заново.

Сразу после декомпозиции по каждой подзадаче проходит аукцион: один раунд закрытых ставок. Кандидаты — активные агенты категории (не более `auction.max_candidates`, при избытке их ранжирует LLM). Каждый агент ищет варианты в моковом каталоге своей компании (`data/mock/*.json`, инструмент LangChain `search_hotels` и т. п.), выбирает позиции и количество, а цену считает код — по каталогу и наценке из YAML агента, поэтому модель не может её выдумать. Ставки собираются параллельно (`auction.max_parallel`), не ответивший за `auction.bid_timeout` секунд агент выбывает. Победитель — по взвешенной оценке `score = w_price·(1 − price_norm) + w_conf·confidence + w_rating·rating_norm` (веса в `auction.weights`), при равенстве — более ранняя ставка. Если сумма лучших ставок превышает бюджет, оркестратор предупреждает и называет самую дешёвую комбинацию.

Команды чата: `/plan`, `/bids [task-N]`, `/history`, `/balance`, `/cancel`, `/help`, `/exit`. Продолжить сессию: `chat --user ivanov --session <id>`.

После обновления программы, если схема базы изменилась, CLI попросит пересоздать базу: `reset-demo -y`.

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

python -m agentic_economics --admin auction show <task_id>    # все ставки по подзадаче
```

## Структура

```
agentic_economics/   код: cli, config, llm, orchestrator, auction, agents, storage, wallet, seed
agents/              YAML-описания агентов-исполнителей (источник истины)
data/seed.yaml       провайдеры и клиенты для reset-demo
data/mock/           моковые каталоги: рейсы, отели, трансферы, товары, страховки
config.yaml          настройки (без секретов)
docs/TZ.md           техническое задание
tests/               pytest
```

## Тесты

```bash
pytest
```

Покрытие логики аукциона и кошельков (NFR-6):

```bash
pytest --cov=agentic_economics.auction --cov=agentic_economics.wallet --cov-report=term-missing
```
