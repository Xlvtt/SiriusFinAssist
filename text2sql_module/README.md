# SiriusFinAssist

Здесь будет отдельный text2sql module:

Input:
- query: str - текстовый запрос пользователя/агента

Output:
- answer: str - текстовый ответ на основе данных


## Запуск

```commandline
uvicorn text2sql_service:app --reload
```