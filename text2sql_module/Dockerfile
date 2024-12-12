FROM python:3.11-slim

WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Открываем порт 8000
EXPOSE 8000

# Запускаем FastAPI приложение
CMD ["uvicorn", "text2sql_service:app", "--host", "0.0.0.0", "--port", "8000"]