FROM python:3.11-slim

WORKDIR /app

# Установим сначала необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install pyautogen python-dotenv gradio_client pandas scikit-learn openpyxl chromadb==0.5.* sentence_transformers duckdb

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]