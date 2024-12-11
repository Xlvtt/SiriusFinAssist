#!/bin/bash

echo "Запускаем проект!"

start_backend() {
    echo "Запускаем Backend..."
    cd Backend
    source venv/Scripts/activate
    uvicorn main:app --reload &
    cd ..
}

start_frontend() {
    echo "✨ Запускаем Frontend..."
    cd Frontend
    source venv/Scripts/activate
    streamlit run frontend.py &
    cd ..
}

start_backend
sleep 3
start_frontend

echo "Проект запущен!"