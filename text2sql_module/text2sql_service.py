from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests
import duckdb
from text2sql_realization import text2sql_function

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    sql: str

def text_to_sql(query: str) -> str:
    return text2sql_function(query)

@app.post("/text2sql", response_model=QueryResponse)
def convert_query(request: QueryRequest):
    try:
        sql_query = text_to_sql(request.query)
        print(sql_query)
        return QueryResponse(sql=sql_query)
    except Exception as e:
        print('ОШИБКА')
        print(e)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


class QueryRequest(BaseModel):
    query: str

class DataFrameResponse(BaseModel):
    data: str

DATABASE_PATH = "data/finance_data.duckdb"
TEXT2SQL_API_URL = "http://127.0.0.1:8000/text2sql"

@app.post("/text2data", response_model=DataFrameResponse)
def execute_query(request: QueryRequest):
    try:
        # Отправляем запрос в сервис text2sql
        response = requests.post(TEXT2SQL_API_URL, json={"query": request.query})
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Ошибка вызова text2sql API")

        # Получаем SQL-запрос
        sql_query = response.json().get("sql")
        if not sql_query:
            raise HTTPException(status_code=500, detail="Ответ text2sql API не содержит SQL-запроса")

        # Выполняем запрос к базе данных
        try:
            conn = duckdb.connect(DATABASE_PATH)
            result_df = conn.execute(sql_query).fetchdf()
            conn.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка выполнения SQL-запроса: {str(e)}")

        # Преобразуем DataFrame в текст (CSV)
        data_text = result_df.to_csv(index=False)
        return DataFrameResponse(data=data_text)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Общая ошибка: {str(e)}")
