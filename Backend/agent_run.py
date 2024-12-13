from dotenv import load_dotenv
load_dotenv()

import os
import re
import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

import chromadb
from chromadb.utils import embedding_functions

from autogen import AssistantAgent, UserProxyAgent, register_function

db_file = 'workdir/finance_data.duckdb'

CHROMADB_DIR = "vector_dbs/text2sql_few-shot"
COLLECTION_NAME = "text2sql_few-shot_examples"
chromadb_client = chromadb.PersistentClient(path=CHROMADB_DIR)

model_path = "embedding_models/e5-large"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path)

def load_data(file_path):
    return pd.read_excel(file_path)

def store_embeddings(file_path):
    data = load_data(file_path)

    collection = chromadb_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    questions = data['question'].tolist()
    correct_sql = data['correct_sql'].tolist()
    ids = [str(i) for i in range(len(questions))]
    
    collection.add(
        documents=questions,
        metadatas=[{"sql": sql} for sql in correct_sql],
        ids=ids
    )

def delete_collection(collection_name):
    try:
        chromadb_client.delete_collection(name=collection_name)
        print(f"Коллекция '{collection_name}' успешно удалена.")
    except Exception as e:
        print(f"Ошибка при удалении коллекции '{collection_name}': {e}")

def get_few_shot_examples(query, top_k=5):
    collection = chromadb_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    query_embedding = embedding_function([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    few_shot_examples = []
    for idx in range(top_k):
        few_shot_examples.append({
            'document': results['documents'][0][idx],
            'sql': results['metadatas'][0][idx]['sql']
        })
    
    return few_shot_examples


def get_rows_with_columns(db_file: str, table_name: str, n: int = 5):
    conn = duckdb.connect(db_file)

    column_query = f"PRAGMA table_info({table_name})"
    columns = [row[1] for row in conn.execute(column_query).fetchall()]

    query = f"SELECT * FROM {table_name} LIMIT {n}"
    result = conn.execute(query).fetchall()

    column_names = " | ".join(columns)
    rows_text = "\n".join([" | ".join(map(str, row)) for row in result])

    result_text = f"Columns: {column_names}\n{rows_text}"

    conn.close()

    return result_text

def get_database_schema_as_text(db_path: str) -> str:
    connection = duckdb.connect(database=db_path)
    schema_text = []
    try:
        
        tables = connection.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            schema_text.append(f"Таблица: {table_name}")
            
            columns = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            for col in columns:
                schema_text.append(f"  - {col[1]}: {col[2]}")
            schema_text.append("")
    finally:
        connection.close()
    return "\n".join(schema_text)


def delete_table(database_path: str, table_name: str):
    try:
        conn = duckdb.connect(database_path)

        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        print(f"Таблица '{table_name}' успешно удалена (если существовала).")

    except Exception as e:
        print(f"Ошибка при удалении таблицы: {e}")

    finally:
        conn.close()


def get_database_description(query: str) -> str:
    # Схема
    schema_text = get_database_schema_as_text(db_file)

    # Few-shot
    few_shot_examples = 'Вот тебе примеры:\n'
    few_shot_text2sql_examples = get_few_shot_examples(query, top_k=3)
    for text2sql_example in few_shot_text2sql_examples:
        few_shot_examples += f'Вопрос: {text2sql_example["document"]}\nSQL: {text2sql_example["sql"]}\n'
    few_shot_examples += '\n'

    # Data samples
    table_name = 'operations_cte'
    data_samples = get_rows_with_columns(db_file, table_name)

    database_description = f'''Схема базы данных: 
{schema_text}
Описание таблиц и колонок:
{text_description}

Вот тебе пример данных из таблицы: {table_name}
{data_samples}

Примеры вопросов к базе данных и ответов в виде SQL-запросов:
{few_shot_examples}
'''
    return database_description

def execute_query(sql_query: str, table_name: str) -> str:
    db_path = 'workdir/finance_data.duckdb'

    try:
        conn = duckdb.connect(db_path)
        conn.execute(f"CREATE TABLE {table_name} AS ({sql_query})")
        conn.close()
        return f"Данные успешно сохранены в таблицу: {table_name}"
    except Exception as e:
        return f"Ошибка при выполнении запроса: {e}"
    

def fetch_table_data(table_name: str) -> str:
    database_path = 'workdir/finance_data.duckdb'
    try:
        # Connect to the DuckDB database
        conn = duckdb.connect(database_path)

        # Verify if the table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        if table_name not in table_names:
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

        # Fetch data from the specified table
        query = f"SELECT * FROM {table_name} LIMIT 10"
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]

        # Format the result as a text string
        text_result = "\t".join(columns) + "\n"  # Add column headers
        for row in result:
            text_result += "\t".join(map(str, row)) + "\n"

        return text_result
    except Exception as e:
        raise e
    finally:
        conn.close()

def is_end_conversation(msg: str) -> bool:
    if msg.get("content"):
        if "'Ответ:'" in msg["content"]:
            return False
        if 'Ответ:' in msg["content"]:
            return True
    return False

text_description = '''Таблица operations_cte: Содержит подробные записи о транзакциях, включая сумму, категорию, описание и статус операции.
amount_currency: Валюта операции, например, RUB или другая локальная валюта.
amount_value: Сумма операции в числовом формате.
brand: Название торговой марки или организации, связанной с операцией, например, 'Acer', 'Магнит'.
card: Последние четыре цифры карты, используемой для операции, например '*0117'.
category: Категория расхода, например, 'Фастфуд', 'Супермаркеты' или 'Переводы'.
description: Описание операции, например, название магазина или услуги.
mcc: Код категории продавца (MCC), указывающий на тип торговой точки.
op_time: Время проведения операции в формате временной метки.
op_type: Тип операции, 'Debit' (списание) или 'Credit' (зачисление).
status: Статус операции,'OK' или 'FAILED'.
'''


file_path = "few-shot_data/train_text2sql_data.xlsx"
# store_embeddings(file_path)


llm_config = {"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]}

assistant = AssistantAgent("assistant", llm_config=llm_config)

user_proxy = UserProxyAgent("user_proxy",
                            human_input_mode="NEVER",
                            code_execution_config={"work_dir": "workdir", "use_docker": False},
                            is_termination_msg=is_end_conversation
                            )

register_function(
    get_database_description,
    caller=assistant,
    executor=user_proxy,
    name="get_database_description",
    description="Получение информации о базе данных. В качестве параметров передаётся текстовый запрос с указанием вопроса к данным. Пример: 'Последние транзакции'"
)

register_function(
    execute_query,
    caller=assistant,
    executor=user_proxy,
    name="execute_query",
    description="Исполнение SQL запроса к базе данных. В качестве параметров передаётся SQL-запрос и таблица, в которую сохраняется результат (например для промежуточных вычислений). При составлении запроса не ставь ; в конце."
)

register_function(
    fetch_table_data,
    caller=assistant,
    executor=user_proxy,
    name="fetch_table_data",
    description="Извлечение 10 строк данных из определённой таблицы. Название таблицы указывается как аргумент."
)



def get_agent_answer(question: str) -> str:
    chatresult = user_proxy.initiate_chat(assistant,
                                            message=f'''Ответь на вопрос пользователя о его финансах: {question}
    Действуй по следующему плану:
    1. Посмотри информацию о базе данных
    2. Сделай промежуточные таблицы
    3. Сделай выбор нужных таблиц для ответа на вопрос
    Ответ дай подробно, указывая все необходимые данные.
    Формат ответа: 'Ответ:'
    ''')
    agent_answer = chatresult.summary.split('Ответ:')[-1].strip()
    return agent_answer


if __name__ == '__main__':
    question = 'На какую сумму был совершён последний перевод?'
    chatresult = user_proxy.initiate_chat(assistant,
                                            message=f'''Ответь на вопрос пользователя о его финансах: {question}
    Действуй по следующему плану:
    1. Посмотри информацию о базе данных
    2. Сделай промежуточные таблицы
    3. Сделай выбор нужных таблиц для ответа на вопрос
    Ответ дай подробно, указывая все необходимые данные.
    Формат ответа: 'Ответ:'
    ''')
    print(chatresult.summary.split('Ответ:')[-1].strip())