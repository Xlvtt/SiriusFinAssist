import re
import pandas as pd

import duckdb
import chromadb
from chromadb.utils import embedding_functions
from gradio_client import Client

CHROMADB_DIR = "vector_dbs/text2sql_few-shot"
chromadb_client = chromadb.PersistentClient(path=CHROMADB_DIR)
COLLECTION_NAME = "text2sql_few-shot_examples"

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


def get_few_shot_examples(query, top_k=5):
    """
    Получение few-shot примеров из базы данных для данного текстового запроса.

    :param query: Текст запроса, для которого нужно найти похожие примеры.
    :param top_k: Количество похожих примеров, которые нужно вернуть.
    :return: Список найденных примеров (вопрос и SQL).
    """

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


def get_model_answer(system_prompt: str, user_prompt: str) -> str:
    result = client.predict(
                    query=user_prompt,
                    history=[],
                    system=system_prompt,
                    radio="Coder-7B",
                    api_name="/model_chat"
            )
    text_answer = result[1][0][-1]['text']
    return text_answer


def extract_sql_blocks(text):
    pattern = r"```sql\s+([\s\S]*?)```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [match.strip() for match in matches]


# Init vector db
file_path = "data/train_text2sql_data.xlsx"
store_embeddings(file_path)

# Init client for LLM
client = Client('Qwen/Qwen2.5')

# Get schema from db
db_file = 'data/finance_data.duckdb'
schema_text = get_database_schema_as_text(db_file)

# Get data examples from db
table_name = 'operations_cte'
data_samples = get_rows_with_columns(db_file, table_name)

text_description = '''Таблица accounts: Содержит информацию о банковских счетах пользователя, включая их идентификатор, тип, валюту и текущий баланс.
id: Уникальный идентификатор аккаунта в текстовом формате.
name: Название аккаунта, например, имя пользователя или название счета.
type: Тип аккаунта, например, дебетовый, кредитный или инвестиционный.
currency: Валюта счета, например, RUB, USD.
balance: Текущий баланс счета в числовом формате.

Таблица operations_cte: Содержит подробные записи о транзакциях, включая сумму, категорию, описание и статус операции.
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


def text2sql_function(question: str) -> str:
    few_shot_examples = 'Вот тебе примеры:\n'
    few_shot_text2sql_examples = get_few_shot_examples(question)
    for text2sql_example in few_shot_text2sql_examples:
        few_shot_examples += f'Вопрос: {text2sql_example["document"]} SQL: {text2sql_example["sql"]}\n'

    few_shot_examples += '\n'

    system_prompt = 'Ты профессиональный аналитик SQL.'
    user_prompt = f'''Тебе дана следующая схема базы данных: 
    {schema_text}

    Описание таблиц и колонок:
    {text_description}

    Вот тебе пример данных из таблицы: {table_name}
    {data_samples}

    {few_shot_examples}
    Тебе необходимо написать SQL запрос, который достаёт все нужные данные для ответа на следующий вопрос: {question}.
    В качестве ответа напиши только SQL.'''
    print(user_prompt)

    model_answer = get_model_answer(system_prompt, user_prompt)

    response_sql = extract_sql_blocks(model_answer)[0]

    return response_sql

print('Init text2sql realization')
if __name__ == '__main__':
    question = 'В каком магазине я купила последний раз косметику?'
    print(text2sql_function(question))
    print('Success')