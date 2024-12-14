from dotenv import load_dotenv
load_dotenv()

import os
import re
import duckdb
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

import chromadb
from chromadb.utils import embedding_functions

from autogen import AssistantAgent, UserProxyAgent, register_function

db_file = 'bank_data_user1.duckdb'
model_name = 'gpt-4o'

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

CATEGORY_CHROMADB_DIR = "vector_dbs/data_samples"
CATEGORY_COLLECTION_NAME = "category_data_samples"

category_chromadb_client = chromadb.PersistentClient(path=CATEGORY_CHROMADB_DIR)

category_collection = chromadb_client.get_or_create_collection(
        name=CATEGORY_COLLECTION_NAME,
        embedding_function=embedding_function
    )

conn = duckdb.connect(db_file)
table_name = 'operations_cte'
unique_categories_query = f"SELECT DISTINCT category FROM {table_name}"
categories = [row[0] for row in conn.execute(unique_categories_query).fetchall() if row[0]]
conn.close()

ids = [str(i) for i in range(len(categories))]
category_collection.add(documents=categories, ids=ids)

def get_similar_categories(db_file: str, user_query: str, top_n: int = 3) -> str:
    table_name = 'operations_cte'
    conn = duckdb.connect(db_file)

    user_embedding = embedding_function([user_query])[0]

    results = category_collection.query(query_embeddings=[user_embedding], n_results=top_n)
    top_categories = results['documents'][0]

    example_texts = []
    for category in top_categories:
        examples_query = f"SELECT * FROM {table_name} WHERE category = ? LIMIT 1"
        examples = conn.execute(examples_query, [category]).fetchall()

        example_rows = "\n".join([" | ".join(map(str, row)) for row in examples])
        example_texts.append(f"Category: {category}\n{example_rows}")

    result_text = "\n\n".join(example_texts)

    conn.close()

    return result_text


DESCRIPTION_COLLECTION_NAME = "description_data_samples"

description_collection = chromadb_client.get_or_create_collection(
        name=DESCRIPTION_COLLECTION_NAME,
        embedding_function=embedding_function
    )

conn = duckdb.connect(db_file)
table_name = 'operations_cte'
unique_description_query = f"SELECT DISTINCT description FROM {table_name}"
descriptions = [row[0] for row in conn.execute(unique_description_query).fetchall() if row[0]]
conn.close()

ids = [str(i) for i in range(len(descriptions))]
description_collection.add(documents=descriptions, ids=ids)

def get_similar_descriptions(db_file: str, user_query: str, top_n: int = 3) -> str:
    table_name = 'operations_cte'
    conn = duckdb.connect(db_file)

    user_embedding = embedding_function([user_query])[0]

    results = description_collection.query(query_embeddings=[user_embedding], n_results=top_n)
    top_descriptions = results['documents'][0]

    example_texts = []
    for category in top_descriptions:
        examples_query = f"SELECT * FROM {table_name} WHERE description = ? LIMIT 1"
        examples = conn.execute(examples_query, [category]).fetchall()

        example_rows = "\n".join([" | ".join(map(str, row)) for row in examples])
        example_texts.append(f"Description: {category}\n{example_rows}")

    result_text = "\n\n".join(example_texts)

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

# For storage
file_path = "few-shot_data/train_text2sql_data.xlsx"
store_embeddings(file_path)

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
    data_samples = get_rows_with_columns(db_file, table_name, n= 3)

    categories_data_samples = get_similar_categories(db_file, query)
    descriptions_data_samples = get_similar_descriptions(db_file, query)

    database_description = f'''Схема базы данных в PostgreSQL: 
{schema_text}
Описание таблиц и колонок:
{text_description}

Вот тебе пример данных из таблицы: {table_name}
{data_samples}

{categories_data_samples}
{descriptions_data_samples}

Примеры вопросов к базе данных и ответов в виде SQL-запросов:
{few_shot_examples}
'''
    return database_description


def execute_query(sql_query: str, table_name: str) -> str:
    db_path = db_file
    # Удаляем ; в конце запроса, если она есть
    if sql_query.strip().endswith(';'):
        sql_query = sql_query.strip()[:-1]

    try:
        conn = duckdb.connect(db_path)
        # Выполняем запрос и создаём таблицу
        conn.execute(f"CREATE TABLE {table_name} AS ({sql_query})")
        
        # Извлекаем имена колонок и 3 строки из таблицы
        columns_query = f"PRAGMA table_info({table_name})"
        columns = [row[1] for row in conn.execute(columns_query).fetchall()]
        
        examples_query = f"SELECT * FROM {table_name} LIMIT 3"
        examples = conn.execute(examples_query).fetchall()
        
        conn.close()
        
        # Формируем текст с примерами
        examples_text = "\nПримеры 3 строк (" + ", ".join(columns) + "):\n"
        for row in examples:
            examples_text += "  " + str(row) + "\n"

        return f"Данные успешно сохранены в таблицу: {table_name}{examples_text}"
    except Exception as e:
        return f"Ошибка при выполнении запроса: {e}"

def delete_table(database_path: str, table_name: str):
    try:
        conn = duckdb.connect(database_path)

        conn.execute(f"DROP TABLE IF EXISTS {table_name};")
        print(f"Таблица '{table_name}' успешно удалена (если существовала).")

    except Exception as e:
        print(f"Ошибка при удалении таблицы: {e}")

    finally:
        conn.close()


def fetch_table_data(table_name: str) -> str:
    database_path = db_file
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

def list_and_clean_tables(database_path: str, base_tables: list):
    try:
        connection = duckdb.connect(database_path)
        
        tables = connection.execute("SHOW TABLES").fetchall()

        print("Список таблиц в базе данных:")
        for table in tables:
            print(table[0])
        
        for table in tables:
            if table[0] not in base_tables:
                connection.execute(f"DROP TABLE IF EXISTS {table[0]}")
                print(f"Таблица '{table[0]}' удалена.")
                
    except Exception as e:
        print(f"Ошибка: {e}")
    
    finally:
        connection.close()

def is_end_conversation(msg: str) -> bool:
    if msg.get("content"):
        if "'Ответ:'" in msg["content"]:
            return False
        if 'Ответ:' in msg["content"]:
            return True
        if 'TERMINATE' in msg["content"]:
            return True
    return False

def get_current_date():
    current_date = datetime.now()
    return current_date.strftime("%d день, %m месяц, %Y год")

word_info = f'''Дополнительная инфрмация: текущая дата {get_current_date()}'''

question_answer_examples = '''Пример:
Вопрос: Сколько я потратил на отели в 2024?
Ответ: На отели вами было потрачено <N> рублей.
Вопрос: Когда был последний перевод?
Ответ: Последний перевод был совершен на сумму <N> RUB и произошел 10 декабря 2024 года в 23:36.
Вопрос: Сколько я потратила в Пятёрочке?
Ответ: Ты потратила <N> RUB в Пятёрочке'''

additional_instructions = '''Если речь идёт о расходах, то не используй минус при ответе.
Если из базы данных возвращается данных, то отвечай, что не было найдено подходящих данных.'''

llm_config = {"config_list": [{"model": model_name, "api_key": os.environ["OPENAI_API_KEY"]}]}

assistant = AssistantAgent("assistant", llm_config=llm_config, 
                           system_message='''Ты профессиональный финансовый ассистент. Ты создан отвечать на вопросы пользователей об их финансах, быть полезным.
Действуй по следующему плану:
1. Посмотри информацию о базе данных
2. Сделай промежуточные таблицы
3. Сделай выбор нужных таблиц для ответа на вопрос и сагрегируй результат

{additional_instructions}

{question_answer_examples}

Ответ давай подробно, указывая все необходимые данные.
Если в ответе есть число денег, то укажи валюту.
Если в ответе есть различные операции, то укажи при наличии информацию когда и в каком магазине они были совершены.
Если чего-то не знаешь, то не пиши этого.
{word_info}''')

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
    description="Получение информации о базе данных. В качестве параметров передаётся текстовый запрос с указанием вопроса к данным. Пример: 'Последние транзакции', 'Покупки в магазине Магнит', 'Траты в Вкусно и Точка'"
)

register_function(
    execute_query,
    caller=assistant,
    executor=user_proxy,
    name="execute_query",
    description="Исполнение SQL запроса к базе данных PostgreSQL. При составлении SQL выбирай как можно больше столбцов, чтобы дать качественее ответ. В качестве параметров передаётся SQL-запрос и таблица, в которую сохраняется результат (используй для промежуточных вычислений)."
)

register_function(
    fetch_table_data,
    caller=assistant,
    executor=user_proxy,
    name="fetch_table_data",
    description="Извлечение 10 строк данных из определённой таблицы. Название таблицы указывается как аргумент."
)

BASE_TABLES = ['accounts', 'operations_cte']

log_dir = "chat_logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

def get_agent_answer(question: str) -> str:
    logging.info('question: %s', question)

    try:
        chatresult = user_proxy.initiate_chat(assistant,
                                                message=f'''Тебе дан следующий диалог с пользователем: {question}
Для ответа используй извлечение данных из какой-либо таблицы из базы данных.
В ответе упомяни промежуточные вычисления, если такие были.
Формат ответа: 'Ответ:'
В конце ответа пиши TERMINATE.''')
        agent_answer = chatresult.summary.split('Ответ:')[-1].strip()
        logging.info('agent_answer: %s', agent_answer)
        list_and_clean_tables(db_file, BASE_TABLES)
        return agent_answer.rstrip('.')
    except Exception as e:
        logging.error('Ошибка: %s', str(e))
        return 'Извините, при вашем обращении произошла ошибка. Уже чиним это 😊'


print('Agent is READY!')

if __name__ == '__main__':
    question = 'На какую сумму был совершён последний перевод?'
    print(get_agent_answer(question))