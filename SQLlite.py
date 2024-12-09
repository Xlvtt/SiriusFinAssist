import yaml
import sqlite3
import os


# Функция для загрузки данных из YAML файла
def load_yaml(file_path):
    # Проверяем, существует ли файл по указанному пути
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        return None  # Возвращаем None, если файл не найден
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


# Функция для создания базы данных SQLite
def create_db(db_name):
    conn = sqlite3.connect(db_name)
    return conn


# Функция для создания таблиц на основе структуры данных в YAML
def create_tables(conn, table_data):
    cursor = conn.cursor()

    for table_name, table_info in table_data.items():
        # Строим SQL запрос для создания таблицы
        columns = table_info.get('columns', [])
        column_defs = ', '.join([f"{col[0]} {col[1]}" for col in columns])

        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs});"
        cursor.execute(create_table_query)

        # Вставляем данные в таблицу
        data = table_info.get('data', [])
        if data:
            placeholders = ', '.join(['?'] * len(columns))
            insert_query = f"INSERT INTO {table_name} ({', '.join([col[0] for col in columns])}) VALUES ({placeholders})"
            cursor.executemany(insert_query, data)

    conn.commit()


# Главная функция для обработки нескольких YAML файлов
def process_yaml_files(yaml_files, db_name):
    conn = create_db(db_name)

    for yaml_file in yaml_files:
        print(f"Processing file: {yaml_file}")
        table_data = load_yaml(yaml_file)
        if table_data is None:
            continue  # Пропускаем файл, если он не был найден

        table_data = table_data.get('tables', {})
        create_tables(conn, table_data)

    conn.close()


if __name__ == "__main__":
    # Список файлов YAML для обработки
    yaml_files = [
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data001.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data002.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data003.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data004.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data005.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data006.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data007.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data008.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data009.yaml",
        r"C:\Users\User\Desktop\SQLlite\filesyaml\user_data010.yaml"
    ]

    # Имя базы данных
    db_name = 'my_database.db'

    # Обработка всех файлов YAML и создание базы данных
    process_yaml_files(yaml_files, db_name)
    print(f"Database '{db_name}' created and populated successfully.")
