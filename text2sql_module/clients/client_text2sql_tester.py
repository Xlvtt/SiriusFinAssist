import re
import requests
import duckdb
import pandas as pd
from difflib import SequenceMatcher
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("text2sql_pipeline.log")  # Log to file
    ],
    encoding='utf-8'
)

def remove_sql_aliases(sql_query: str) -> str:
    """
    Удаляет все псевдонимы из SQL-запроса.

    :param sql_query: str, SQL-запрос
    :return: str, SQL-запрос без псевдонимов
    """
    # Регулярное выражение для поиска псевдонимов в SQL
    alias_pattern = re.compile(r"\bAS\s+[a-zA-Z_][a-zA-Z0-9_]*\b", re.IGNORECASE)

    # Удаляем псевдонимы с помощью регулярного выражения
    cleaned_query = alias_pattern.sub("", sql_query)

    # Удаляем лишние пробелы, оставшиеся после удаления псевдонимов
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

    return cleaned_query

class TextToSQLClient:
    def __init__(self, base_url: str):
        """
        Инициализация клиента.

        :param base_url: Базовый URL API сервиса
        """
        self.base_url = base_url

    def convert_query(self, query: str) -> str:
        """
        Отправляет текстовый запрос в сервис и получает SQL-запрос.

        :param query: Вопрос на естественном языке
        :return: Сгенерированный SQL-запрос
        :raises Exception: Если сервис возвращает ошибку
        """
        url = f"{self.base_url}/text2sql"
        payload = {"query": query}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Проверка на наличие HTTP ошибок
            return response.json()["sql"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при взаимодействии с API: {str(e)}")
        except KeyError:
            raise Exception("Ошибка в формате ответа от API.")

    def calculate_similarity(self, sql1, sql2):
        """Calculate similarity between two SQL strings."""
        similarity = SequenceMatcher(None, sql1, sql2).ratio()
        logging.info("Calculated similarity: %.2f", similarity)
        return similarity

    def run_text2sql_pipeline(self, file_path, text2sql_api_url, database_path, output_path):
        logging.info("Starting Text-to-SQL pipeline.")
        data = pd.read_excel(file_path)
        total_queries = len(data)
        logging.info("Loaded %d queries from file: %s", total_queries, file_path)

        correct_executions = 0
        similarities = []
        results = []

        for index, row in data.iterrows():
            question = row['question']
            expected_sql = row['correct_sql']
            logging.info("Processing question: %s", question)
            count = index + 1

            print(f"Query count - {count}")
            try:
                # Step 1: Convert question to SQL
                response = requests.post(f"{text2sql_api_url}/text2sql", json={"query": question})
                response.raise_for_status()
                generated_sql = response.json().get("sql")
                logging.info("Generated SQL: %s", generated_sql)

                if not generated_sql:
                    logging.warning("No SQL generated for question: %s", question)
                    results.append({
                        "question": question,
                        "expected_sql": expected_sql,
                        "generated_sql": None,
                        "similarity": 0,
                        "execution_match": False
                    })
                    continue

                # Step 2: Calculate similarity
                similarity = self.calculate_similarity(generated_sql, expected_sql)
                similarities.append(similarity)

                # Step 3: Execute and compare results
                with duckdb.connect(database_path) as conn:
                    logging.info("Executing SQL queries on database.")
                    # generated_sql = remove_sql_aliases(generated_sql)
                    # expected_sql = remove_sql_aliases(expected_sql)

                    generated_result_df = conn.execute(generated_sql).fetchdf()
                    expected_result_df = conn.execute(expected_sql).fetchdf()

                # Оставляем только те колонки, которые есть в expected_result_df
                common_columns = set(expected_result_df.columns) & set(generated_result_df.columns)
                generated_result_df = generated_result_df[list(common_columns)]

                # Compare columns
                if set(generated_result_df.columns) != set(expected_result_df.columns):
                    logging.warning("Column mismatch for question: %s", question)
                    results.append({
                        "question": question,
                        "expected_sql": expected_sql,
                        "generated_sql": generated_sql,
                        "similarity": similarity,
                        "execution_match": False,
                        "error": "Column mismatch"
                    })
                    continue

                # Compare rows
                if not generated_result_df.sort_values(by=generated_result_df.columns.tolist()).reset_index(
                        drop=True).equals(
                        expected_result_df.sort_values(by=expected_result_df.columns.tolist()).reset_index(drop=True)
                ):
                    logging.warning("Row mismatch for question: %s", question)
                    results.append({
                        "question": question,
                        "expected_sql": expected_sql,
                        "generated_sql": generated_sql,
                        "similarity": similarity,
                        "execution_match": False,
                        "error": "Row mismatch"
                    })
                    continue

                # If both columns and rows match, the execution is correct
                correct_executions += 1
                results.append({
                    "question": question,
                    "expected_sql": expected_sql,
                    "generated_sql": generated_sql,
                    "similarity": similarity,
                    "execution_match": True
                })

            except Exception as e:
                logging.error("Error processing question: %s\nError: %s", question, e)
                results.append({
                    "question": question,
                    "expected_sql": expected_sql,
                    "generated_sql": None,
                    "similarity": 0,
                    "execution_match": False,
                    "error": str(e)
                })

        # Step 4: Metrics
        execution_accuracy = correct_executions / total_queries
        average_similarity = sum(similarities) / len(similarities) if similarities else 0

        logging.info("Execution Accuracy (EX): %.2f%%", execution_accuracy * 100)
        logging.info("Average SQL Similarity: %.2f%%", average_similarity * 100)

        # Save results to a file
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False)
        logging.info("Results saved to file: %s", output_path)

        print(f"Execution Accuracy (EX): {execution_accuracy:.2%}")
        print(f"Average SQL Similarity: {average_similarity:.2%}")


if __name__ == "__main__":
    BASE_URL = "http://127.0.0.1:8000"

    client = TextToSQLClient(base_url=BASE_URL)
    file_path = "../data/test_text2sql_data.xlsx"
    database_path = "../data/finance_data.duckdb"
    output_path = "../data/text2sql_results.xlsx"

    client.run_text2sql_pipeline(file_path,  BASE_URL, database_path, output_path)