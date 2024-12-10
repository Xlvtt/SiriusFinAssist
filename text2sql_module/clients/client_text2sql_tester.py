import requests

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

# Пример использования клиента
if __name__ == "__main__":
    # Замените на ваш URL, если сервер работает на другом хосте или порту
    BASE_URL = "http://127.0.0.1:8000"

    client = TextToSQLClient(base_url=BASE_URL)

    # Пример вопроса
    questions = [
        "Какие товары были проданы в январе?",
        "Сколько пользователей зарегистрировалось в 2024 году?",
        "Покажи все записи из таблицы users.",
    ]

    for question in questions:
        try:
            sql_query = client.convert_query(query=question)
            print(f"Вопрос: {question}")
            print(f"SQL: {sql_query}\n")
        except Exception as e:
            print(f"Ошибка обработки вопроса '{question}': {e}")
