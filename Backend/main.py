from dotenv import load_dotenv
load_dotenv()

from graph import assistant
from tools import text2sql_client
# from deep_translator import GoogleTranslator

# вот вернул он кучу данных. Их же просуммировать как-то надо.

if __name__ == "__main__":
    query = input("Задайте свой вопрос:")
    # print(text2sql_client.execute_query(query))

    # Retrieve all income transactions for the may

    for step in assistant.stream({"input": query}):
        for key, value in step.items():
            print(value)

