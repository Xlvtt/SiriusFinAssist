from dotenv import load_dotenv
load_dotenv()

from graph import assistant

if __name__ == "__main__":
    query = input("Задайте свой вопрос:")
    for step in assistant.stream({"input": query, "steps_limit": 10}):
        for key, value in step.items():
            print(value)
