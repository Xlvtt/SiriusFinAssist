import pandas as pd
import requests

# Путь к Excel-файлу
file_path = "validation_data/Vladimir_text_questions.xlsx"
TEST_NAME = 'pro_agent_gpt-4o'
# TEST_NAME = 'pro_agent_gpt-3.5-turbo'

# URL API
api_url = "http://localhost:8001/chat"

# Чтение Excel-файла в DataFrame
df = pd.read_excel(file_path)

# Проверка, что столбец 'question' существует
if 'question' not in df.columns:
    raise ValueError("Столбец 'question' отсутствует в Excel-файле")

# Функция для получения ответа от API
def get_agent_answer(message):
    print(message)
    try:
        response = requests.get(api_url, params={"message": message})
        response_data = response.json()
        if response.status_code == 200 and response_data.get("success"):
            print("Success")
            return response_data.get("message")
        else:
            print("Error API")
            return f"Ошибка: {response_data.get('error', 'Неизвестная ошибка')}"
    except Exception as e:
        print("Error")
        return f"Исключение: {str(e)}"

# Создание нового столбца 'agent_answer'
df['agent_answer'] = df['question'].apply(get_agent_answer)

# Сохранение результата в новый Excel-файл
output_file_path = f"validation_data/{TEST_NAME}_Vladimir_text_questions.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Результаты сохранены в файл: {output_file_path}")
