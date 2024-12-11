
from gradio_client import Client

class Bot:
    def __init__(self):
        self.client = Client("Qwen/Qwen2.5")

    def get(self, message: str):
        system_prompt = '''Ты профессиональный финансовый планнировщик, который быстро реагирует. 
        тебе дан план ответа на вопрос . 
        Пример: 1) оценить платежеспособность пользователя по его транзакциям 

        тебе дана история пользователя о его финансовых транзакциях, тебе нужно проанализировать его операции и на их основе дать консультацию.'''
        user_prompt = 'Тебе необходимо составить пошаговый план для решения вопроса '
        params = 'Годовой доход 1200000р, траты в месяц на жилье 50000р, затраты на все остальное дают возможность откладывать средства по 5000 в месяц'

        user_prompt = f'Дана информация: {params} {user_prompt} {message}'

        result = self.client.predict(
                        query=user_prompt,
                        history=[],
                        system=system_prompt,
                        radio="14B",
                        api_name="/model_chat"
                )
        return result[1][0][-1]['text']

