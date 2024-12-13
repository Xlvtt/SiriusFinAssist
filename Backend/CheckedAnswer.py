from openpyxl import load_workbook
import os
import agent_run as bot
import random as Random

bot.db_file = 'finance_data_user1.duckdb'

class CheckedAnswerWB:

    def __init__(self, nameFile: str = None):
        self.filename = nameFile
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(os.path.dirname(self.current_dir), nameFile)
        if not os.path.exists(self.file_path):
            print(f"Файл {nameFile} не найден!")
            return
        
        self.WB = load_workbook(self.file_path)
        self._sheet = self.WB['Лист1']

    def ForeachWBAllRequest(self, Function, FunctionIter):
        i = 2
        while True:
            cell = self._sheet[f'A{i}']
            if cell.value is None or cell.value == '':
                break
            sellCheck = self._sheet[f'C{i}']
            if sellCheck.value is None or sellCheck.value == '':
                self._sheet[f'C{i}'] = Function(cell.value)
            i = FunctionIter(i)

    def save(self):
        self.WB.save(self.file_path)



checkWB = CheckedAnswerWB('Anna_text_questions.xlsx')

def test(text) -> str:
    print(str(text))
    return 'test'

def iterator(i):
    return i + 1


checkWB.ForeachWBAllRequest(bot.get_agent_answer, iterator)

checkWB.save()


    

    
    

    
    
    