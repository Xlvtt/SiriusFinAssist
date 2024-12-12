from langchain_community.tools import TavilySearchResults
from text2sql_module.clients.client_text2data import Text2SQLClient
from langchain_community.tools import tool
from typing import Annotated

text2sql_client = Text2SQLClient(base_url="http://backend:8000")


@tool
def text2sql_tool(
        query: Annotated[str, "Запрос пользователя к базе данных с транзакциями"] # TODO prompt
):
    """
    Инструмент имеет доступ ко всем транзакциям пользователя, задавшего вопрос. Он отвечает на запрос, опираясь на данные из базы.

    Используй этот инструмент, если для ответа на запрос тебе необходима личная информация о пользователе:
    например, его образ жизни, категории трат, суммы расходов, доходов, увлечения и тд.
    """
    try:
        result = text2sql_client.execute_query(query)
        print("Результат запроса:")
        print(result)
    except RuntimeError as e:
        print(f"Произошла ошибка: {e}")


search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    description="""
    Инструмент ищет информацию в интернете.
    
    Используй инструмент, если для ответа на запрос достаточно общедоступной информации из интернета
    """
)
print(search_tool.description)
tools_list = [text2sql_tool, search_tool]
