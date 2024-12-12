from langchain_community.tools import TavilySearchResults
from text2sql_module.clients.client_text2data import Text2SQLClient
from langchain_community.tools import tool
from typing import Annotated

text2sql_client = Text2SQLClient(base_url="http://127.0.0.1:8000")


@tool
def text2sql_tool(
        query: Annotated[str, "Запрос пользователя к базе данных с транзакциями"]  # TODO prompt
):
    """
    The tool has access to all transactions of the user who asked the question. He responds to the request based on data from the database.

    Use this tool if you need personal information about the user to respond to the request.:
    for example, his lifestyle, spending categories, amounts of expenses, income, hobbies, etc.

    """
    try:
        result = text2sql_client.execute_query(query)
        print("Результат запроса:")
        print(result)
        return result
    except RuntimeError as e:
        print(f"Произошла ошибка: {e}")


search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    description="""
    The tool searches for information on the Internet.
    
    Use the tool if there is enough publicly available information from the Internet to respond to the request.
    """
)

# TODO FEW SHOT
# TODO AGENT

print(search_tool.description)
tools_list = [text2sql_tool, search_tool]
