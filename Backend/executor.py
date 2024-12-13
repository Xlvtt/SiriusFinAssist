from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain import hub
from typing import TypedDict, Annotated
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools import tools_list


class ExecutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps


# Как мэтчится структурированный вывод, граф и tool calling??? У меня одно не мешает другому?
# Запрос выходит за пределы таблицы

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a useful financial assistant, you have two sources of information at your disposal: \
         a database with user accounts and transactions and all Internet data.
         You can use them if you need it.
         Then process information, correct your mistakes and draw logical conclusions to achieve your goal.
         """),
        ("placeholder", "{messages}")
    ]
)  # TODO reasoning
# После каждого вызова тула его надо просить рефлексировать

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# агент не знает свои выполненные шаги вообще и по кругу делает одно и то же
# типа если данные не найдены...
# В агента надо возвращать тулу так, чтобы он все не переделывал
# TODO явно ограничить remaining_steps

def llm_node(state: ExecutorState):
    pass


executor_graph_builder = StateGraph(ExecutorState)

executor_agent = create_react_agent(
    llm, tools=tools_list, state_modifier=prompt, state_schema=ExecutorState,
)