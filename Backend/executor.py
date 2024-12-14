from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain import hub
from typing import TypedDict, Annotated
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools import tools_list


class ExecutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep


class ExecutorOutput(BaseModel):
    answer: str = Field(
        description="answer to user`s question"
    )
    reasoning: str = Field(
        description="If a tool was called as a result of the query, describe why this particular tool was selected and how it will help answer the question"
    )
# Возможно разифать структуру


# Как мэтчится структурированный вывод, граф и tool calling??? У меня одно не мешает другому?
# Запрос выходит за пределы таблицы

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are a useful financial assistant, you have two sources of information at your disposal: \
         a database with user accounts and transactions and all Internet data.
         Use them, process information, correct your mistakes and draw logical conclusions to achieve your goal.
         """),
        ("placeholder", "{messages}")
    ]
)  # TODO reasoning

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
executor_agent = create_react_agent(
    llm, tools=tools_list, state_modifier=prompt, state_schema=ExecutorState,
)
