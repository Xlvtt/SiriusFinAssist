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
    reasoning: Annotated[AIMessage, ""]


# class ExecutorOutput(BaseModel):
#     reasoning: str = Field(
#         description="If a tool was called as a result of the query, describe why this particular tool was selected and how it will help answer the question"
#     )
#     answer: str = Field(
#         description="" # TODO DESCRIPTION, TOOL CHOOSING
#     )


# prompt = hub.pull("ih/ih-react-agent-executor")
# prompt.pretty_print()

# Как мэтчится структурированный вывод, граф и tool calling??? У меня одно не мешает другому?
# TODO экузекутор должен быть агентом и уметь пробовать разные псевдонимы
# TODO STRUCTURED WITH REASONING
# TODO добавтить в промт текущее время

# TODO with structured output


# экзекутор должен видеть, что глобально было сделано, что запланировано и
#     """
# Помни, что тебе нужно будет выполнить весь план.
# Сделай текущий шаг так, чтобы он был полезен для выполнения остального плана
# Если запрос выходит за рамки временных интервалов таблицы...
# """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are helpful assistant"),
        ("placeholder", "{messages}")
    ]
) # TODO PROMPT WITH FEW SHOT

llm = ChatOpenAI(temperature=0)
executor_agent = create_react_agent(
    llm, tools=tools_list, state_modifier=prompt, state_schema=ExecutorState, messages_modifier=
)
