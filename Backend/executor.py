from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain import hub
from typing import TypedDict, Annotated
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate

from tools import tools_list


class ExecutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep
    reasoning: Annotated[AIMessage, ""]


prompt = hub.pull("ih/ih-react-agent-executor")
prompt.pretty_print()

# TODO экузекутор должен быть агентом и уметь пробовать разные псевдонимы
# TODO STRUCTURED WITH REASONING
# TODO добавтить в промт текущее время


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
)

# TODO у нас экзекутор вообще получает план на вход?????
llm = ChatOpenAI(temperature=0)
executor_agent = create_react_agent(
    llm, tools=tools_list, state_modifier=prompt,  state_schema=ExecutorState
)
