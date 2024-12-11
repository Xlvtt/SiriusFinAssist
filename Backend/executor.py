from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langchain import hub
from typing import TypedDict, Annotated
from langgraph.graph.message import AnyMessage, add_messages

from tools import tools_list


class ExecutorState(TypedDict):
    messages: Annotated[
        list[AnyMessage], add_messages]
    user_id: int
    is_last_step: bool


prompt = hub.pull("ih/ih-react-agent-executor")
prompt.pretty_print()

llm = ChatOpenAI(temperature=0)
executor_agent = create_react_agent(
    llm, tools=tools_list, state_modifier=prompt, #state_schema=State
)