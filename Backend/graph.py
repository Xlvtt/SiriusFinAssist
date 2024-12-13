import datetime
from typing import TypedDict, List, Tuple, Annotated
import operator
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from planners import planner, Response
from executor import executor_agent, llm


def get_world_info():
    return f"""
    Date and time: {datetime.datetime.now()}
    """


class PlanState(TypedDict):
    input: str
    world_info: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


def executor_step(state: PlanState):
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(state["plan"]))
    task = state["plan"][len(state["past_steps"])]

    task_formatted = f"""
    There is some info about current state of the world: {state["world_info"]}.
    For the following plan: 
    {plan_str}

    Using the knowledge gained in the previous steps, complete the current task: {len(state["past_steps"]) + 1}. {task}.
    Execute it so that the result is useful for the whole plan.

    My question was: {state["input"]}
    You have currently done the follow steps: 
    {state["past_steps"]}
    """
    answer = executor_agent.invoke({"messages": [("user", task_formatted)]})
    return {
        "past_steps": [task, answer["messages"][-1].content]
    }


# world_info на момент запроса задается и пишется
def planner_step(state: PlanState):
    plan = planner.invoke(
        {"input": state["input"], "messages": [("user", state["input"])], "world_info": get_world_info()}
    )
    return {"plan": plan.steps, "world_info": get_world_info()}


def end_condition(state: PlanState):
    if len(state["past_steps"]) < len(state["plan"]):
        return "executor"
    else:
        return "finalize"


def final_answer(state: PlanState):
    return state


graph_builder = StateGraph(PlanState)

graph_builder.add_node("planner", planner_step)
graph_builder.add_node("executor", executor_step)
graph_builder.add_node("finalize", final_answer)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_conditional_edges("executor", end_condition)
graph_builder.add_edge("finalize", END)

assistant = graph_builder.compile()
