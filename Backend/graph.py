from typing import TypedDict, List, Tuple, Annotated
import operator
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from planners import planner, replanner, Response
from executor import executor_agent


class PlanState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


def executor_step(state: PlanState):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]

    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = executor_agent.invoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


def planner_step(state: PlanState):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


def replanner_step(state: PlanState):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanState):
    if "response" in state and state["response"]:
        return END
    else:
        return "executor"


graph_builder = StateGraph(PlanState)

graph_builder.add_node("planner", planner_step)
graph_builder.add_node("executor", executor_step)
graph_builder.add_node("replanner", replanner_step)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_edge("executor", "replanner")
graph_builder.add_conditional_edges("replanner", should_end, {"executor": "executor", END: END})

assistant = graph_builder.compile()