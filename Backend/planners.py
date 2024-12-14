from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Union
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps of plan to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Final response to user."""
    response: str


class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you think that plan has finished and you are ready to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )  # может вернуть финальный ответ или новый план


planner_llm = ChatOpenAI()
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a useful assistant for making detailed and logical action plans. \
            You have a financial assistant under your command. \
            He knows all the relevant information about the modern world stored on the internet, and also has access to the user`s transactions and accounts database \
            and is able to make any calculations with personal financial. \
            Your task is to make detailed action plans for him so that, following them successfully, the assistant can answer any user's request about finances. \
            These plans should include separate tasks that, when completed sequentially, will give the correct answer.
            The result of the last step always should be the final answer.
            You are responsible for the final result.
            """,
        ),
        ("system", "There is some info about current state of the world: {world_info}"),
        ("user", """
            Make a plan for your financial assistant to answer the question.
            
            Example of a question: Last purchase on the WB
            
            An example of a BAD plan:
            - find the latest purchase on the WB
            - give an answer
            
            An example of a GOOD plan:
            - Find out on the Internet what WB means
            - Knowing what WB means, find in the transaction database all the information about the last purchase on WB.
            - Determine the purchase amount, date, place of purchase (if applicable), and category of the product/service 
            - Give the user a response based on all the information that has been extracted. If no information was found, inform the user about it.
            
            My question: {input}
            
            Your GOOD plan:
            
        """)
    ]
)
planner = planner_prompt | planner_llm.with_structured_output(Plan)

# TODO потом объединить планнер и репланнер
# Для эксперимента выкинуть репланнер

replanner_llm = ChatOpenAI(temperature=0)
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a useful assistant for making detailed and logical action plans. \
            You have a financial assistant under your command. \
            He knows all the relevant information about the modern world stored on the internet, and also has access to the user`s transactions and accounts database \
            and is able to make any calculations with personal financial. \
            Your task is to make detailed action plans for him so that, following them successfully, the assistant can answer any user's request about finances. \
            These plans should include separate tasks that, when completed sequentially, will give the correct answer.
            The result of the last step always should be the final answer.
            You are responsible for the final result.
            """,
        ),
        ("system", "There is some info about current state of the world: {world_info}"),
        (
            "user",
            """
            My question was: {input}
            Your original plan was this:
            {plan}

            Your assistant have currently done the follow steps:
            {past_steps}
            
            Update your plan. Do not return the steps you have already completed as part of the plan. 
            If no further action is required, return to the user and give the final answer. 
            Otherwise, fill out the plan with the steps that remain to be completed to achieve the goal. 
            
            Adjust the plan so that the total number of completed steps does not exceed {steps_limit}. 
            As soon as the limit is exceeded, issue a response based on all the steps already completed. 
            If you are not sure about the final answer, inform me about it.
            """
        )
    ]
)
replanner = replanner_prompt | replanner_llm.with_structured_output(Act)

# TODO dynamic few-shot for plans (разметить планы)
# TODO one-shot replaning example
# репланнер должен добавлять в план последнюю извлеченную информацию
# Можно что: следовать начальному плану, но помнить, что было сделано
