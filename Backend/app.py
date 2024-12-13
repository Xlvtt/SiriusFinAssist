from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from agent_run import get_agent_answer as GoAnswer
import agent_run
import os
from dotenv import load_dotenv

load_dotenv()

print('-------------> NEW FILE INIT')

app = FastAPI()

print(os.path.abspath('finance_data.duckdb'))

print(os.getcwd())



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatResponse(BaseModel):
    message: str
    success: bool = True
    error: Optional[str] = None


@app.get("/chat")
async def chat_endpoint(
    message: str = Query(..., description="Сообщение от пользователя")
):
    try:
        #respObj = ChatResponse(message=bot.get(message))
        print('------>' + message)
        respObj = ChatResponse(message=agent_run.get_agent_answer(message))
        print(str(respObj))
        return respObj
    except Exception as e:
        return ChatResponse(
            message="Error",
            success=False,
            error=str(e)
        )


# Добавим корневой роут для проверки
@app.get("/")
async def root():
    return {"status": "API работает!"}

# uvicorn.run('app:app', host = "localhost", port=8001,reload=True)

