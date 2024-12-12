from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Qwen import Bot

from dotenv import load_dotenv

load_dotenv()

from graph import assistant

app = FastAPI()

bot = Bot()

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
        respObj = ChatResponse(message=assistant.invoke({"input": message}))
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


