from typing import Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from Qwen import Bot
from gradio_client import Client


class MockAssistant:
    def __init__(self):
        self.client = Client("Qwen/Qwen2.5")

    def invoke(self, input_text: str):
        result = self.client.predict(
            query=input_text,
            history=[],
            system="You are helpful finance assistant. You know some information about user: his average income is 100 000 rubs na dhe loves pony",
            radio="72B",
            api_name="/model_chat"
        )[1][0][-1]['text']
        return result


assistant = MockAssistant()


app = FastAPI()

bot = Bot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()


class ChatResponse(BaseModel):
    message: str
    success: bool = True
    error: Optional[str] = None


@app.get("/chat")
async def chat_endpoint(
    message: str = Query(..., description="Сообщение от пользователя")
):
    try:
        respObj = assistant.invoke(message)
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

uvicorn.run(app, host="0.0.0.0", port=8001)

