from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import infer
app=FastAPI()
class ChatRequest(BaseModel):
    message: str
@app.get("/")
def home():
    return{
        "status":"API is running"
    }
@app.post("/chat")
def chat(request: ChatRequest):
    response=infer(request.message)
    return{
        "response": response
    }