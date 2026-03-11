import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="YanaAI", version="2.0")

KAGGLE_URL = "https://overloud-lanelle-unmaterialistically.ngrok-free.dev"

class ChatRequest(BaseModel):
    message: str

class AnalyzeRequest(BaseModel):
    url: str
    question: str

@app.get("/")
def home():
    return {"status": "✅ YanaAI v2 Running!", "version": "2.0"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        r = requests.post(
            f"{KAGGLE_URL}/chat",
            json={"message": req.message},
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {"response": "YanaAI starting... 30 seconds wait!", "error": str(e)}

@app.post("/analyze-web")
def analyze_web(req: AnalyzeRequest):
    try:
        r = requests.post(
            f"{KAGGLE_URL}/analyze-web",
            json={"url": req.url, "question": req.question},
            timeout=120
        )
        return r.json()
    except Exception as e:
        return {"answer": "Analyzer starting... Try again!", "error": str(e)}

@app.get("/health")
def health():
    try:
        r = requests.get(f"{KAGGLE_URL}/health", timeout=10)
        return r.json()
    except:
        return {"status": "starting..."}