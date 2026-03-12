import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from supabase import create_client

app = FastAPI(title="YanaAI Platform", version="3.0")

# Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

KAGGLE_URL = os.environ.get("KAGGLE_URL", "https://overloud-lanelle-unmaterialistically.ngrok-free.dev")

class ChatRequest(BaseModel):
    message: str
    company_id: str = None
    customer_phone: str = None
    customer_name: str = None

class AnalyzeRequest(BaseModel):
    url: str
    question: str
    company_id: str = None

@app.get("/")
def home():
    return {"status": "YanaAI Platform v3.0 Running!"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Company data fetch (RAG)
        context = ""
        if req.company_id:
            data = supabase.table("company_data")\
                .select("content")\
                .eq("company_id", req.company_id)\
                .execute()
            if data.data:
                context = "\n".join([d["content"] for d in data.data])

        # Kaggle AI call
        r = requests.post(
            f"{KAGGLE_URL}/chat",
            json={"message": req.message, "context": context},
            timeout=60
        )
        response = r.json()

        # Save conversation
        if req.company_id:
            customer_id = None
            if req.customer_phone:
                existing = supabase.table("customers")\
                    .select("id")\
                    .eq("phone", req.customer_phone)\
                    .eq("company_id", req.company_id)\
                    .execute()
                if existing.data:
                    customer_id = existing.data[0]["id"]
                else:
                    new_c = supabase.table("customers").insert({
                        "company_id": req.company_id,
                        "name": req.customer_name or "Unknown",
                        "phone": req.customer_phone
                    }).execute()
                    customer_id = new_c.data[0]["id"]

            supabase.table("conversations").insert({
                "company_id": req.company_id,
                "customer_id": customer_id,
                "message": req.message,
                "response": response.get("response", ""),
                "language": response.get("language", "english")
            }).execute()

        return response

    except Exception as e:
        return {"response": "YanaAI starting... Try again!", "error": str(e)}

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
        return {"answer": "Try again!", "error": str(e)}

@app.post("/company/register")
def register_company(data: dict):
    try:
        result = supabase.table("companies").insert({
            "name": data.get("name"),
            "email": data.get("email"),
            "password": data.get("password"),
            "bot_name": data.get("bot_name", "AI Assistant"),
            "bot_color": data.get("bot_color", "#007bff"),
            "plan": data.get("plan", "starter")
        }).execute()
        return {"status": "Company registered!", "company": result.data[0]}
    except Exception as e:
        return {"status": "Error", "error": str(e)}

@app.post("/company/upload-data")
def upload_data(data: dict):
    try:
        result = supabase.table("company_data").insert({
            "company_id": data.get("company_id"),
            "type": data.get("type", "text"),
            "content": data.get("content"),
            "filename": data.get("filename", "")
        }).execute()
        return {"status": "Data uploaded!", "id": result.data[0]["id"]}
    except Exception as e:
        return {"status": "Error", "error": str(e)}

@app.get("/company/{company_id}/analytics")
def get_analytics(company_id: str):
    try:
        chats = supabase.table("conversations")\
            .select("id").eq("company_id", company_id).execute()
        customers = supabase.table("customers")\
            .select("id").eq("company_id", company_id).execute()
        return {
            "total_chats": len(chats.data),
            "total_customers": len(customers.data)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.0"}