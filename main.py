import os
from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN", "")
login(token=HF_TOKEN)
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
import requests, torch, re, uvicorn

app = FastAPI(title="My Business AI", version="1.0")

# ============ Load Model ============
print("📥 Loading model...")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

tokenizer = AutoTokenizer.from_pretrained(
    "Dayana0905/my-business-ai",
    token=HF_TOKEN
)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map=None
)
model = PeftModel.from_pretrained(
    base_model,
    "Dayana0905/my-business-ai",
    token=HF_TOKEN
)
model.eval()
device = "cpu"
print(f"✅ Model loaded!")

# ============ Rules ============
RULES = {
    "வணக்கம்": "வணக்கம்! நான் உன் AI assistant. எப்படி உதவட்டும்?",
    "vanakkam": "வணக்கம்! என்ன help வேணும்?",
    "hello": "Hello! I'm your AI assistant. How can I help you?",
    "hi there": "Hi there! How can I assist you today?",
    "நாளை 3pm appointment வேணும்": "நாளை மாலை 3 மணிக்கு confirmed! ID: #1006 ✅",
    "நாளை 9am appointment வேணும்": "நாளை காலை 9 மணிக்கு confirmed! ID: #1001 ✅",
    "நாளை 2pm appointment வேணும்": "நாளை மதியம் 2 மணிக்கு confirmed! ID: #1005 ✅",
    "naalai 3pm ku appointment venum": "நாளை 3pm confirmed da! ID: #1006 ✅",
    "naalai 9am ku appointment venum": "நாளை 9am confirmed da! ID: #1001 ✅",
    "tomorrow 3pm appointment": "Tomorrow 3pm confirmed! ID: #1006 ✅",
    "tomorrow 9am appointment": "Tomorrow 9am confirmed! ID: #1001 ✅",
    "kal 3 baje appointment": "कल दोपहर 3 बजे confirmed! ID: #1006 ✅",
    "kal 9 baje appointment": "कल सुबह 9 बजे confirmed! ID: #1001 ✅",
    "හෙට 3pm appointment": "හෙට දහවල් 3 ට confirmed! ID: #1006 ✅",
    "හෙට 9am appointment": "හෙට උදේ 9 ට confirmed! ID: #1001 ✅",
    "pipe leak": "Plumber assigned! Kumar will arrive in 2 hours ✅",
    "plumbing problem irukku": "Plumber assign pannitten! Kumar 2 hours la varuvan ✅",
    "light poguthu": "Electrician assign பண்ணிட்டேன்! Ravi 1 hour-ல் வருவார் ✅",
    "pipe leak ho raha": "Plumber assign किया! Kumar 2 घंटे में आएगा ✅",
    "bijli nahi": "Electrician assign किया! Ravi 1 घंटे में आएगा ✅",
    "ac not working": "AC technician assigned! Suresh will arrive in 2 hours ✅",
    "ac kharab": "AC technician assign किया! Suresh 2 घंटे में आएगा ✅",
    "පයිප්ප leak": "Plumber assign කළා! Kumar ඉක්මනින් එයි ✅",
    "විදුලිය නැහැ": "Electrician assign කළා! Ravi පැය 1 ඇතුළත එයි ✅",
    "price என்ன": "Basic ₹500, Standard ₹1000, Premium ₹2000 😊",
    "what is the price": "Basic ₹500, Standard ₹1000, Premium ₹2000 😊",
    "kitne paise": "Basic ₹500, Standard ₹1000, Premium ₹2000 😊",
    "මිල කීයද": "Basic ₹500, Standard ₹1000, Premium ₹2000 😊",
    "නமස්ते": "नमस्ते! मैं आपका AI assistant हूं। कैसे मदद करूं?",
    "नमस्ते": "नमस्ते! मैं आपका AI assistant हूं। कैसे मदद करूं?",
    "ආයුබෝවන්": "ආයුබෝවන්! මම ඔබේ AI assistant. කෙසේ උදව් කරන්නද?",
    "urgent": "Emergency team on the way! 🚨 Call: +91-98765-43210",
    "හදිසි": "ඉක්මනින් team එවනවා! 🚨",
}

# ============ Helper Functions ============
def detect_language(text):
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', text))
    if sinhala_chars > 0: return "sinhala"
    elif tamil_chars > 0: return "tamil"
    elif hindi_chars > 0: return "hindi"
    elif any(w in text.lower() for w in ["da","venum","irukku","poguthu","vanakkam","naalai"]): return "tanglish"
    elif any(w in text.lower() for w in ["hai","chahiye","baje","kal","nahi","aap"]): return "hindi"
    else: return "english"

def translate_text(text, target_lang):
    try:
        lang_map = {"tamil": "ta", "hindi": "hi", "sinhala": "si"}
        if target_lang in lang_map:
            return GoogleTranslator(source='en', target=lang_map[target_lang]).translate(text)
        return text
    except:
        return text

def ai_generate(message, system):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    return tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

def extract_web(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(['script','style','nav','footer']):
            tag.decompose()
        lines = [l.strip() for l in soup.get_text(separator='\n').split('\n') if l.strip()]
        return '\n'.join(lines[:200])
    except Exception as e:
        return f"Error: {e}"

# ============ API Endpoints ============
class ChatRequest(BaseModel):
    message: str

class AnalyzeRequest(BaseModel):
    url: str
    question: str

@app.get("/")
def home():
    return {"status": "✅ My Business AI Running!", "version": "1.0"}

@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message
    msg_lower = msg.lower().strip().rstrip("!").rstrip("?")
    lang = detect_language(msg)

    sorted_rules = sorted(RULES.items(), key=lambda x: len(x[0]), reverse=True)
    for key, response in sorted_rules:
        if key in msg_lower:
            return {"input": msg, "language": lang, "response": response, "type": "rule"}

    response = ai_generate(msg, "You are a helpful multilingual business AI. Be short and helpful.")
    if lang in ["tamil", "hindi", "sinhala"]:
        response = translate_text(response, lang)

    return {"input": msg, "language": lang, "response": response, "type": "ai"}

@app.post("/analyze-web")
def analyze_web(req: AnalyzeRequest):
    lang = detect_language(req.question)
    content = extract_web(req.url)

    en_question = req.question
    if lang != "english":
        try:
            en_question = GoogleTranslator(source='auto', target='en').translate(req.question)
        except:
            pass

    en_answer = ai_generate(
        f"Content:\n{content[:3000]}\n\nQuestion: {en_question}\n\nAnswer:",
        "You are a helpful document analyzer. Answer clearly in English."
    )

    final = translate_text(en_answer, lang) if lang in ["tamil","hindi","sinhala"] else en_answer
    return {"url": req.url, "question": req.question, "language": lang, "answer": final}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "Dayana0905/my-business-ai"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)