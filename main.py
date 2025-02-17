from fastapi import FastAPI
from pydantic import BaseModel
import os
from bhashini_translator import Bhashini
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    text: str

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    bhashini = Bhashini(request.source_language, request.target_language)
    translated_text = bhashini.translate(request.text)
    return {"translated_text": translated_text}
