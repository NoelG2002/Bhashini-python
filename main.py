from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from bhashini_translator import Bhashini

# Set up environment variables for authentication keys (assumes they are set)
user_id = os.getenv("USER_ID")
ulca_api_key = os.getenv("ULCA_API_KEY")
inference_api_key = os.getenv("INFERENCE_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# CORS setup to allow the frontend to access the API
origins = [
    "http://localhost:3000",  # Development frontend URL (Next.js)
    "https://bhashini-python-front-end.vercel.app/",  # Production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows your frontend to interact with backend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Request model for translation
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

# Route to handle text translation
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        bhashini = Bhashini(request.source_language, request.target_language)
        translated_text = bhashini.translate(request.text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle Text-to-Speech (TTS) functionality
@app.post("/tts")
async def text_to_speech(request: TranslationRequest):
    try:
        bhashini = Bhashini(request.source_language)
        base64_string = bhashini.tts(request.text)
        return {"audio_base64": base64_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle Automatic Speech Recognition (ASR) and NMT translation
@app.post("/asr_nmt")
async def asr_nmt(request: TranslationRequest):
    try:
        bhashini = Bhashini(request.source_language, request.target_language)
        text = bhashini.asr_nmt(request.text)  # Here text will be the base64 audio
        return {"translated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

