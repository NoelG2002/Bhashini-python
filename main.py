from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bhashini_translator import Bhashini
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API credentials from environment variables
USER_ID = os.getenv("ULCA_USER_ID")
ULCA_API_KEY = os.getenv("ULCA_API_KEY")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Language codes to handle valid inputs
LANGUAGES = [
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "eng_Latn", "gom_Deva", 
    "hin_Deva", "kas_Arab", "kas_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", 
    "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", 
    "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml", "tel_Telu", "urd_Arab"
]

# Pydantic model for request body
class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    text: str

# Initialize Bhashini translator instance with API keys
def get_bhashini_instance(source_language: str, target_language: str = None):
    return Bhashini(
        source_language, 
        target_language,
        user_id=USER_ID,
        ulca_api_key=ULCA_API_KEY,
        inference_api_key=INFERENCE_API_KEY
    )

@app.post("/translate")
async def translate(request: TranslationRequest):
    # Validate if provided languages are supported
    if request.source_language not in LANGUAGES or request.target_language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    # Initialize Bhashini instance
    bhashini = get_bhashini_instance(request.source_language, request.target_language)
    
    try:
        translated_text = bhashini.translate(request.text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/tts")
async def text_to_speech(request: TranslationRequest):
    # Validate languages
    if request.source_language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid source language code")
    
    # Initialize Bhashini instance for TTS
    bhashini = get_bhashini_instance(request.source_language)
    
    try:
        base64_string = bhashini.tts(request.text)
        return {"base64_string": base64_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/asr_nmt")
async def automatic_speech_recognition(request: TranslationRequest):
    # Validate languages
    if request.source_language not in LANGUAGES or request.target_language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid language code")
    
    # Initialize Bhashini instance for ASR
    bhashini = get_bhashini_instance(request.source_language, request.target_language)
    
    try:
        text = bhashini.asr_nmt(request.text)  # Assume `text` is base64 audio data here
        return {"translated_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")
