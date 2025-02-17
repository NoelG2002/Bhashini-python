from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from bhashini_translator import Bhashini
import shutil
import io
from dotenv import load_dotenv

load_dotenv()

# Set up environment variables for authentication keys (assumes they are set)
userID= os.getenv("userID")
ulcaApiKey = os.getenv("ulcaApiKey")
InferenceApiKey = os.getenv("InferenceApiKey")

# Initialize FastAPI app
app = FastAPI()

# CORS setup to allow the frontend to access the API
origins = [
    "http://localhost:3000",  # Development frontend URL (Next.js)
    "https://bhashini-python-front-end.vercel.app",  # Production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported languages and their corresponding codes
LANGUAGES = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Maithili": "mai_Deva",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
}

# Request model for translation
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

# Route to handle text translation
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        source_lang_code = LANGUAGES.get(request.source_language)
        target_lang_code = LANGUAGES.get(request.target_language)
        
        if not source_lang_code or not target_lang_code:
            raise HTTPException(status_code=400, detail="Invalid language code.")
        
        bhashini = Bhashini(source_lang_code, target_lang_code)
        translated_text = bhashini.translate(request.text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle Text-to-Speech (TTS) functionality
@app.post("/tts")
async def text_to_speech(request: TranslationRequest):
    try:
        source_lang_code = LANGUAGES.get(request.source_language)
        
        if not source_lang_code:
            raise HTTPException(status_code=400, detail="Invalid language code.")
        
        bhashini = Bhashini(source_lang_code)
        base64_string = bhashini.tts(request.text)
        return {"audio_base64": base64_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle Automatic Speech Recognition (ASR) and NMT translation
@app.post("/asr_nmt")
async def asr_nmt(source_language: str, target_language: str, file: UploadFile = File(...)):
    try:
        source_lang_code = LANGUAGES.get(source_language)
        target_lang_code = LANGUAGES.get(target_language)
        
        if not source_lang_code or not target_lang_code:
            raise HTTPException(status_code=400, detail="Invalid language code.")
        
        # Save the uploaded file temporarily
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        # Initialize Bhashini for ASR and NMT
        bhashini = Bhashini(source_lang_code, target_lang_code)
        
        # Pass the audio file directly to Bhashini's ASR and NMT method
        translated_text, recognized_text = bhashini.asr_nmt(audio_path)

        # Clean up the temporary audio file
        os.remove(audio_path)

        return {"recognized_text": recognized_text, "translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
