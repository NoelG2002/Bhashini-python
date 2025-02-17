from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bhashini_translator import Bhashini
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Initialize Bhashini translator instance
def get_bhashini_instance(source_language: str, target_language: str = None):
    return Bhashini(source_language, target_language)

@app.post("/translate")
async def translate(request: TranslationRequest):
    # Validate if provided languages are supported
    if request.source_language not in LANGUAGES or request.target_language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    # Initialize Bhashini instance
    bhashini = get_bhashini_instance(request.source_language, request.target_language)
    
    try:
        translated_text = bhashini.translate(request.text)
        retur
