
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from bhashini_translator import Bhashini
import shutil
from pydub import AudioSegment
import base64
import io
import re
from io import BytesIO
import asyncio
from dotenv import load_dotenv
import uvicorn





load_dotenv()

# Set up environment variables for authentication keys (assumes they are set)
userID= os.getenv("userID")
ulcaApiKey = os.getenv("ulcaApiKey")
InferenceApiKey = os.getenv("InferenceApiKey")

os.environ['userID'] = userID
os.environ['ulcaApiKey'] = ulcaApiKey
os.environ['InferenceApiKey']=InferenceApiKey
os.environ['DefaultPipeLineId']='64392f96daac500b55c543cd'

# Initialize FastAPI app
app = FastAPI()

# CORS setup to allow the frontend to access the API
origins = [
    "http://localhost:3000",  # Development frontend URL (Next.js)
    "https://bhashini-kamco.vercel.app",  # Production frontend URL
    "https://agrivaani.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported language codes
LANGUAGE_CODES = {
    "hi", "gom", "kn", "doi", "brx", "ur", "ta", "ks", "as", "bn", "mr", "sd", "mai", "pa", "ml", "mni", "te", "sa", "ne", "sat", "gu", "or", "en"
}

# Request model for translation with only language codes
class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


# Route to handle text translation
@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        # Check if the source and target languages are valid codes
        if request.source_language not in LANGUAGE_CODES or request.target_language not in LANGUAGE_CODES:
            raise HTTPException(status_code=400, detail="Invalid language code.")
        
        bhashini = Bhashini(request.source_language, request.target_language)
        translated_text = bhashini.translate(request.text)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle Text-to-Speech (TTS) functionality
@app.post("/tts")
async def text_to_speech(request: TranslationRequest):
    try:
        # Instantiate the Bhashini object for translation (if needed)
        bhashini = Bhashini(request.source_language, request.target_language)
        translated_text = bhashini.translate(request.text)

        # Instantiate the Bhashini object for TTS
        bhashini = Bhashini(request.target_language)
        base64_string = bhashini.tts(translated_text)

        # Decode base64 string to binary audio
        audio_data = base64.b64decode(base64_string)

        # Save the audio file
        audio_file_path = "output_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)

        return {"audio_file": audio_file_path, "audio_base64": base64_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from pydub import AudioSegment
import base64
import io

def preprocess_audio(audio_bytes):
    # Convert bytes to an audio file in memory
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    # Convert to 16kHz mono WAV (recommended for ASR)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Compress & export to memory
    buffer = io.BytesIO()
    audio.export(buffer, format="wav", bitrate="32k")
    
    return buffer.getvalue()  # Return processed audio bytes



# Route to handle Automatic Speech Recognition (ASR) and NMT translation
@app.post("/asr_nmt")
async def asr_nmt(audio_file: UploadFile = File(...), source_language: str = Form(...), target_language: str = Form(...)):
     try:
        audio_content = await audio_file.read()
        processed_audio = preprocess_audio(audio_content)
        
        # Encode efficiently in Base64 (without memory overflow)
        audio_base64 = base64.b64encode(processed_audio).decode('utf-8')
        
        # Initialize ASR and NMT processing
        bhashini = Bhashini(source_language, target_language)
        translated_text = bhashini.asr_nmt(audio_base64)

        if not translated_text:
            raise HTTPException(status_code=500, detail="No translated text returned from ASR")

        return {"translated_text": translated_text}
     except HTTPException as e:
         print(f"API Response: {e.detail}")
         return {"error": f"API Response: {e.detail}"}
     except Exception as e:
         import traceback
         print(f"General Error: {str(e)}")
         traceback.print_exc()  # Prints the full error traceback
         return {"error": f"General Error: {str(e)}"}

