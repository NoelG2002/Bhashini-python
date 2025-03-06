from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from bhashini_translator import Bhashini
import shutil
from pydub import AudioSegment
import base64
import io
from io import BytesIO
import tempfile
from dotenv import load_dotenv



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

def split_audio(audio_path, chunk_length_ms=20000):
    """
    Splits an audio file into chunks of the given length (in milliseconds).
    Returns a list of paths to chunk files.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    
    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        chunk_path = f"chunk_{idx}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    
    return chunk_paths


# Route to handle Automatic Speech Recognition (ASR) and NMT translation
import asyncio
import aiofiles
from tempfile import NamedTemporaryFile

async def process_audio_chunk(chunk_path, bhashini):
    """Handles ASR processing of a single audio chunk asynchronously."""
    async with aiofiles.open(chunk_path, "rb") as f:
        audio_base64 = base64.b64encode(await f.read()).decode("utf-8")

    return await asyncio.to_thread(bhashini.asr_nmt, audio_base64)  # Run blocking ASR in a thread

@app.post("/asr_nmt")
async def asr_nmt(
    audio_file: UploadFile = File(...), 
    source_language: str = Form(...), 
    target_language: str = Form(...)
):
    try:
        if source_language not in LANGUAGE_CODES or target_language not in LANGUAGE_CODES:
            raise HTTPException(status_code=400, detail="Invalid language code.")

        # ✅ Corrected Async File Handling
        temp_file_path = None
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_file_path = temp_audio.name
            await temp_audio.write(await audio_file.read())

        # ✅ Split Audio into Chunks
        chunk_paths = split_audio(temp_file_path, chunk_length_ms=20000)

        # ✅ Initialize Bhashini API
        bhashini = Bhashini(source_language, target_language)

        # ✅ Process all audio chunks concurrently
        translated_texts = await asyncio.gather(*[process_audio_chunk(chunk, bhashini) for chunk in chunk_paths])

        return {"translated_text": " ".join(translated_texts)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ✅ Cleanup Temporary Files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        for chunk in chunk_paths:
            if os.path.exists(chunk):
                os.remove(chunk)
