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
from pydub.silence import detect_nonsilent

async def split_audio(audio_path, chunk_length_ms=20000, min_silence_len=700, silence_thresh=-40):
    """Splits audio at natural pauses instead of fixed-length chunks."""
    def sync_split():
        audio = AudioSegment.from_file(audio_path)

        # Detect speech segments (non-silent parts)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

        chunks = []
        start = 0

        for end in nonsilent_ranges:
            if end[1] - start > chunk_length_ms:  # If speech segment exceeds chunk size
                end = start + chunk_length_ms  # Enforce max length
            chunk = audio[start:end[1]]
            chunks.append(chunk)
            start = end[1]  # Move to next segment

        return chunks

    # Process chunking asynchronously
    chunks = await asyncio.to_thread(sync_split)

    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        chunk_path = f"chunk_{idx}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return chunk_paths



async def process_chunk(chunk_path, bhashini):
    """Process a single chunk for ASR and NMT."""
    with open(chunk_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Ensure asr_nmt is called correctly (depending on whether it's async or not)
    return await asyncio.to_thread(bhashini.asr_nmt, audio_base64)


import difflib

def merge_sentences(translated_texts):
    """Merges overlapping text chunks by detecting repeated phrases using similarity matching."""
    merged_text = []
    prev_sentence = ""

    for text in translated_texts:
        text = text.strip()

        if prev_sentence:
            # Find longest common substring to detect overlap
            matcher = difflib.SequenceMatcher(None, prev_sentence, text)
            match = matcher.find_longest_match(0, len(prev_sentence), 0, len(text))

            if match.size > 10:  # If significant overlap detected
                text = text[match.b:end]  # Trim overlapping part
            
        merged_text.append(text)
        prev_sentence = text  # Update last sentence for next iteration

    return " ".join(merged_text)




# Route to handle Automatic Speech Recognition (ASR) and NMT translation
@app.post("/asr_nmt")
async def asr_nmt(audio_file: UploadFile = File(...), source_language: str = Form(...), target_language: str = Form(...)):
    try:
        # Check if source and target languages are valid codes
        if source_language not in LANGUAGE_CODES or target_language not in LANGUAGE_CODES:
            raise HTTPException(status_code=400, detail="Invalid language code.")

        # Read the uploaded file (audio)
       temp_file = f"temp_{audio_file.filename}"
        with open(temp_file, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)

        # Use improved split function
        chunk_paths = await split_audio(temp_file)

        # Initialize Bhashini API
        bhashini = Bhashini(source_language, target_language)

        # Process each chunk asynchronously
        translated_texts = await asyncio.gather(*(process_chunk(chunk, bhashini) for chunk in chunk_paths))

        # Merge translated sentences with improved merging logic
        merged_translation = merge_sentences(translated_texts)

        # Clean up temporary files
        for chunk_path in chunk_paths:
            os.remove(chunk_path)
        os.remove(temp_file)

        return {"translated_text": merged_translation}


        # Convert the file content to base64
        #audio_content = await audio_file.read()
        #audio_base64 = base64.b64encode(audio_content).decode('utf-8')
               
        #os.remove(temp_file)

        # Initialize Bhashini for ASR and NMT
        #bhashini = Bhashini(source_language, target_language)
        
        # Pass the base64-encoded string to Bhashini's asr_nmt method
        #translated_text = bhashini.asr_nmt(audio_base64)

        #return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
