
from fastapi import FastAPI, UploadFile, File
import requests

app = FastAPI()


HUGGINGFACE_TOKEN = "abc"

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio using Hugging Face Whisper model (medium).
    """
    model_url = "https://api-inference.huggingface.co/models/openai/whisper-medium"

    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    audio_bytes = await file.read()

    response = requests.post(model_url, headers=headers, data=audio_bytes)

    if response.status_code != 200:
        return {"error": "Failed to transcribe audio", "details": response.text}

    result = response.json()

   
    if isinstance(result, list) and len(result) > 0:
        text = result[0].get("text", "")
    elif isinstance(result, dict):
        text = result.get("text", "")
    else:
        text = ""

    return {"transcript": text}
