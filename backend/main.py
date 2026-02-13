from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from helper import translate_query_to_english, detect_language_code
from gemini_helper import generate_answer
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import re
import time
import tempfile
import json
from faster_whisper import WhisperModel  
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LOCAL_WHISPER_PATH = r"C:\Users\Admin\Desktop\hugging_face\whisper-medium-ct2"

print(" Loading Whisper model in INT8 (Fast + Low RAM)...")
asr_model = WhisperModel(
    LOCAL_WHISPER_PATH,
    device="cpu",               
    compute_type="int8_float32" 
)
print("Whisper model loaded successfully in INT8 mode!")


class Query(BaseModel):
    text: str

BASE_PATH = r"C:\Users\Admin\Desktop\curio\voice_assistence_python\backend\vector_db"
model_path = r"C:\Users\Admin\Desktop\hugging_face\all-distilroberta-v1"

print(" Loading embedding model...")
embed_model = SentenceTransformer(model_path)

print(" Loading FAISS index & metadata...")
index = faiss.read_index(os.path.join(BASE_PATH, "vector_db.index"))
metadata = np.load(os.path.join(BASE_PATH, "metadata.npy"), allow_pickle=True)
chunks_path = os.path.join(BASE_PATH, "chunks.npy")
chunks = np.load(chunks_path, allow_pickle=True) if os.path.exists(chunks_path) else []

print(f" Loaded {len(chunks)} text chunks into memory")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"error": "Empty audio file."}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        print(" Transcribing with Faster-Whisper (INT8)...")
        start_transcribe = time.time()
        segments, info = asr_model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True
        )
        text = " ".join([segment.text for segment in segments]).strip()
        end_transcribe = time.time()
        os.remove(tmp_path)

        print(f"⏱ Transcription time: {end_transcribe - start_transcribe:.2f} sec")
        print(f"⏱ Total processing time: {end_transcribe - start_time:.2f} sec")

        return {"transcript": text}
    except Exception as e:
        return {"error": str(e)}



@app.post("/transcribe_stream")
async def transcribe_stream(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse({"error": "Empty audio file."}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        def ndjson_generator():
            start_time = time.time()
            start_human_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            print(f"🚀 Transcription Started at: {start_human_time}")

            yield json.dumps({
                "type": "start",
                "start_time": start_human_time
            }) + "\n"

            full_text = []
            seg_count = 0

            for segment in asr_model.transcribe(
                tmp_path, beam_size=3, vad_filter=True
            )[0]:
                seg_count += 1
                full_text.append(segment.text or "")
                elapsed_sec = round((time.time() - start_time), 2)

                print(f"Segment {seg_count} | Time: {elapsed_sec} sec | Text: {segment.text}")

                yield json.dumps({
                    "type": "partial",
                    "text": " ".join(full_text).strip(),
                    "segment_index": seg_count,
                    "elapsed_sec": elapsed_sec
                }) + "\n"

            end_time = time.time()
            total_sec = round((end_time - start_time), 2)
            end_human_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

            print(f"Final Transcript Ready at: {end_human_time}")
            print(f"Total Transcription Time: {total_sec} sec")
            print(f"Final Text: {' '.join(full_text).strip()}")

            yield json.dumps({
                "type": "final",
                "text": " ".join(full_text).strip(),
                "segments": seg_count,
                "elapsed_sec": total_sec,
                "end_time": end_human_time
            }) + "\n"

            try:
                os.remove(tmp_path)
            except:
                pass

        return StreamingResponse(ndjson_generator(), media_type="application/x-ndjson")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ask")
async def ask(query: Query):
    original_text = query.text or ""
    original_text = original_text.strip()

    if not original_text:
        return {"question": original_text, "answer": "Please provide a valid question or transcript."}

    user_lang = detect_language_code(original_text)

    start_query_translation = time.time()
    english_query = translate_query_to_english(original_text)
    end_query_translation = time.time()
    print(f"⏱ Query translation time: {end_query_translation - start_query_translation:.2f} sec")

    start_search_time = time.time()
    query_emb = embed_model.encode(english_query)
    D, I = index.search(np.array([query_emb]).astype("float32"), k=5)
    context_eng = "\n".join([chunks[i] for i in I[0]]) if len(chunks) > 0 else ""
    end_search_time = time.time()
    print(f"⏱ Searching time: {end_search_time - start_search_time:.2f} sec")

    if not context_eng.strip():
        return {"question": original_text, "answer": "Sorry, I don't have information on this topic."}

    prompt = f"""
    You are a multilingual tutor.

    User's question language: {user_lang}
    The user's question below is in their original language.
    You MUST reply only in the same language — do NOT translate it.
    Use simple and clear language. Explain in numbered points (1, 2, 3, ...).
    Avoid Markdown symbols like **, #, or bullets.

    Question: {original_text}

    Context:
    {context_eng}

    Instructions:
    - Use the context to answer as thoroughly as possible.
    - If context is partial or incomplete, do your best to reason and generate a meaningful answer.
    - If the context does not contain the answer, reply: "Sorry, I don't have information on this topic."
    """

    start_gemini = time.time()
    answer = generate_answer(original_text, context_eng, user_lang=user_lang, prompt_override=prompt)
    end_gemini = time.time()
    print(f"⏱ Gemini response time: {end_gemini - start_gemini:.2f} sec")

    clean_answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    total_time = (
        (end_query_translation - start_query_translation)
        + (end_gemini - start_gemini)
        + (end_search_time - start_search_time)
    )

    print(f"⏱ Total processing time: {total_time:.2f} sec")
    return {"question": original_text, "answer": clean_answer}
