
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


# Local NLLB model path

LOCAL_PATH = r"C:\Users\Admin\Desktop\hugging_face\nllb-600M-offline"

# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)


# Load and quantize model

print(" Loading NLLB model (with dynamic quantization)...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_PATH, local_files_only=True)

# Apply dynamic quantization (only Linear layers)
model = torch.quantization.quantize_dynamic(
    base_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Move to CPU (quantized models run best on CPU)
device = torch.device("cpu")
model = model.to(device)

print(" NLLB model loaded with dynamic quantization on CPU.")


# Detect language code

def detect_language_code(text: str) -> str:
    text = text.strip()
    if not text:
        return "eng_Latn"
    # Devanagari (Hindi)
    if re.search(r'[\u0900-\u097F]', text):
        return "hin_Deva"
    # Gujarati
    elif re.search(r'[\u0A80-\u0AFF]', text):
        return "guj_Gujr"
    # English (Latin)
    elif re.search(r'[A-Za-z]', text):
        # Roman Hindi detection: if contains non-ASCII chars
        non_ascii_count = sum(1 for c in text if not c.isascii())
        if non_ascii_count > 0:
            return "hin_Deva"
        return "eng_Latn"
    else:
        return "eng_Latn"


# Thread-safe translation

def translate_text(text: str, target_lang: str, src_lang: str = None) -> str:
    """
    Translate text to target_lang using NLLB model.
    Optimized with dynamic quantization for faster CPU inference.
    """
    if not src_lang:
        src_lang = detect_language_code(text)

    try:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_length=256
            )

        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text.strip()

    except Exception as e:
        print(f" Translation error: {e}")
        return text


# Always translate to English

def translate_query_to_english(text: str) -> str:
    return translate_text(text, target_lang="eng_Latn")
