
import google.generativeai as genai
import re

# Configure Gemini API
genai.configure(api_key="abc")

# Map language codes to human-readable names
LANG_MAP = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "guj_Gujr": "Gujarati"
}

def generate_answer(question: str, context: str, user_lang: str = "eng_Latn", prompt_override: str = None) -> str:
    """
    Generate an answer using Gemini model in the user's language.
    """
    target_language = LANG_MAP.get(user_lang, "English")

    # Use custom prompt if provided
    prompt = prompt_override if prompt_override else f"""
You are a helpful multilingual tutor.

Question language: {target_language}

Answer the following question using ONLY {target_language}.
Do NOT translate it to English or any other language.
Use clear and simple language.
Write the answer in point-wise form (each point on a new line).
Avoid Markdown symbols like **, #, or bullets.

Question:
{question}

Context:
{context}
"""

    # Generate content using Gemini
    response = genai.GenerativeModel("models/gemini-2.5-pro").generate_content(prompt)
    text = response.text

    # Remove extra markdown symbols just in case
    clean_text = re.sub(r"[\*\#]", "", text).strip()
    return clean_text

