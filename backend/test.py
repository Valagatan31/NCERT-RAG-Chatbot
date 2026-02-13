from openai import OpenAI
client = OpenAI(api_key="abc")

with open("test.mp3", "rb") as f:
    resp = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
print(resp.text)
