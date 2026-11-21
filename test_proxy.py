import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model = os.getenv("DEFAULT_MODEL", "gemini-3.8-flash")

# Point the official OpenAI SDK at YOUR local gateway
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key",
)

print("--- 1. Non-Streaming Request ---")
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain HTTP connection pooling in one sentence."}],
)
print(resp.choices[0].message.content)

print("\n--- 2. Streaming Request ---")
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a 5-bullet checklist for deploying a FastAPI app."}],
    stream=True,
)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()