import os
import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENAI_API_KEY"]

PAIRS = [
    # True Positives (SHOULD hit cache)
    ("TRUE PARAPHRASE 1", "What kind of role is he looking for?", "What sort of opportunity is he currently targeting?"),
    ("TRUE PARAPHRASE 2", "Who is Zaman Haider?", "who is zaman"),
    ("TRUE PARAPHRASE 3", "What is his CGPA?", "what is the cgpa of zaman"),
    # True Negatives (MUST NOT hit cache — false positives would give wrong answers!)
    ("UNRELATED 1", "What kind of role is he looking for?", "What is his CGPA?"),
    ("UNRELATED 2", "Where is Zaman located?", "What programming languages does he know?"),
]

all_texts = list({t for _, a, b in PAIRS for t in (a, b)})
resp = httpx.post(
    "https://api.openai.com/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "text-embedding-3-small", "input": all_texts},
    timeout=30.0,
)
resp.raise_for_status()

vecs = {}
for text, item in zip(all_texts, resp.json()["data"]):
    arr = np.array(item["embedding"], dtype=np.float32)
    vecs[text] = arr / np.linalg.norm(arr)

print(f"{'Pair Type':<20} | {'Cosine Dist':<12} | Current (0.25)")
print("-" * 55)
for label, q1, q2 in PAIRS:
    # RedisVL cosine distance = 1 - cosine_similarity
    dist = 1.0 - float(np.dot(vecs[q1], vecs[q2]))
    status = "HIT" if dist <= 0.25 else "MISS"
    print(f"{label:<20} | {dist:<12.4f} | {status}")