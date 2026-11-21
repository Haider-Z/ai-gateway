import os
from dotenv import load_dotenv
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import OpenAITextVectorizer

load_dotenv()

cache = SemanticCache(
    name="ai_gateway_cache",
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    distance_threshold=1.0,
    vectorizer=OpenAITextVectorizer(
        model="text-embedding-3-small",
        api_config={"api_key": os.environ["OPENAI_API_KEY"]},
    ),
    filterable_fields=[
        {"name": "tenant_id", "type": "tag"},
        {"name": "model", "type": "tag"},
    ],
)

# Seed one reference entry so this script works even on an empty Redis
cache.store(
    prompt="What is the capital of France?",
    response="Paris.",
    filters={"tenant_id": "tenant_a", "model": os.getenv("DEFAULT_MODEL", "gpt-6-luna")},
)

test_prompts = [
    "Tell me the capital city of France.",   # SHOULD HIT (same meaning)
    "Which city is France's capital?",       # SHOULD HIT (same meaning)
    "What is the capital of Germany?",       # MUST MISS (false-positive trap!)
]

for p in test_prompts:
    res = cache.check(prompt=p, num_results=1, distance_threshold=1.0)
    if res:
        dist = res[0].get("vector_distance", res[0].get("cosine_distance"))
        print(f"Distance: {float(dist):.4f} | Prompt: '{p}' -> Matched: '{res[0]['prompt']}'")
    else:
        print(f"No match found for: '{p}'")