import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model = os.getenv("DEFAULT_MODEL", "gpt-6-luna")

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy-key")


def ask(label: str, prompt: str, tenant_id: str = "tenant_a"):
    t0 = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={"X-Tenant-ID": tenant_id},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    status = raw.headers.get("x-cache-status", "UNKNOWN")
    distance = raw.headers.get("x-cache-distance", "-")
    answer = raw.parse().choices[0].message.content

    print(f"\n[{label}]")
    print(f"  Cache Status : {status} (distance={distance})")
    print(f"  Latency      : {elapsed_ms:.1f} ms")
    print(f"  Answer       : {answer[:100]}...")


# 1. Cold prompt -> should MISS and call OpenAI
ask("1. Cold Query (tenant_a)", "What is the capital of France?")

# 2. Exact same prompt (different casing/spaces) -> should hit Tier 1 in ~5ms
ask("2. Exact Repeat (tenant_a)", "  what is the capital of france?  ")

# 3. Paraphrased prompt -> should hit Tier 2 Semantic Cache!
ask("3. Paraphrased Query (tenant_a)", "Tell me the capital city of France.")

# 4. Same prompt, different tenant -> MUST MISS (tenant isolation!)
ask("4. Isolation Check (tenant_b)", "What is the capital of France?", tenant_id="tenant_b")