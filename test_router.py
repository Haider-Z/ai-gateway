import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy-key")


def run_test(label: str, prompt: str, model: str = "auto"):
    t0 = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={"X-Tenant-ID": "router_demo"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    h = raw.headers
    answer = raw.parse().choices[0].message.content

    print(f"\n[{label}]")
    print(f"  Cache Status : {h.get('x-cache-status')}")
    print(f"  Routed To    : {h.get('x-routed-tier')} -> {h.get('x-routed-provider')}/{h.get('x-routed-model')}")
    print(f"  Reason       : {h.get('x-router-reason')}")
    print(f"  Latency      : {elapsed_ms:.1f} ms")
    print(f"  Answer       : {answer[:110].replace(chr(10), ' ')}...")


# 1. Simple prompt -> should route to FAST tier (Gemini)
run_test(
    "1. Simple Grammar Task (auto)",
    "Fix the grammar in this sentence: 'Me and him goes to store yesterday.'",
)

# 2. Complex prompt -> should trigger heuristic or semantic STRONG tier (OpenAI)
run_test(
    "2. Complex System Design (auto)",
    "Design a distributed rate limiter using Redis and explain how to handle clock skew and race conditions.",
)

# 3. Explicit model override -> should bypass router completely
run_test(
    "3. Explicit Override (gpt-6-luna)",
    "Define TCP in one sentence.",
    model="gpt-6-luna",
)

# 4. Repeat of Simple prompt -> should hit Cache and skip the router!
run_test(
    "4. Repeat Simple Task (cache hit)",
    "Fix the grammar in this sentence: 'Me and him goes to store yesterday.'",
)