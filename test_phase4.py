import json
import httpx
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy-key")

print("--- 1. PII Redaction Guardrail Test ---")
raw1 = client.chat.completions.with_raw_response.create(
    model="auto",
    messages=[
        {
            "role": "user",
            "content": "Repeat back my contact details: email john.doe@example.com and phone 555-867-5309.",
        }
    ],
    extra_headers={"X-Tenant-ID": "phase4_demo"},
)
print("  X-PII-Redacted :", raw1.headers.get("x-pii-redacted"))
print("  Answer         :", raw1.parse().choices[0].message.content)

print("\n--- 2. Automatic Cross-Provider Failover Test (Simulated Gemini 503) ---")
raw2 = client.chat.completions.with_raw_response.create(
    model="auto",
    messages=[{"role": "user", "content": "Define HTTP 503 in one short sentence."}],
    extra_headers={
        "X-Tenant-ID": "phase4_demo",
        "X-Simulate-Failure": "true",
        "Cache-Control": "no-cache",
    },
)
print("  X-Failover-Used  :", raw2.headers.get("x-failover-used"))
print("  Served By        :", f"{raw2.headers.get('x-routed-provider')}/{raw2.headers.get('x-routed-model')}")
print("  Answer           :", raw2.parse().choices[0].message.content)

print("\n--- 3. Live PostgreSQL Telemetry Summary (/api/telemetry) ---")
stats = httpx.get("http://localhost:8000/api/telemetry").json()
print(json.dumps({k: v for k, v in stats.items() if k != "recent_requests"}, indent=2))