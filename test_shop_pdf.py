import httpx
from fpdf import FPDF

BASE = "http://localhost:8000"


def create_sample_menu_pdf(path: str = "scoop_and_byte_menu.pdf") -> str:
    """Generates a realistic 2-page Scoop & Byte Creamery PDF for testing."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1: Menu & Allergens
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Scoop & Byte Creamery - Official Menu & Allergens", ln=True)
    pdf.set_font("Helvetica", "", 11)
    page1_text = (
        "CLASSIC & SIGNATURE DAIRY FLAVORS:\n"
        "- Madagascar Bean Vanilla: Single $4.50, Double $7.00, Pint $10.50. Allergens: Dairy, Egg. Gluten-Free, Nut-Free.\n"
        "- Double Dark Fudge: Single $4.75, Double $7.25, Pint $11.00. Allergens: Dairy, Soy. Gluten-Free, Nut-Free.\n"
        "- Roasted Sicilian Pistachio: Single $5.50, Double $8.25, Pint $12.50. Allergens: Dairy, Tree Nuts (Pistachio).\n"
        "- Peanut Butter Circuit: Single $5.25, Double $7.95, Pint $12.00. Allergens: Dairy, Peanuts, Wheat/Gluten, Soy.\n\n"
        "PLANT-BASED / VEGAN FLAVORS:\n"
        "- Midnight Oat Fudge (Vegan): Single $5.25, Double $7.75, Pint $11.50. Allergens: Wheat/Gluten (brownies), Soy. Dairy-Free, Nut-Free, Peanut-Free.\n"
        "- Mango Coconut Sorbet (Vegan): Single $4.75, Double $7.00, Pint $10.50. Allergens: Coconut. Dairy-Free, Gluten-Free, Nut-Free, Peanut-Free.\n"
        "- Roasted Strawberry Almond (Vegan): Single $5.25, Double $7.75, Pint $11.50. Allergens: Tree Nuts (Almond). Dairy-Free, Gluten-Free, Peanut-Free.\n"
        "- Matcha Oat Swirl (Vegan): Single $5.50, Double $8.00, Pint $12.00. Allergens: None (Top-8 Allergen Free). Dairy-Free, Gluten-Free, Nut-Free, Peanut-Free.\n\n"
        "CONES & TOPPINGS:\n"
        "Paper Cup (Free), House Waffle Cone (+$1.25, wheat/dairy/egg), Gluten-Free Sugar Cone (+$1.00, vegan & nut-free).\n"
        "Allergy Safety Protocol: Upon request, staff open a fresh sealed tub and use a dedicated purple allergen-safe scoop."
    )
    pdf.multi_cell(0, 7, page1_text)

    # Page 2: Catering & Refunds
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Scoop & Byte Creamery - Catering & Policies", ln=True)
    pdf.set_font("Helvetica", "", 11)
    page2_text = (
        "CATERING PACKAGES (Minimum 15 guests, 48-hour notice required):\n"
        "1) Mini Pint Pack ($5.50 per guest): 15-25 guests, 2 flavors, self-serve.\n"
        "2) Classic Sundae Bar ($7.50 per guest): 20-60 guests, 3 flavors, 4 toppings, cups + waffle cones, free dry-ice cooler.\n"
        "3) Full-Service Scoop Cart ($10.00 per guest + $75 flat staffing fee for 2 hours): 30-150 guests, 5 flavors, dedicated scooper.\n\n"
        "ADD-ONS, DISCOUNTS & TAX:\n"
        "- Dedicated Allergen-Free Sealed Tub Setup: $20 flat fee.\n"
        "- Volume Discount: Orders over $250 receive a 10% discount before tax.\n"
        "- Sales Tax: 8% on all catering orders.\n\n"
        "CANCELLATION & REFUND POLICY:\n"
        "Cancellations made at least 48 hours before the event receive a 100% refund. "
        "Cancellations within 48 hours of the event receive a 50% store credit."
    )
    pdf.multi_cell(0, 7, page2_text)
    pdf.output(path)
    return path


# 1. Generate & Upload PDF as the Shop Owner
pdf_path = create_sample_menu_pdf()
print("=== 1. Owner Uploading PDF to /api/shop/upload ===")
with open(pdf_path, "rb") as f:
    up_resp = httpx.post(
        f"{BASE}/api/shop/upload",
        files={"file": (pdf_path, f, "application/pdf")},
        timeout=30.0,
    )
up_resp.raise_for_status()
print("  Upload Result:", up_resp.json())

# 2. Owner Warms Top Static FAQs into Redis
print("\n=== 2. Owner Warming Static FAQs ===")
faq_resp = httpx.post(
    f"{BASE}/api/shop/faqs/warm",
    json={
        "faqs": [
            {
                "question": "What are your opening hours?",
                "answer": "We are open Mon-Thu 12-10 PM, Fri-Sat 11 AM-11:30 PM, and Sunday 12-9 PM.",
            },
            {
                "question": "Where are you located?",
                "answer": "404 Silicon Plaza, Suite B, with free 2-hour parking behind the building.",
            },
        ]
    },
    timeout=30.0,
)
faq_resp.raise_for_status()
print("  FAQ Warm Result:", faq_resp.json())


def ask_shop(label: str, message: str):
    resp = httpx.post(f"{BASE}/api/shop/chat", json={"message": message}, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    x = data["xray"]
    sources = [f"{c['source']} (p.{c['page']})" for c in x["rag_chunks"]]
    print(f"\n=== {label} ===")
    print(f"  Cache Status : {x['cache_status']} (dist={x['cache_distance']})")
    print(f"  RAG Sources  : {sources}")
    print(f"  Routed To    : {x['routed_tier']} -> {x['provider']}/{x['model']} ({x['router_reason']})")
    print(f"  PII Scrubbed : {x['pii_redacted']} | Latency: {x['latency_ms']} ms")
    print(f"  Answer       : {data['answer'][:220].replace(chr(10), ' ')}...")


# 3. Test all 4 live paths
ask_shop("3A. Warmed FAQ Paraphrase (Cache Hit)", "When do you guys close on Sundays?")
ask_shop("3B. Allergen Lookup (PDF RAG + Fast)", "Which vegan flavors are peanut-free and under $5.50 for a single scoop?")
ask_shop("3C. Catering Math (PDF RAG + Strong)", "Plan a nut-free, dairy-free Classic Sundae Bar for 40 guests with the sealed tub setup, calculate the exact itemized cost including discount and 8% tax, and suggest 3 safe flavors.")
ask_shop("3D. PII Redaction + Refund Policy PDF", "My phone is 555-867-5309 and email is alex@example.com - what is the refund if I cancel catering 24 hours before?")