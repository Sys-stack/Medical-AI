import os
import base64
import json
import re
from urllib.parse import quote

import requests
from google import genai

# ── Config ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY)

OPENFDA_BASE = "https://api.fda.gov"
NLM_BASE = "https://rxnav.nlm.nih.gov/REST"

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — GEMINI (Prescription Extraction)
# ════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """
You are a STRICT medical prescription parser.

Return ONLY valid JSON.

FORMAT:
{
  "summary": "short summary",
  "drugs": ["drug names"]
}

Rules:
- No markdown
- No explanation
- Only real drug names
"""


def encode_image(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gemini(text=None, image_path=None):
    try:
        content = _EXTRACTION_PROMPT + "\n"

        if text:
            content += f"\nPrescription:\n{text}\n"

        if image_path:
            b64 = encode_image(image_path)
            content += f"\n[Image Base64]: {b64[:100]}...\n"

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=content,
            config={"temperature": 0.2},
        )

        raw = response.text

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {"summary": raw[:300], "drugs": []}

    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA FETCH (OpenFDA + NLM)
# ════════════════════════════════════════════════════════════════════════


def build_endpoints(drug_name: str):
    return {
        "openfda_url": f"{OPENFDA_BASE}/drug/label.json?search=openfda.generic_name:\"{drug_name}\"&limit=3",
        "nlm_url": f"{NLM_BASE}/rxcui.json?name={quote(drug_name)}&search=1",
    }


def fetch_openfda(url: str):
    if OPENFDA_API_KEY:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={OPENFDA_API_KEY}"

    res = requests.get(url, timeout=10)
    if res.status_code != 200:
        return {"error": res.text}

    raw = res.json()
    return {"parsed": parse_openfda(raw)}


def parse_openfda(raw):
    results = raw.get("results", [])
    parsed = []

    for r in results:
        openfda = r.get("openfda", {})
        parsed.append({
            "brand": openfda.get("brand_name", ["N/A"])[0],
            "generic": openfda.get("generic_name", ["N/A"])[0],
            "purpose": r.get("purpose", ["N/A"])[0] if r.get("purpose") else "N/A",
            "warnings": (r.get("warnings", ["N/A"])[0][:200] if r.get("warnings") else "N/A"),
        })

    return parsed


def fetch_nlm(url: str):
    res = requests.get(url, timeout=10)
    if res.status_code != 200:
        return {"error": "NLM failed"}

    raw = res.json()
    ids = raw.get("idGroup", {}).get("rxnormId", [])

    if not ids:
        return {"error": "No RxCUI"}

    rxcui = ids[0]
    props_url = f"{NLM_BASE}/rxcui/{rxcui}/allProperties.json?prop=all"
    props = requests.get(props_url).json()

    return {"parsed": parse_nlm(props, rxcui)}


def parse_nlm(raw, rxcui):
    props = raw.get("propConceptGroup", {}).get("propConcept", [])
    prop_map = {p["propName"]: p["propValue"] for p in props}

    return {
        "rxcui": rxcui,
        "name": prop_map.get("RxNorm Name", "N/A"),
        "synonym": prop_map.get("RxNorm Synonym", "N/A"),
    }


def data_fetch(endpoints):
    return {
        "openfda": fetch_openfda(endpoints.get("openfda_url")),
        "nlm": fetch_nlm(endpoints.get("nlm_url")),
    }


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — GEMINI (User-Friendly Response)
# ════════════════════════════════════════════════════════════════════════

_RESPONSE_PROMPT = """
You are a friendly health assistant.
Explain medical info in simple language.
Be clear and short.
Always add: consult a doctor.
"""


def call_gemini_for_response(user_message, summary, drug_details):
    context = ""

    for drug, data in drug_details.items():
        fda = data.get("openfda", {}).get("parsed", [])
        if fda:
            first = fda[0]
            context += f"{drug}: {first.get('purpose')} | Warning: {first.get('warnings')}\n"

    prompt = f"""
{_RESPONSE_PROMPT}

User: {user_message}
Summary: {summary}
Drugs:\n{context}
"""

    try:
        res = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={"temperature": 0.4},
        )
        return res.text
    except Exception as e:
        return str(e)


# ════════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════════


def process_prescription(text=None, image_path=None):
    step1 = call_gemini(text, image_path)

    if "error" in step1:
        return step1

    summary = step1.get("summary", "")
    drugs = step1.get("drugs", [])

    drug_details = {}
    for d in drugs:
        endpoints = build_endpoints(d)
        drug_details[d] = data_fetch(endpoints)

    response = call_gemini_for_response(text or "Explain this", summary, drug_details)

    return {
        "summary": summary,
        "drugs": drugs,
        "details": drug_details,
        "response": response,
    }