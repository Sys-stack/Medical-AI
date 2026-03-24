import requests
import base64
import json
from urllib.parse import quote
import os

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY")   # Replace with your key
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY")  # Replace with your key

GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
OPENFDA_BASE = "https://api.fda.gov"
NLM_BASE     = "https://rxnav.nlm.nih.gov/REST"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GEMINI  (prescription reader)
# Reads prescription text/image → returns summary + drug names
# ══════════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """
You are a medical prescription analyzer.

Given a prescription (text or image), you must respond ONLY in this JSON format with no extra text:

{
  "summary": "Plain language summary of the full prescription — what condition is being treated, what drugs are prescribed, dosage, and how long to take them",
  "drugs": ["drug_name_1", "drug_name_2"]
}

Rules:
- summary: clear, simple explanation a patient can understand — include condition, drug names, dosage, frequency, and duration if available
- drugs: list of generic drug names found in the prescription (lowercase)
- Return ONLY the JSON, no markdown, no explanation
"""


def encode_image(image_path: str) -> tuple[str, str]:
    """Converts image file to base64. Returns (base64_string, mime_type)."""
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
    }
    ext       = "." + image_path.split(".")[-1].lower()
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime_type


def call_gemini(text: str = None, image_path: str = None) -> dict:
    """
    Step 1 — Calls Gemini with a prescription (text and/or image).
    Returns: { summary: str, drugs: list[str] }
    On error returns: { error: str }
    """
    parts = [{"text": _EXTRACTION_PROMPT}]

    if text:
        parts.append({"text": text})

    if image_path:
        b64, mime = encode_image(image_path)
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data":      b64,
            }
        })

    payload  = {"contents": [{"parts": parts}]}
    print("[Gemini] Sending prescription to Gemini for extraction...")
    response = requests.post(GEMINI_URL, json=payload, timeout=30)

    if response.status_code != 200:
        return {"error": f"Gemini error {response.status_code}: {response.text[:200]}"}

    raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    clean    = raw_text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        print("[Gemini RAW OUTPUT]", clean)

        # fallback: extract manually
        return {
            "summary": clean[:500],
            "drugs": []
        }


def build_endpoints(drug_name: str) -> dict:
    """
    Builds OpenFDA and NLM URLs for a given drug name.
    Output feeds directly into data_fetch().
    """
    return {
        "openfda_url": f"{OPENFDA_BASE}/drug/label.json?search=openfda.generic_name:\"{drug_name}\"&limit=3",
        "nlm_url":     f"{NLM_BASE}/rxcui.json?name={quote(drug_name)}&search=1",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA FETCH
# Takes endpoints from Step 1 → hits OpenFDA + NLM → returns drug details
# ══════════════════════════════════════════════════════════════════════════════

def fetch_openfda(url: str) -> dict:
    """Fetches drug data from OpenFDA. Returns: { raw, parsed }"""
    if "api_key" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={OPENFDA_API_KEY}"

    print(f"[OpenFDA] GET {url}")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return {
            "raw":    None,
            "parsed": None,
            "error":  f"OpenFDA error {response.status_code}: {response.text[:200]}",
        }

    raw    = response.json()
    parsed = parse_openfda(raw)
    return {"raw": raw, "parsed": parsed}


def parse_openfda(raw: dict) -> list[dict]:
    """Extracts useful fields from OpenFDA label results."""
    results = raw.get("results", [])
    parsed  = []

    for r in results:
        openfda = r.get("openfda", {})
        parsed.append({
            "brand_name":        openfda.get("brand_name",        ["N/A"])[0],
            "generic_name":      openfda.get("generic_name",      ["N/A"])[0],
            "manufacturer":      openfda.get("manufacturer_name", ["N/A"])[0],
            "purpose":           r.get("purpose",                 ["N/A"])[0] if r.get("purpose")                 else "N/A",
            "active_ingredient": r.get("active_ingredient",       ["N/A"])[0] if r.get("active_ingredient")       else "N/A",
            "warnings":          r.get("warnings",                ["N/A"])[0][:300] if r.get("warnings")          else "N/A",
            "dosage":            r.get("dosage_and_administration",["N/A"])[0][:300] if r.get("dosage_and_administration") else "N/A",
        })

    return parsed


def fetch_nlm(url: str) -> dict:
    """Fetches drug data from NLM RxNav. Returns: { raw, parsed }"""
    print(f"[NLM] GET {url}")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return {"raw": None, "parsed": None, "error": f"NLM error {response.status_code}"}

    raw = response.json()
    ids = raw.get("idGroup", {}).get("rxnormId", [])

    if not ids:
        return {"raw": raw, "parsed": None, "error": "No RxCUI found"}

    rxcui      = ids[0]
    drug_name  = url.split("name=")[-1].split("&")[0]
    props_url  = f"{NLM_BASE}/rxcui/{rxcui}/allProperties.json?prop=all"
    props_resp = requests.get(props_url, timeout=10)

    if props_resp.status_code != 200:
        return {"raw": raw, "parsed": None, "error": "Could not fetch NLM properties"}

    props_raw = props_resp.json()
    parsed    = parse_nlm(props_raw, drug_name, rxcui)
    return {"raw": props_raw, "parsed": parsed}


def parse_nlm(raw: dict, drug_name: str, rxcui: str) -> dict:
    """Extracts useful fields from NLM RxNav properties response."""
    props    = raw.get("propConceptGroup", {}).get("propConcept", [])
    prop_map = {p["propName"]: p["propValue"] for p in props}

    return {
        "drug_name": drug_name,
        "rxcui":     rxcui,
        "synonym":   prop_map.get("RxNorm Synonym", "N/A"),
        "tty":       prop_map.get("TTY", "N/A"),
        "full_name": prop_map.get("RxNorm Name", drug_name),
    }


def data_fetch(endpoints: dict) -> dict:
    """
    Receives endpoints dict from build_endpoints().
    Hits OpenFDA + NLM and returns combined results.
    Returns: { openfda: { raw, parsed }, nlm: { raw, parsed } }
    """
    openfda_url = endpoints.get("openfda_url", "")
    nlm_url     = endpoints.get("nlm_url",     "")

    return {
        "openfda": fetch_openfda(openfda_url) if openfda_url else {"raw": None, "parsed": None, "error": "No URL"},
        "nlm":     fetch_nlm(nlm_url)         if nlm_url     else {"raw": None, "parsed": None, "error": "No URL"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GEMINI  (layman's response generator)
# Takes the user message + enriched drug data → generates a plain-language reply
# ══════════════════════════════════════════════════════════════════════════════

_RESPONSE_SYSTEM_PROMPT = """
You are Cura, a friendly and empathetic health assistant.

Your job is to translate complex medical findings, prescriptions, and research into
clear, simple language that anyone can understand — no medical degree required.

Guidelines:
- Use plain, everyday language. Avoid jargon.
- Be warm, reassuring, and supportive.
- Structure your response with short paragraphs (no bullet walls).
- Always end with a gentle reminder to consult a qualified doctor for personal medical advice.
- Do NOT diagnose or prescribe. You explain; you do not treat.
"""


def call_gemini_for_response(user_message: str, summary: str, drug_details: dict) -> str:
    """
    Step 3 — Second Gemini call.
    Combines the user's original question, Gemini's prescription summary,
    and the enriched OpenFDA / NLM data into a warm, layman-friendly reply.

    Returns: plain-text response string.
    """
    # Build a compact context block from the fetched drug data
    drug_context_lines = []
    for drug_name, details in drug_details.items():
        fda  = details.get("openfda", {})
        nlm  = details.get("nlm",     {})

        fda_parsed = fda.get("parsed") or []
        nlm_parsed = nlm.get("parsed") or {}

        drug_context_lines.append(f"Drug: {drug_name}")

        if fda_parsed:
            first = fda_parsed[0]
            drug_context_lines.append(f"  Brand name  : {first.get('brand_name', 'N/A')}")
            drug_context_lines.append(f"  Purpose     : {first.get('purpose', 'N/A')}")
            drug_context_lines.append(f"  Dosage      : {first.get('dosage', 'N/A')}")
            drug_context_lines.append(f"  Warnings    : {first.get('warnings', 'N/A')}")

        if nlm_parsed:
            drug_context_lines.append(f"  RxNorm name : {nlm_parsed.get('full_name', 'N/A')}")
            drug_context_lines.append(f"  Synonym     : {nlm_parsed.get('synonym', 'N/A')}")

        drug_context_lines.append("")

    drug_context = "\n".join(drug_context_lines) if drug_context_lines else "No drug data available."

    user_content = f"""
The user asked: "{user_message}"

Prescription summary (extracted by AI):
{summary}

Drug data from official sources (OpenFDA + NLM):
{drug_context}

Please respond to the user in plain, friendly language. Explain what the prescription means,
what each drug does, what to watch out for, and any important dosage or warning information —
all in terms a non-medical person can easily understand.
"""

    parts   = [{"text": _RESPONSE_SYSTEM_PROMPT}, {"text": user_content}]
    payload = {"contents": [{"parts": parts}]}

    print("[Gemini] Generating layman's response...")
    response = requests.post(GEMINI_URL, json=payload, timeout=30)

    if response.status_code != 200:
        return f"I was unable to generate a response right now (Gemini error {response.status_code}). Please try again."

    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON — GEMINI  (health-progress analyser)
# Takes a chronological timeline string → returns a progress report
# ══════════════════════════════════════════════════════════════════════════════

_COMPARISON_SYSTEM_PROMPT = """
You are Cura, a caring health assistant reviewing a patient's conversation history.

Given a chronological set of health conversations, analyse how the patient's condition
has progressed over time. Your report should:

1. Briefly summarise what the patient discussed in each conversation.
2. Highlight any improvements or positive trends.
3. Flag any recurring or worsening issues.
4. Note anything that might warrant attention from a clinician.
5. Close with an encouraging, supportive message.

Write in clear, compassionate plain language. Use short paragraphs.
Always remind the reader that this analysis is not a substitute for professional medical advice.
"""


def call_gemini_for_comparison(timeline: str) -> str:
    """
    Comparison call — analyses a patient's health timeline across multiple conversations.
    timeline: output of build_timeline() from classes.py
    Returns: plain-text progress report string.
    """
    user_content = f"""
Here are the patient's health conversations in chronological order:

{timeline}

Please analyse the progression of this patient's health and provide a clear, supportive summary.
"""

    parts   = [{"text": _COMPARISON_SYSTEM_PROMPT}, {"text": user_content}]
    payload = {"contents": [{"parts": parts}]}

    print("[Gemini] Generating health comparison report...")
    response = requests.post(GEMINI_URL, json=payload, timeout=30)

    if response.status_code != 200:
        return f"Could not generate comparison (Gemini error {response.status_code}). Please try again."

    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE ENTRY POINT  (standalone / testing use)
# ══════════════════════════════════════════════════════════════════════════════

def process_prescription(text: str = None, image_path: str = None) -> dict:
    """
    Full pipeline entry point for standalone testing.

    Flow:
      Step 1 — call_gemini()          → summary + drug names
      Step 2 — data_fetch()           → OpenFDA + NLM drug details
      Step 3 — call_gemini_for_response() → patient-friendly reply

    Returns:
      {
        "summary"  : str,
        "response" : str,
        "drugs"    : { drug_name: { openfda: {...}, nlm: {...} } }
      }
    """
    # Step 1
    gemini_result = call_gemini(text=text, image_path=image_path)
    if "error" in gemini_result:
        return {"error": gemini_result["error"]}

    summary   = gemini_result.get("summary", "")
    drug_list = gemini_result.get("drugs",   [])

    print(f"[Pipeline] Summary     : {summary}")
    print(f"[Pipeline] Drugs found : {drug_list}")

    # Step 2
    drug_details: dict = {}
    for drug in drug_list:
        endpoints          = build_endpoints(drug)
        drug_details[drug] = data_fetch(endpoints)

    # Step 3
    response_text = call_gemini_for_response(
        user_message = text or "Please explain my prescription.",
        summary      = summary,
        drug_details = drug_details,
    )

    return {
        "summary":  summary,
        "response": response_text,
        "drugs":    drug_details,
    }
