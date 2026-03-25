import os
import json
import re
from urllib.parse import quote

import requests
from google import genai
from google.genai import types

# ── Config ─────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY")
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY)

OPENFDA_BASE = "https://api.fda.gov"
NLM_BASE     = "https://rxnav.nlm.nih.gov/REST"

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — GEMINI  (Prescription Extraction)
# ════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """
You are a STRICT medical prescription parser.

Return ONLY valid JSON — no markdown, no explanation.

FORMAT:
{
  "summary": "short summary of the prescription",
  "drugs": ["drug name 1", "drug name 2"]
}

Rules:
- Only real drug / medicine names in the drugs list.
- If no drugs are found, return an empty list for drugs.
"""


def call_gemini(text: str) -> dict:
    """
    Step 1 — extract a structured { summary, drugs } dict from a prescription.

    Accepts plain text.
    Returns a dict with keys 'summary' and 'drugs' on success,
    or { 'error': <message> } on failure.
    """
    try:
        parts: list = [_EXTRACTION_PROMPT, f"\nPrescription text:\n{text}\n"]

        response = client.models.generate_content(
            model    = "gemini-3-flash-preview",
            contents = parts,
            config   = types.GenerateContentConfig(temperature=0.2),
        )

        raw = response.text or ""

        # Strip any accidental markdown fences.
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())

        # Fallback: return whatever the model said as the summary.
        return {"summary": raw[:300], "drugs": []}

    except Exception as exc:
        return {"error": str(exc)}


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA FETCH  (OpenFDA + NLM via requests — external REST APIs)
# ════════════════════════════════════════════════════════════════════════

def build_endpoints(drug_name: str) -> dict:
    return {
        "openfda_url": (
            f"{OPENFDA_BASE}/drug/label.json"
            f"?search=openfda.generic_name:\"{drug_name}\"&limit=3"
        ),
        "nlm_url": f"{NLM_BASE}/rxcui.json?name={quote(drug_name)}&search=1",
    }


def fetch_openfda(url: str) -> dict:
    if OPENFDA_API_KEY:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={OPENFDA_API_KEY}"

    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return {"error": res.text[:200]}
        return {"parsed": parse_openfda(res.json())}
    except Exception as exc:
        return {"error": str(exc)}


def parse_openfda(raw: dict) -> list:
    results = raw.get("results", [])
    parsed  = []
    for r in results:
        openfda = r.get("openfda", {})
        parsed.append({
            "brand":    openfda.get("brand_name",   ["N/A"])[0],
            "generic":  openfda.get("generic_name", ["N/A"])[0],
            "purpose":  r.get("purpose",  ["N/A"])[0] if r.get("purpose")  else "N/A",
            "warnings": r.get("warnings", ["N/A"])[0][:200] if r.get("warnings") else "N/A",
        })
    return parsed


def fetch_nlm(url: str) -> dict:
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return {"error": "NLM request failed"}

        ids = res.json().get("idGroup", {}).get("rxnormId", [])
        if not ids:
            return {"error": "No RxCUI found"}

        rxcui     = ids[0]
        props_url = f"{NLM_BASE}/rxcui/{rxcui}/allProperties.json?prop=all"
        props_res = requests.get(props_url, timeout=10)
        if props_res.status_code != 200:
            return {"error": "NLM properties request failed"}

        return {"parsed": parse_nlm(props_res.json(), rxcui)}
    except Exception as exc:
        return {"error": str(exc)}


def parse_nlm(raw: dict, rxcui: str) -> dict:
    props    = raw.get("propConceptGroup", {}).get("propConcept", [])
    prop_map = {p["propName"]: p["propValue"] for p in props}
    return {
        "rxcui":   rxcui,
        "name":    prop_map.get("RxNorm Name",    "N/A"),
        "synonym": prop_map.get("RxNorm Synonym", "N/A"),
    }


def data_fetch(endpoints: dict) -> dict:
    return {
        "openfda": fetch_openfda(endpoints.get("openfda_url", "")),
        "nlm":     fetch_nlm(endpoints.get("nlm_url", "")),
    }


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — GEMINI  (Patient-Friendly Response)
# ════════════════════════════════════════════════════════════════════════

_RESPONSE_SYSTEM = (
    "You are Cura, a friendly health assistant. "
    "Explain medical information in simple, clear language a patient can understand. "
    "Keep the reply concise. Always remind the user to consult a doctor."
)


def call_gemini_for_response(
    user_message: str,
    summary: str,
    drug_details: dict,
) -> str:
    """
    Step 3 — produce a patient-friendly explanation of the prescription.
    """
    drug_context = ""
    for drug, data in drug_details.items():
        fda = data.get("openfda", {}).get("parsed", [])
        if fda:
            first        = fda[0]
            drug_context += (
                f"• {drug}: purpose — {first.get('purpose')} | "
                f"warning — {first.get('warnings')}\n"
            )

    user_turn = (
        f"User question / message: {user_message or 'Please explain this prescription.'}\n\n"
        f"Prescription summary: {summary}\n\n"
        f"Drug information:\n{drug_context or 'No drug data available.'}"
    )

    try:
        response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=user_turn,
        config=types.GenerateContentConfig(
            temperature=0.4,
            system_instruction=_RESPONSE_SYSTEM
        ),
    )
        return response.text or "No response generated."
    except Exception as exc:
        return f"Error generating response: {exc}"


# ════════════════════════════════════════════════════════════════════════
# STEP 4 — GEMINI  (Health-Progress Comparison)
# ════════════════════════════════════════════════════════════════════════

_COMPARISON_SYSTEM = (
    "You are Cura, a health-progress analyst. "
    "You will be given a chronological timeline of a patient's health conversations. "
    "Analyse the progression: highlight improvements, recurring issues, and any concerns. "
    "Write clearly so a non-medical reader can understand. "
    "Always recommend consulting a healthcare professional."
)


def call_gemini_for_comparison(timeline: str) -> str:
    """
    Step 4 — compare multiple past conversations and produce a health-progress report.

    Parameters
    ----------
    timeline : str
        Plain-text chronological timeline built by classes.build_timeline().

    Returns
    -------
    str — the AI-generated health-progress report.
    """
    user_turn = (
        "Here is the patient's health conversation timeline in chronological order:\n\n"
        f"{timeline}\n\n"
        "Please analyse the progression of the patient's condition and write a clear report."
    )

    try:
        response = client.models.generate_content(
            model    = "gemini-1.5-flash",
            contents = [
                types.Content(
                    role  = "user",
                    parts = [
                        types.Part.from_text(_COMPARISON_SYSTEM),
                        types.Part.from_text(user_turn),
                    ],
                )
            ],
            config = types.GenerateContentConfig(temperature=0.3),
        )
        return response.text or "No comparison report generated."
    except Exception as exc:
        return f"Error generating comparison: {exc}"


# ════════════════════════════════════════════════════════════════════════
# CONVENIENCE PIPELINE  (used by mainbackend.py's /chat route)
# ════════════════════════════════════════════════════════════════════════

def process_prescription(text: str) -> dict:
    """
    Full pipeline: extract → fetch drug data → generate reply.
    Returns a dict with keys: summary, drugs, details, response.
    """
    step1 = call_gemini(text)
    if "error" in step1:
        return step1

    summary      = step1.get("summary", "")
    drugs        = step1.get("drugs", [])
    drug_details = {}

    for drug in drugs:
        endpoints          = build_endpoints(drug)
        drug_details[drug] = data_fetch(endpoints)

    reply = call_gemini_for_response(
        user_message = text or "Explain this prescription.",
        summary      = summary,
        drug_details = drug_details,
    )

    return {
        "summary":  summary,
        "drugs":    drugs,
        "details":  drug_details,
        "response": reply,
    }
