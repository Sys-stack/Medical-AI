import os
import json
import re
import base64
from urllib.parse import quote

import requests
from google import genai
from google.genai import types

# ── Config ─────────────────────────────────────────────────────────────
API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY_1"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3")
]

OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY")

API_KEYS = [k for k in API_KEYS if k]

if not API_KEYS:
    raise ValueError("No GEMINI API keys provided")


OPENFDA_BASE = "https://api.fda.gov"
NLM_BASE     = "https://rxnav.nlm.nih.gov/REST"

# ── Model fallback chain ────────────────────────────────────────────────
# Primary model first; if it hits a rate/quota limit the next is tried.
GEMINI_MODELS = [
    # 🔥 Latest high-performance
    "gemini-3-flash",

    # ⚡ Best balance
    "gemini-2.5-flash",

    # 🔄 Alias (keep LAST)
    "gemini-flash-latest"
]

def get_client(api_key):
    return genai.Client(api_key=api_key)

# ════════════════════════════════════════════════════════════════════════
# MARKDOWN → HTML  (so the chat bubble renders rich text, not raw symbols)
# ════════════════════════════════════════════════════════════════════════

def markdown_to_html(text: str) -> str:
    """
    Convert Gemini markdown output to clean HTML safe for innerHTML injection.
    Handles: headings (#–###), **bold**, *italic*, bullet/numbered lists,
    horizontal rules, inline code, and plain paragraphs.
    """
    if not text:
        return ""

    lines  = text.split("\n")
    html   = []
    in_ul  = False
    in_ol  = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html.append("</ul>")
            in_ul = False
        if in_ol:
            html.append("</ol>")
            in_ol = False

    def inline_fmt(s: str) -> str:
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        s = re.sub(r'__(.+?)__',     r'<strong>\1</strong>', s)
        s = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', s)
        s = re.sub(r'_(.+?)_',       r'<em>\1</em>', s)
        s = re.sub(r'`(.+?)`',       r'<code class="md-code">\1</code>', s)
        return s

    for line in lines:
        stripped = line.strip()

        # Horizontal rule
        if re.match(r'^[-*_]{3,}$', stripped):
            close_lists()
            html.append('<hr class="md-hr">')
            continue

        # Headings
        m = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if m:
            close_lists()
            level = min(len(m.group(1)) + 2, 6)
            html.append(f'<h{level} class="md-h">{inline_fmt(m.group(2))}</h{level}>')
            continue

        # Unordered list
        m = re.match(r'^[-*+]\s+(.+)', stripped)
        if m:
            if in_ol:
                html.append("</ol>"); in_ol = False
            if not in_ul:
                html.append('<ul class="md-ul">'); in_ul = True
            html.append(f'<li>{inline_fmt(m.group(1))}</li>')
            continue

        # Ordered list
        m = re.match(r'^\d+[.)]\s+(.+)', stripped)
        if m:
            if in_ul:
                html.append("</ul>"); in_ul = False
            if not in_ol:
                html.append('<ol class="md-ol">'); in_ol = True
            html.append(f'<li>{inline_fmt(m.group(1))}</li>')
            continue

        # Blank line
        if stripped == "":
            close_lists()
            continue

        # Paragraph
        close_lists()
        html.append(f'<p class="md-p">{inline_fmt(stripped)}</p>')

    close_lists()
    return "\n".join(html)


# ════════════════════════════════════════════════════════════════════════
# GEMINI CALL WITH MODEL FALLBACK
# ════════════════════════════════════════════════════════════════════════

def _generate_with_fallback(
    contents,
    temperature: float = 0.4,
    system_instruction: str = None,
) -> str:
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )

    last_err = None

    for api_key in API_KEYS:
        client = get_client(api_key)

        for model in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""

            except Exception as exc:
                err_str = str(exc).lower()

                # If quota issue → try next model / key
                if any(kw in err_str for kw in ("429", "quota", "rate", "resource exhausted", "limit")):
                    last_err = exc
                    continue

                # Other errors → skip model but keep same key
                if any(kw in err_str for kw in ("not found", "404", "unsupported")):
                    continue

                # Critical error → stop everything
                raise

    raise RuntimeError(f"All API keys and models exhausted. Last error: {last_err}")


# ════════════════════════════════════════════════════════════════════════
# STEP 1 — GEMINI  (Prescription Extraction — text or image)
# ════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """
You are a STRICT medical prescription parser.

Return ONLY valid JSON — no markdown, no explanation.

FORMAT:
{
  "summary": "short summary of the prescription",
  "drugs": ["drug name 1", "drug name 2"],
  "patient_type": "human or pet"
}

Rules:
- Only real drug / medicine names in the drugs list.
- If no drugs are found, return an empty list for drugs.
- Set patient_type to "pet" if you detect veterinary language (animal species mentioned,
  vet clinic header, dosing "per kg body weight for dogs/cats", etc.).
  Default to "human" if unclear.
"""


def call_gemini(text: str = "", image_b64: str = None, image_mime: str = "image/jpeg") -> dict:
    """
    Step 1 — extract { summary, drugs, patient_type }.
    Accepts plain text and/or a base64-encoded prescription image.
    """
    try:
        parts: list = [_EXTRACTION_PROMPT]

        if image_b64:
            parts.append(
                types.Part.from_bytes(
                    data      = base64.b64decode(image_b64),
                    mime_type = image_mime,
                )
            )

        if text:
            parts.append(f"\nPrescription text:\n{text}\n")

        raw   = _generate_with_fallback(parts, temperature=0.2)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {"summary": raw[:300], "drugs": [], "patient_type": "human"}

    except Exception as exc:
        return {"error": str(exc)}


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA FETCH  (OpenFDA + NLM)
# ════════════════════════════════════════════════════════════════════════

def build_endpoints(drug_name: str, patient_type: str = "human") -> dict:
    if patient_type == "pet":
        openfda_url = (
            f"{OPENFDA_BASE}/animalandveterinary/event.json"
            f"?search=drug.medicinalproduct:\"{quote(drug_name)}\"&limit=3"
        )
    else:
        openfda_url = (
            f"{OPENFDA_BASE}/drug/label.json"
            f"?search=openfda.generic_name:\"{drug_name}\"&limit=3"
        )
    return {
        "openfda_url":  openfda_url,
        "nlm_url":      f"{NLM_BASE}/rxcui.json?name={quote(drug_name)}&search=1",
        "patient_type": patient_type,
    }


def fetch_openfda(url: str, patient_type: str = "human") -> dict:
    if OPENFDA_API_KEY:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={OPENFDA_API_KEY}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return {"error": res.text[:200]}
        parser = parse_openfda_vet if patient_type == "pet" else parse_openfda
        return {"parsed": parser(res.json())}
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


def parse_openfda_vet(raw: dict) -> list:
    results = raw.get("results", [])
    parsed  = []
    for r in results:
        drug = (r.get("drug") or [{}])[0]
        parsed.append({
            "brand":    drug.get("medicinalproduct", "N/A"),
            "generic":  drug.get("activeingredient", "N/A"),
            "purpose":  "Veterinary use",
            "warnings": "Always follow your veterinarian's instructions.",
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
    pt = endpoints.get("patient_type", "human")
    return {
        "openfda": fetch_openfda(endpoints.get("openfda_url", ""), patient_type=pt),
        "nlm":     fetch_nlm(endpoints.get("nlm_url", "")),
    }


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — GEMINI  (Patient-Friendly Response)
# ════════════════════════════════════════════════════════════════════════

_RESPONSE_SYSTEM_HUMAN = (
    "You are Cura, a friendly health assistant. "
    "Explain medical information in simple, clear language a patient can understand. "
    "Use markdown: ## headings, **bold** key terms, bullet lists. "
    "Keep the reply concise and well-structured. Always remind the user to consult a doctor."

        """Rules:
            The system should generate a professional wellness report including:
        •  Individual Profile Summary
        •  Key Observations from Assessment
        •  Interpretation of Findings in Simple Language
        •  Identified Risk Indicators
        •  Wellness Insights
        •  Personalized Recommendations
        •  Preventive Lifestyle Suggestions
        Note: if its unrelated, do not give an output. Instead say something like Not my area of expertise """
)

_RESPONSE_SYSTEM_PET = (
    "You are Cura, a friendly veterinary health assistant. "
    "Explain animal medication information clearly for a pet owner. "
    "Use markdown: ## headings, **bold** key terms, bullet lists. "
    "Keep the reply concise. Always remind the owner to follow their veterinarian's instructions."

        """Rules:
            The system should generate a professional wellness report including:
        •  Individual Profile Summary
        •  Key Observations from Assessment
        •  Interpretation of Findings in Simple Language
        •  Identified Risk Indicators
        •  Wellness Insights
        •  Personalized Recommendations
        •  Preventive Lifestyle Suggestions
        Note: if its unrelated, do not give an output. Instead say something like Not my area of expertise"""
        
)


def call_gemini_for_response(
    user_message: str,
    summary: str,
    drug_details: dict,
    patient_type: str = "human",
) -> str:
    """Step 3 — patient-friendly HTML response."""
    drug_context = ""
    for drug, data in drug_details.items():
        fda = data.get("openfda", {}).get("parsed", [])
        if fda:
            first = fda[0]
            drug_context += (
                f"• {drug}: purpose — {first.get('purpose')} | "
                f"warning — {first.get('warnings')}\n"
            )

    user_turn = (
        f"User question / message: {user_message or 'Please explain this prescription.'}\n\n"
        f"Prescription summary: {summary}\n\n"
        f"Drug information:\n{drug_context or 'No drug data available.'}"
    )

    sys = _RESPONSE_SYSTEM_PET if patient_type == "pet" else _RESPONSE_SYSTEM_HUMAN

    try:
        raw = _generate_with_fallback(user_turn, temperature=0.4, system_instruction=sys)
        return markdown_to_html(raw or "No response generated.")
    except Exception as exc:
        return f"<p>Error generating response: {exc}</p>"


# ════════════════════════════════════════════════════════════════════════
# STEP 4 — GEMINI  (Health-Progress Comparison)
# ════════════════════════════════════════════════════════════════════════

_COMPARISON_SYSTEM = (
    "You are Cura, a health-progress analyst. "
    "You will be given a chronological timeline of a patient's health conversations. "
    "Analyse the progression: highlight improvements, recurring issues, and concerns. "
    "Use markdown: ## headings, **bold** key terms, bullet lists. "
    "Write clearly for a non-medical reader. Always recommend consulting a healthcare professional."

        """Rules:
            The system should generate a professional wellness report including:
        •  Individual Profile Summary
        •  Key Observations from Assessment
        •  Interpretation of Findings in Simple Language
        •  Identified Risk Indicators
        •  Wellness Insights
        •  Personalized Recommendations
        •  Preventive Lifestyle Suggestions"""

)


def call_gemini_for_comparison(timeline: str) -> str:
    """Step 4 — health-progress HTML report."""
    user_turn = (
        "Here is the patient's health conversation timeline in chronological order:\n\n"
        f"{timeline}\n\n"
        "Please analyse the progression of the patient's condition and write a clear report."
    )
    try:
        raw = _generate_with_fallback(user_turn, temperature=0.4, system_instruction=_COMPARISON_SYSTEM)
        return markdown_to_html(raw or "No comparison report generated.")
    except Exception as exc:
        return f"<p>Error generating comparison: {exc}</p>"


# ════════════════════════════════════════════════════════════════════════
# STEP 5 — GEMINI  (Body-Map Data Extraction)
# ════════════════════════════════════════════════════════════════════════

_BODY_MAP_SYSTEM = """
You are a STRICT medical data extractor for a body-map visualisation feature.

Return ONLY valid JSON — no markdown, no explanation, no code fences.

FORMAT:
{
  "body_part": "<one of the exact IDs listed below, or null>",
  "condition": "<short plain-english condition name, e.g. 'Pneumonia', 'Sinusitis'>",
  "severity": <float 0.0–1.0, where 1.0 = severe / newly diagnosed, 0.0 = fully healed>,
  "treatment_days": <integer total days of the treatment course, or null if unknown>,
  "days_elapsed": <integer days already taken/elapsed, or 0 if just started>,
  "show_map": <true if a specific anatomical location can be identified, false otherwise>
}

VALID body_part IDs (use exactly one, or null):
  "head", "neck", "left_lung", "right_lung", "heart", "stomach", "liver",
  "left_kidney", "right_kidney", "intestines", "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow", "left_hand", "right_hand", "left_knee", "right_knee",
  "left_foot", "right_foot", "spine", "throat", "nose", "left_ear", "right_ear",
  "eye_left", "eye_right", "skin", "bladder"

Rules:
- Set show_map=false if the message is a general question (no specific body part / diagnosis).
- severity must represent current state: if treatment_days and days_elapsed are known,
  calculate severity = max(0.0, 1.0 - days_elapsed / treatment_days).
- If days_elapsed >= treatment_days, set severity=0.05 (nearly healed visual).
- Never invent information. If uncertain, set show_map=false.

"""


def call_gemini_for_body_map(prompt: str, summary: str, drug_details: dict) -> dict:
    """
    Step 5 — extract body-map data from the prescription context.
    Returns a dict with keys: body_part, condition, severity, treatment_days,
    days_elapsed, show_map.
    Falls back to {"show_map": false} on any error.
    """
    drug_context = ""
    for drug, data in drug_details.items():
        fda = data.get("openfda", {}).get("parsed", [])
        if fda:
            first = fda[0]
            drug_context += f"• {drug}: purpose — {first.get('purpose')}\n"

    user_turn = (
        f"User message: {prompt or '(none)'}\n\n"
        f"Prescription summary: {summary or '(none)'}\n\n"
        f"Drug information:\n{drug_context or 'None available.'}"
    )

    try:
        raw   = _generate_with_fallback(user_turn, temperature=0.1, system_instruction=_BODY_MAP_SYSTEM)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            # Ensure numeric bounds
            parsed["severity"]     = max(0.0, min(1.0, float(parsed.get("severity", 0.5))))
            parsed["show_map"]     = bool(parsed.get("show_map", False))
            parsed["treatment_days"] = parsed.get("treatment_days")
            parsed["days_elapsed"] = int(parsed.get("days_elapsed", 0))
            return parsed
    except Exception:
        pass

    return {"show_map": False}


# ════════════════════════════════════════════════════════════════════════
# STEP 6 — GEMINI  (Tracking Dashboard Insights)
# ════════════════════════════════════════════════════════════════════════

_TRACKING_INSIGHTS_SYSTEM = (
    "You are Cura, a health tracking analyst. "
    "You will be given a structured summary of a patient's entire medication and condition history. "
    "Write 3–5 concise, actionable insight bullets for the patient's dashboard. "
    "Cover: overall recovery trend, any medication adherence notes, upcoming milestones, and one wellness tip. "
    "Use markdown: **bold** key terms, bullet list only. "
    "Keep it motivational and easy to read. Always recommend consulting a healthcare professional for medical decisions."
        """Rules:
            The system should generate a professional wellness report including:
        •  Individual Profile Summary
        •  Key Observations from Assessment
        •  Interpretation of Findings in Simple Language
        •  Identified Risk Indicators
        •  Wellness Insights
        •  Personalized Recommendations
        •  Preventive Lifestyle Suggestions
        """
        
)


def call_gemini_for_tracking_insights(tracking_summary: str) -> str:
    """
    Step 6 — Generate dashboard insight bullets from the patient's full tracking history.
    tracking_summary: plain-text summary built by the /tracking-data route.
    Returns HTML string (via markdown_to_html).
    """
    user_turn = (
        "Here is the patient's current medication and condition tracking summary:\n\n"
        f"{tracking_summary}\n\n"
        "Please generate 3–5 actionable insight bullets for their health dashboard."
    )
    try:
        raw = _generate_with_fallback(user_turn, temperature=0.45, system_instruction=_TRACKING_INSIGHTS_SYSTEM)
        return markdown_to_html(raw or "No insights available at this time.")
    except Exception as exc:
        return f"<p>Could not generate insights: {exc}</p>"


# ════════════════════════════════════════════════════════════════════════
# CONVENIENCE PIPELINE
# ════════════════════════════════════════════════════════════════════════

def process_prescription(text: str = "", image_b64: str = None, image_mime: str = "image/jpeg") -> dict:
    """Full pipeline: extract → fetch → reply. Returns dict with summary/drugs/details/response."""
    step1 = call_gemini(text=text, image_b64=image_b64, image_mime=image_mime)
    if "error" in step1:
        return step1

    summary      = step1.get("summary", "")
    drugs        = step1.get("drugs", [])
    patient_type = step1.get("patient_type", "human")
    drug_details = {}

    for drug in drugs:
        endpoints          = build_endpoints(drug, patient_type=patient_type)
        drug_details[drug] = data_fetch(endpoints)

    reply = call_gemini_for_response(
        user_message = text or "Explain this prescription.",
        summary      = summary,
        drug_details = drug_details,
        patient_type = patient_type,
    )

    return {
        "summary":      summary,
        "drugs":        drugs,
        "patient_type": patient_type,
        "details":      drug_details,
        "response":     reply,
    }
