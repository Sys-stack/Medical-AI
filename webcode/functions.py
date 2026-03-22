import requests
import json
from urllib.parse import urlencode, quote
import os

# ── Config ────────────────────────────────────────────────────────────────────
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY")
OPENFDA_BASE    = "https://api.fda.gov"
NLM_BASE        = "https://rxnav.nlm.nih.gov/REST"


# ── OpenFDA ───────────────────────────────────────────────────────────────────

def fetch_openfda(input_data: str | dict) -> dict:
    """
    Accepts either:
      - A ready-made URL string  e.g. "https://api.fda.gov/drug/label.json?search=..."
      - A search term string     e.g. "ibuprofen"
      - A dict with keys: endpoint, search, limit
    Returns: { raw, parsed }
    """

    # Case 1: full URL passed in directly
    if isinstance(input_data, str) and input_data.startswith("http"):
        url = input_data
        # Append API key if not already present
        if "api_key" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={OPENFDA_API_KEY}"

    # Case 2: plain search term (e.g. "ibuprofen")
    elif isinstance(input_data, str):
        params = {
            "api_key": OPENFDA_API_KEY,
            "search":  f'openfda.generic_name:"{input_data}"',
            "limit":   5
        }
        url = f"{OPENFDA_BASE}/drug/label.json?{urlencode(params)}"

    # Case 3: dict with endpoint + search + optional limit
    elif isinstance(input_data, dict):
        endpoint = input_data.get("endpoint", "/drug/label.json")
        search   = input_data.get("search", "")
        limit    = input_data.get("limit", 5)
        params   = {"api_key": OPENFDA_API_KEY, "search": search, "limit": limit}
        url      = f"{OPENFDA_BASE}{endpoint}?{urlencode(params)}"

    else:
        return {"error": "Invalid input type for fetch_openfda"}

    print(f"[OpenFDA] GET {url}")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return {
            "raw":    None,
            "parsed": None,
            "error":  f"OpenFDA error {response.status_code}: {response.text[:200]}"
        }

    raw  = response.json()
    parsed = parse_openfda(raw)
    return {"raw": raw, "parsed": parsed}


def parse_openfda(raw: dict) -> list[dict]:
    """Extracts the most useful fields from OpenFDA label results."""
    results = raw.get("results", [])
    parsed  = []

    for r in results:
        openfda = r.get("openfda", {})
        parsed.append({
            "brand_name":        openfda.get("brand_name", ["N/A"])[0],
            "generic_name":      openfda.get("generic_name", ["N/A"])[0],
            "manufacturer":      openfda.get("manufacturer_name", ["N/A"])[0],
            "purpose":           r.get("purpose", ["N/A"])[0] if r.get("purpose") else "N/A",
            "active_ingredient": r.get("active_ingredient", ["N/A"])[0] if r.get("active_ingredient") else "N/A",
            "warnings":          r.get("warnings", ["N/A"])[0][:300] if r.get("warnings") else "N/A",
            "dosage":            r.get("dosage_and_administration", ["N/A"])[0][:300] if r.get("dosage_and_administration") else "N/A",
        })

    return parsed


# ── NLM RxNav ─────────────────────────────────────────────────────────────────

def fetch_nlm(drug_name: str) -> dict:
    """
    Queries NLM RxNav for drug info by name.
    Returns: { raw, parsed }
    """

    # Step 1: Get RxCUI (unique drug ID) from drug name
    rxcui = get_rxcui(drug_name)
    if not rxcui:
        return {
            "raw":    None,
            "parsed": None,
            "error":  f"No RxCUI found for '{drug_name}'"
        }

    # Step 2: Get drug properties using RxCUI
    url = f"{NLM_BASE}/rxcui/{rxcui}/allProperties.json?prop=all"
    print(f"[NLM] GET {url}")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return {
            "raw":    None,
            "parsed": None,
            "error":  f"NLM error {response.status_code}"
        }

    raw    = response.json()
    parsed = parse_nlm(raw, drug_name, rxcui)
    return {"raw": raw, "parsed": parsed}


def get_rxcui(drug_name: str) -> str | None:
    """Looks up the RxCUI for a drug name."""
    url      = f"{NLM_BASE}/rxcui.json?name={quote(drug_name)}&search=1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return None
    data  = response.json()
    ideas = data.get("idGroup", {}).get("rxnormId", [])
    return ideas[0] if ideas else None


def parse_nlm(raw: dict, drug_name: str, rxcui: str) -> dict:
    """Extracts useful fields from NLM RxNav properties response."""
    props      = raw.get("propConceptGroup", {}).get("propConcept", [])
    prop_map   = {p["propName"]: p["propValue"] for p in props}

    return {
        "drug_name": drug_name,
        "rxcui":     rxcui,
        "synonym":   prop_map.get("RxNorm Synonym", "N/A"),
        "tty":       prop_map.get("TTY", "N/A"),  # Term type e.g. IN, BN, SCD
        "full_name": prop_map.get("RxNorm Name", drug_name),
    }


# ── Main fetch (combines both) ────────────────────────────────────────────────

def data_fetch(gemini_output: dict) -> dict:
    """
    Main entry point — receives the output from the Gemini step.

    Expected gemini_output format:
    {
        "openfda_url": "https://api.fda.gov/drug/label.json?search=...",
        "nlm_url":     "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=ibuprofen"
    }

    Returns combined result:
    {
        "openfda": { "raw": {...}, "parsed": [{...}] },
        "nlm":     { "raw": {...}, "parsed": {...} }
    }
    """

    openfda_url = gemini_output.get("openfda_url", "")
    nlm_url     = gemini_output.get("nlm_url", "")

    # Hit OpenFDA with the URL from Gemini
    openfda_result = fetch_openfda(openfda_url) if openfda_url else {"raw": None, "parsed": None, "error": "No OpenFDA URL provided"}

    # Hit NLM with the URL from Gemini
    nlm_result = fetch_nlm_url(nlm_url) if nlm_url else {"raw": None, "parsed": None, "error": "No NLM URL provided"}

    return {
        "openfda": openfda_result,
        "nlm":     nlm_result
    }


def fetch_nlm_url(url: str) -> dict:
    """Fetches NLM using a ready-made URL from Gemini."""
    print(f"[NLM] GET {url}")
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        return {
            "raw":    None,
            "parsed": None,
            "error":  f"NLM error {response.status_code}"
        }

    raw = response.json()

    # Try to extract RxCUI and get more details if possible
    ids = raw.get("idGroup", {}).get("rxnormId", [])
    if ids:
        rxcui       = ids[0]
        drug_name   = url.split("name=")[-1].split("&")[0]
        detail_data = fetch_nlm(drug_name)
        return detail_data

    return {"raw": raw, "parsed": None}


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated Gemini output
    gemini_output = {
        "openfda_url": "https://api.fda.gov/drug/label.json?search=openfda.generic_name:\"ibuprofen\"&limit=3",
        "nlm_url":     "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=ibuprofen&search=1"
    }

    result = data_fetch(gemini_output)
    print(json.dumps(result, indent=2))
