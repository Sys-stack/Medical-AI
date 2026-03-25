# =============================================================
# mainbackend.py — Cura Flask Backend
# Run: python mainbackend.py
# =============================================================
import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, session
from classes import convoHistory, get_sorted_history, lookup_convos, build_timeline
from functions import (
    data_fetch,
    call_gemini,
    build_endpoints,
    call_gemini_for_response,
    call_gemini_for_comparison,   # now defined in functions.py
)

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']            = os.environ.get('commkey', 'dev-secret-key-change-in-prod')
app.config['DEBUG']                 = True
app.config['MAX_CONTENT_LENGTH']    = 16 * 1024 * 1024   # 16 MB upload limit

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


# =============================================================
# SESSION-SCOPED CONVERSATION STORE
# Each browser session gets its own isolated dict so one user
# cannot see another user's history.
# Key   : session_id  →  { convo_id: convoHistory }
# =============================================================
_session_store: dict[str, dict[str, convoHistory]] = {}


def _get_session_id() -> str:
    """Return (and create if needed) a unique id for this browser session."""
    if 'cura_session_id' not in session:
        session['cura_session_id'] = uuid.uuid4().hex
    return session['cura_session_id']


def _get_user_convos() -> dict[str, convoHistory]:
    """Return the conversation dict that belongs to the current session."""
    sid = _get_session_id()
    if sid not in _session_store:
        _session_store[sid] = {}
    return _session_store[sid]


def _allowed_image(filename: str) -> bool:
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


# =============================================================
# PAGES
# =============================================================

@app.route('/', methods=['GET'])
def home():
    return render_template("homepage.html")


@app.route('/about', methods=['GET'])
def about():
    return render_template("about.html")


@app.route('/chatAI', methods=['GET'])
def chatAI():
    return render_template("chat.html")


# =============================================================
# CHAT  —  POST /chat
# Accepts multipart/form-data (text + optional image) OR JSON.
# Frontend: sendMessage() in chat.html
#
# Multipart fields:
#   message : str  — the user's text
#   image   : file — optional image (prescription scan, lab result, etc.)
#
# Returns : { "response": "<AI reply>", "convo_id": "<id>" }
# =============================================================

@app.route('/chat', methods=['POST'])
def chat():
    dictofconvos = _get_user_convos()

    # ── 1. Parse incoming request (multipart OR JSON) ─────────
    if request.content_type and 'multipart/form-data' in request.content_type:
        message    = (request.form.get("message", "") or "").strip()
        image_file = request.files.get("image")
    else:
        data       = request.get_json(force=True)
        message    = (data.get("message", "") or "").strip()
        image_file = None

    if not message and not image_file:
        return jsonify({"error": "Empty message"}), 400

    # ── 2. Handle uploaded image (save temporarily) ───────────
    image_path = None
    if image_file and _allowed_image(image_file.filename):
        tmp_dir    = os.path.join(os.path.dirname(__file__), "tmp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        ext        = image_file.filename.rsplit('.', 1)[1].lower()
        image_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.{ext}")
        image_file.save(image_path)

    # ── 3. Create conversation object ─────────────────────────
    prompt_label = message if message else "[Image upload]"
    convo        = convoHistory(prompt_label)

    # ── 4. PIPELINE ───────────────────────────────────────────
    #
    # Step A — Gemini reads the prescription / medical text + optional image.
    #          Returns: { summary, drugs: ["drug1", ...] }
    #
    # Step B — For each drug, build API endpoints and fetch data from
    #          OpenFDA + NLM via data_fetch().
    #
    # Step C — Second Gemini call: convert all findings into plain-language
    #          layman's terms for the patient.
    # ──────────────────────────────────────────────────────────
    try:
        # Step A: extract summary + drug list from prescription
        gemini_result = call_gemini(
            text       = message if message else None,
            image_path = image_path,
        )

        if "error" in gemini_result:
            response_text = (
                "I had trouble reading that prescription. "
                "Please make sure the text is clear or try uploading a higher-quality image."
            )
        else:
            summary   = gemini_result.get("summary", "")
            drug_list = gemini_result.get("drugs", [])

            # Step B: fetch OpenFDA + NLM data for every identified drug
            drug_details: dict = {}
            for drug in drug_list:
                endpoints          = build_endpoints(drug)
                drug_details[drug] = data_fetch(endpoints)

            # Step C: generate patient-friendly explanation
            response_text = call_gemini_for_response(
                user_message = message,
                summary      = summary,
                drug_details = drug_details,
            )

    except Exception as exc:
        app.logger.error(f"[/chat] Pipeline error: {exc}", exc_info=True)
        response_text = "Something went wrong while processing your request. Please try again."

    finally:
        # Always clean up the temporary image file
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

    # ── 5. Store the completed conversation ───────────────────
    convo.set_response(response_text)
    dictofconvos[convo.id] = convo

    # ── 6. Return response + id to the frontend ───────────────
    return jsonify({
        "response": response_text,
        "convo_id": convo.id,
    })


# =============================================================
# HISTORY  —  GET /history
# Returns only conversations belonging to the current session.
# Frontend: loadHistory() called on page load in chat.html
# Returns : { "conversations": [ { id, date, title, prompt }, ... ] }
# =============================================================

@app.route('/history', methods=['GET'])
def history():
    dictofconvos = _get_user_convos()
    convos       = get_sorted_history(dictofconvos)
    return jsonify({"conversations": convos})


# =============================================================
# COMPARE  —  POST /compare
# Only compares conversations belonging to the current session.
# Frontend: runCompare() in chat.html
# Receives : { "ids": ["id1", "id2", ...] }
# Returns  : { "report": "<health progress analysis>" }
# =============================================================

@app.route('/compare', methods=['POST'])
def compare():
    dictofconvos = _get_user_convos()

    # ── 1. Parse and validate ─────────────────────────────────
    data = request.get_json(force=True)
    ids  = data.get("ids", [])

    if len(ids) < 2:
        return jsonify({"error": "Select at least 2 conversations to compare"}), 400

    # ── 2. Look up conversations (scoped to this session) ─────
    selected = lookup_convos(ids, dictofconvos)

    if len(selected) < 2:
        return jsonify({"error": "Could not find the selected conversations"}), 404

    # ── 3. Build chronological timeline string ────────────────
    timeline = build_timeline(selected)

    # ── 4. Gemini comparison call ─────────────────────────────
    try:
        report = call_gemini_for_comparison(timeline)
    except Exception as exc:
        app.logger.error(f"[/compare] Gemini error: {exc}", exc_info=True)
        report = (
            f"Could not generate AI comparison at this time.\n\n"
            f"Raw timeline for {len(selected)} conversations:\n\n{timeline}"
        )

    return jsonify({"report": report})


# =============================================================
# HOME BUTTON REDIRECT  —  POST /homeclick
# =============================================================

@app.route('/homeclick', methods=['POST'])
def homeclick():
    data         = request.get_json(force=True)
    redirect_con = data.get("click")

    if redirect_con:
        return jsonify({"redirect": "/chatAI"})

    return jsonify({"status": "no action"})


# =============================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
