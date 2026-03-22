# =============================================================
# mainbackend.py — Cura Flask Backend
# Run: python mainbackend.py
# =============================================================

from flask import Flask, render_template, request, jsonify
from classes import convoHistory, get_sorted_history, lookup_convos, build_timeline
from functions import data_fetch
import os

# ── In-memory conversation store ──────────────────────────────
# Key   : convo.id
# Value : convoHistory instance
# NOTE  : Resets on every server restart.
#         Swap dictofconvos for a SQLite/PostgreSQL session store
#         when you need persistence across restarts.
dictofconvos: dict[str, convoHistory] = {}

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('commkey', 'dev-key')
app.config['DEBUG']      = True


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
# Frontend: sendMessage() in chat.html
# Receives : { "message": "<user text>" }
# Returns  : { "response": "<AI reply>", "convo_id": "<id>" }
# =============================================================

@app.route('/chat', methods=['POST'])
def chat():

    # ── 1. Parse and validate incoming JSON ───────────────────
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    # ── 2. Create conversation object and register it ─────────
    convo = convoHistory(message)

    # ── 3. API LOGIC GOES HERE ─────────────────────────────────
    #
    # Replace the placeholder below with your real AI pipeline:
    #
    #   a) First Gemini call — extract medical intent and build API URLs:
    #        gemini_urls = call_gemini_for_urls(message)
    #        # Returns: { "openfda_url": "...", "nlm_url": "..." }
    #
    #   b) Fetch real drug/medical data using functions.py:
    #        fetched_data = data_fetch(gemini_urls)
    #        # Returns: { "openfda": { parsed: [...] }, "nlm": { parsed: {...} } }
    #
    #   c) Second Gemini call — generate the final human-readable reply:
    #        response_text = call_gemini_for_response(message, fetched_data)
    #
    response_text = "API logic not yet implemented."
    # ──────────────────────────────────────────────────────────

    # ── 4. Store the completed conversation ───────────────────
    convo.set_response(response_text)
    dictofconvos[convo.id] = convo

    # ── 5. Return response + id to the frontend ───────────────
    return jsonify({
        "response": response_text,
        "convo_id": convo.id,
    })


# =============================================================
# HISTORY  —  GET /history
# Frontend: loadHistory() called on page load in chat.html
# Returns all stored conversations sorted newest-first for
# the collapsible history sidebar.
# Returns : { "conversations": [ { id, date, title, prompt }, ... ] }
# =============================================================

@app.route('/history', methods=['GET'])
def history():

    # get_sorted_history() serialises every convoHistory via to_dict()
    # and sorts them newest-first — all logic lives in classes.py
    convos = get_sorted_history(dictofconvos)

    return jsonify({"conversations": convos})


# =============================================================
# COMPARE  —  POST /compare
# Frontend: runCompare() triggered when user selects 2+ convos
# and clicks "Compare health →" in the sidebar.
# Receives : { "ids": ["id1", "id2", ...] }
# Returns  : { "report": "<health progress analysis>" }
# =============================================================

@app.route('/compare', methods=['POST'])
def compare():

    # ── 1. Parse and validate ─────────────────────────────────
    data = request.get_json(force=True)
    ids  = data.get("ids", [])

    if len(ids) < 2:
        return jsonify({"error": "Select at least 2 conversations to compare"}), 400

    # ── 2. Look up the requested conversations ────────────────
    selected = lookup_convos(ids, dictofconvos)

    if len(selected) < 2:
        return jsonify({"error": "Could not find the selected conversations"}), 404

    # ── 3. Build chronological timeline string ────────────────
    # build_timeline() sorts by timestamp and formats each entry as:
    #   [date]  User: <prompt>  Cura: <response>
    # This string is ready to be passed straight into your LLM call.
    timeline = build_timeline(selected)

    # ── 4. API LOGIC GOES HERE ─────────────────────────────────
    #
    # Pass the timeline into your Gemini/LLM call for analysis:
    #
    #   report = call_gemini_for_comparison(timeline)
    #
    # Suggested system prompt for Gemini:
    #   "Given the following health conversations in chronological
    #    order, analyse the progression of the user's condition.
    #    Highlight improvements, recurring issues, and any concerns
    #    worth flagging to a clinician."
    #
    # Then pass `timeline` as the user content.
    #
    report = f"Timeline built for {len(selected)} conversations.\nAPI logic not yet implemented.\n\n{timeline}"
    # ──────────────────────────────────────────────────────────

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
