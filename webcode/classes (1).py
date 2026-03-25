from datetime import datetime
import uuid


def generate_id():
    now  = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    rand = uuid.uuid4().hex[:6]
    return f"{now}_{rand}"


class convoHistory:
    """
    Stores a single conversation exchange (one prompt → one response).

    Attributes
    ----------
    id       : str      — unique conversation identifier
    prompt   : str      — the user's original message
    response : str      — the AI reply (set after API call completes)
    time     : datetime — timestamp of the conversation
    title    : str      — first 40 chars of prompt, used as sidebar label
    """

    def __init__(self, message: str):
        self.id       = generate_id()
        self.prompt   = message
        self.response = None                     # populated by set_response() after API call
        self.time     = datetime.now()
        self.title    = message[:40].strip()     # short label shown in the history sidebar

    def set_response(self, response: str):
        """Attach the AI/API response once it is ready."""
        self.response = response

    def to_dict(self) -> dict:
        """
        Serialise for GET /history.
        Shape matches what renderHistory() in chat.html expects:
          { id, date, title, prompt }
        """
        return {
            "id":     self.id,
            "date":   self.time.strftime("%d %b %Y"),   # e.g. "22 Mar 2026"
            "title":  self.title,
            "prompt": self.prompt,
        }


# =============================================================
# Conversation store helpers
# These operate on the dictofconvos dict that lives in mainbackend.py
# and are imported / called from there.
# =============================================================

def get_sorted_history(dictofconvos: dict) -> list[dict]:
    """
    Returns all conversations as a list of dicts, sorted newest-first.
    Used by GET /history to populate the sidebar.
    """
    return sorted(
        [c.to_dict() for c in dictofconvos.values()],
        key=lambda c: c["date"],
        reverse=True
    )


def lookup_convos(ids: list[str], dictofconvos: dict) -> list:
    """
    Given a list of conversation ids, return the matching convoHistory objects.
    Silently skips any id that does not exist in the store.
    """
    return [dictofconvos[i] for i in ids if i in dictofconvos]


def build_timeline(selected: list) -> str:
    """
    Builds a chronological plain-text timeline from a list of convoHistory objects.
    The timeline is passed to the AI for the health-progress comparison.

    Format per entry:
        [22 Mar 2026]
        User : I have had a headache and feel tired.
        Cura : <AI response text>
        ──────────────────────────────────────────

    ── API NOTE ──────────────────────────────────────────────────
    Pass the returned string directly into your Gemini/LLM call
    as context. Example system prompt:

        "Given the following health conversations in chronological
         order, analyse the progression of the user's condition.
         Highlight improvements, recurring issues, and any concerns."

    Then append the timeline as the user content.
    ─────────────────────────────────────────────────────────────
    """
    # Sort oldest → newest so the AI sees the timeline in order
    sorted_convos = sorted(selected, key=lambda c: c.time)

    lines = []
    for c in sorted_convos:
        lines.append(f"[{c.time.strftime('%d %b %Y  %H:%M')}]")
        lines.append(f"User : {c.prompt}")
        lines.append(f"Cura : {c.response if c.response else '(no response stored)'}")
        lines.append("─" * 50)

    return "\n".join(lines)
