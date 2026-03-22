from datetime import datetime
import uuid


def generate_id():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    rand = uuid.uuid4().hex[:6]
    return f"{now}_{rand}"

class convoHistory:

    def __init__(self, message):
        self.prompt = message
        self.time = datetime.now()
        self.id = generate_id()

