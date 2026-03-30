import uuid
from datetime import datetime
from PIL import Image
from typing import Optional


class Session:
    def __init__(self, session_id: str, image: Image.Image, vectorstore):
        self.session_id = session_id
        self.image = image
        self.vectorstore = vectorstore
        self.history: list[dict] = []
        self.created_at = datetime.utcnow().isoformat()

    def add_result(self, result: dict):
        self.history.append({
            **result,
            "timestamp": datetime.utcnow().isoformat()
        })


class SessionStore:
    def __init__(self):
        self._store: dict[str, Session] = {}

    def create(self, image: Image.Image, vectorstore) -> str:
        sid = str(uuid.uuid4())
        self._store[sid] = Session(sid, image, vectorstore)
        return sid

    def get(self, session_id: str) -> Optional[Session]:
        return self._store.get(session_id)


store = SessionStore()