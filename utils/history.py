from datetime import datetime


class SessionHistory:
    def __init__(self, max_items=12):
        self.items = []
        self.max_items = max_items

    def add(self, mode, gesture, message, confidence):
        self.items.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": mode,
            "gesture": gesture,
            "message": message,
            "confidence": confidence,
        })
        self.items = self.items[:self.max_items]

    def recent(self):
        return self.items
