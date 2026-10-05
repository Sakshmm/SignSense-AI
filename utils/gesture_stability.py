import time


class GestureStability:
    """Requires several identical predictions before accepting one."""

    def __init__(self, stable_frames=5, cooldown=2.0):
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self.candidate = None
        self.count = 0
        self.last_triggered = None
        self.last_trigger_time = 0.0

    def update(self, label, confidence):
        now = time.time()

        if label is None:
            self.candidate = None
            self.count = 0
            self.last_triggered = None
            return None

        if label == self.candidate:
            self.count += 1
        else:
            self.candidate = label
            self.count = 1

        if self.count < self.stable_frames:
            return None

        # Same stable gesture should not fire again until the user changes
        # gesture or removes the hand.
        if label == self.last_triggered:
            return None

        if now - self.last_trigger_time < self.cooldown:
            return None

        self.last_triggered = label
        self.last_trigger_time = now
        return label

    def reset_trigger(self):
        self.last_triggered = None
