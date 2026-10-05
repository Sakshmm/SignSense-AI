import subprocess
import threading
import queue


class SpeechManager:
    """Reliable offline text-to-speech manager for macOS."""

    def __init__(self):
        self.queue = queue.Queue()
        self.running = True

        self.worker = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.worker.start()

        print("Mac Speech Manager initialized")

    def _run(self):
        while self.running:

            text = self.queue.get()

            if text is None:
                break

            try:
                # macOS built-in offline speech
                subprocess.run(
                    [
                        "say",
                        "-r",
                        "200",
                        str(text)
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            except Exception as e:
                print(f"[Speech Error] {e}")

            finally:
                self.queue.task_done()

    def speak(self, text):

        if not text:
            return

        text = str(text).strip()

        if not text:
            return

        # Remove old pending speech
        try:
            while True:
                old_text = self.queue.get_nowait()
                self.queue.task_done()

        except queue.Empty:
            pass

        # Speak latest message
        self.queue.put_nowait(text)

    def stop(self):

        self.running = False

        self.queue.put(None)