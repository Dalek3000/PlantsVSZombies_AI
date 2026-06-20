"""
screen_capture.py – Screenshots vom PvZ-Fenster

Verantwortlichkeiten:
  - PvZ-Fenster auf dem Bildschirm finden (pygetwindow)
  - Nur das Spielfenster aufnehmen, nicht den ganzen Bildschirm (mss)
  - Screenshots in eine thread-sichere Queue legen
  - Loop läuft in eigenem Thread, pausiert automatisch wenn State != RUNNING

Abhängigkeiten: config.py, user_interaction.py
Wird aufgerufen von: main.py
"""

import threading
import time
from queue import Queue, Full

import numpy as np
import mss          # pip install mss
import mss.tools
import pygetwindow as gw   # pip install pygetwindow

from config import (
    SCREENSHOT_INTERVAL,
    SCREENSHOT_QUEUE_MAX,
)
from user_interaction import state_manager, GameState


# ---------------------------------------------------------------------------
# Fenster finden
# ---------------------------------------------------------------------------

# Mögliche Fenstertitel von PvZ (Steam-Version)
PVZ_WINDOW_TITLES = [
    "Plants vs. Zombies",
    "Plants vs. Zombies GOTY",
    "PlantsVsZombies",
]


def find_pvz_window():
    """
    Sucht das PvZ-Fenster anhand des Fenstertitels.
    Gibt ein pygetwindow-Window-Objekt zurück oder None wenn nicht gefunden.
    """
    for title in PVZ_WINDOW_TITLES:
        windows = gw.getWindowsWithTitle(title)
        if windows:
            return windows[0]

    # Fallback: alle offenen Fenster auflisten damit der Benutzer den Titel sieht
    all_titles = [w.title for w in gw.getAllWindows() if w.title.strip()]
    print("[ScreenCapture] PvZ-Fenster nicht gefunden.")
    print("  Offene Fenster:")
    for t in all_titles:
        print(f"    '{t}'")
    print("  → Trage den korrekten Titel in PVZ_WINDOW_TITLES in screen_capture.py ein.")
    return None


# Höhe der Windows-Titelleiste in Pixeln.
# Wird von runtime.screen_to_frame() genutzt um Bildschirm- in Frame-Koordinaten
# umzurechnen. Muss mit der tatsächlichen Titelleiste übereinstimmen.
TITLEBAR_HEIGHT = 0   # Titelleiste NICHT abschneiden – Frame-Ursprung = Fenster-Ursprung
                      # Dadurch stimmen kalibrierte Bildschirmkoordinaten direkt mit
                      # Frame-Koordinaten überein (screen_x - window.left, screen_y - window.top)


def get_window_region(window) -> dict:
    """
    Gibt die mss-Region des Fensters zurück.
    mss erwartet: {"left": x, "top": y, "width": w, "height": h}

    Wir erfassen das gesamte Fenster inklusive Titelleiste damit
    Frame-Koordinaten direkt mit Bildschirmkoordinaten vergleichbar sind.
    Die Titelleiste enthält keine Spielinhalte und stört die Erkennung nicht,
    da alle Spielfeld-Koordinaten relativ zu field_top_left berechnet werden.
    """
    return {
        "left":   window.left,
        "top":    window.top,
        "width":  window.width,
        "height": window.height,
    }


# ---------------------------------------------------------------------------
# Screenshot-Loop
# ---------------------------------------------------------------------------

class ScreenCaptureThread(threading.Thread):
    """
    Läuft dauerhaft im Hintergrund und legt Screenshots in eine Queue.

    Andere Threads (vision.py) lesen aus dieser Queue.
    Bei State != RUNNING wird der Thread schlafen gelegt (kein Busy-Wait).

    Verwendung:
        capture_thread = ScreenCaptureThread(screenshot_queue)
        capture_thread.start()
    """

    def __init__(self, queue: Queue):
        super().__init__(daemon=True, name="ScreenCaptureThread")
        self.queue   = queue
        self._window = None   # wird beim ersten RUNNING-Eintritt gesucht

    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[ScreenCapture] Thread gestartet.")

        with mss.mss() as sct:
            while not state_manager.is_ended():

                # Warten bis das Spiel läuft
                if not state_manager.is_running():
                    state_manager.running_event.wait(timeout=0.5)
                    continue

                # Fenster suchen falls noch nicht bekannt oder verschwunden
                if self._window is None or not self._is_window_valid():
                    self._window = find_pvz_window()
                    if self._window is None:
                        print("[ScreenCapture] Warte auf PvZ-Fenster …")
                        time.sleep(1.0)
                        continue

                # Screenshot aufnehmen
                try:
                    region = get_window_region(self._window)
                    raw    = sct.grab(region)

                    # mss liefert BGRA → in BGR-numpy-Array umwandeln (OpenCV-Format)
                    frame = np.array(raw)[:, :, :3]   # Alpha-Kanal verwerfen

                    # In Queue legen (alte Frames verwerfen wenn Queue voll)
                    try:
                        self.queue.put_nowait(frame)
                    except Full:
                        # Queue voll → ältesten Frame rauswerfen, neuen rein
                        try:
                            self.queue.get_nowait()
                        except Exception:
                            pass
                        self.queue.put_nowait(frame)

                except Exception as e:
                    print(f"[ScreenCapture] Fehler beim Screenshot: {e}")
                    self._window = None   # Fenster neu suchen beim nächsten Durchlauf

                # Intervall warten
                time.sleep(SCREENSHOT_INTERVAL)

        print("[ScreenCapture] Thread beendet.")

    # ------------------------------------------------------------------
    def _is_window_valid(self) -> bool:
        """Prüft ob das gespeicherte Fenster noch existiert und sichtbar ist."""
        try:
            return self._window.visible and self._window.width > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Hilfsfunktionen für andere Module
# ---------------------------------------------------------------------------

def take_single_screenshot(window=None) -> np.ndarray | None:
    """
    Macht einen einzelnen Screenshot außerhalb des Loop-Threads.
    Nützlich für die Kalibrierung und game_field.py (einmalige Aufnahmen).

    Gibt ein BGR-numpy-Array zurück oder None bei Fehler.
    """
    if window is None:
        window = find_pvz_window()
    if window is None:
        return None

    with mss.mss() as sct:
        region = get_window_region(window)
        raw    = sct.grab(region)
        frame  = np.array(raw)[:, :, :3]
    return frame


def save_debug_screenshot(frame: np.ndarray, filename: str = "debug.png") -> None:
    """
    Speichert einen Frame als PNG zur Fehlersuche.
    Wird z.B. von vision.py genutzt wenn ein Template nicht erkannt wird.
    """
    import cv2
    from config import runtime
    path = runtime.save_dir / filename
    cv2.imwrite(str(path), frame)
    print(f"[ScreenCapture] Debug-Screenshot gespeichert: {path}")


# ---------------------------------------------------------------------------
# Factory-Funktion für main.py
# ---------------------------------------------------------------------------

def create_capture_pipeline() -> tuple[ScreenCaptureThread, Queue]:
    """
    Erstellt Queue und Thread, gibt beides zurück.
    main.py ruft .start() auf dem Thread auf.

    Rückgabe: (thread, queue)
    """
    queue  = Queue(maxsize=SCREENSHOT_QUEUE_MAX)
    thread = ScreenCaptureThread(queue)
    return thread, queue
