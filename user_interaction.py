"""
user_interaction.py – Benutzerinteraktion, Hotkeys und Zustandsautomat

Verantwortlichkeiten:
  - Startup-Dialog: Speicherpfad, Modell laden, aktive Pflanzen wählen
  - Hotkey-Listener: F5 (Spiel gestartet), F6 (Programm beenden)
  - Zustandsautomat: WAITING → RUNNING → PAUSED → ENDED

Abhängigkeiten: config.py
Wird aufgerufen von: main.py
"""

import sys
import threading
from enum import Enum, auto
from pathlib import Path

import keyboard  # pip install keyboard

from config import (
    runtime,
    PLANT_DATABASE,
    PlantConfig,
    HOTKEY_START_GAME,
    HOTKEY_STOP_PROGRAM,
    DEFAULT_SAVE_DIR,
    MODEL_FILENAME,
    load_calibration,
)


# ---------------------------------------------------------------------------
# Zustandsautomat
# ---------------------------------------------------------------------------

class GameState(Enum):
    """
    Mögliche Zustände des Programms.

    WAITING  → Programm läuft, wartet auf F5 (Benutzer startet Spiel)
    RUNNING  → Spiel läuft, KI ist aktiv
    PAUSED   → Runde beendet (Game-Over / Sieg), wartet auf nächsten F5-Druck
    ENDED    → Programm wird beendet (F6 gedrückt)
    """
    WAITING = auto()
    RUNNING = auto()
    PAUSED  = auto()
    ENDED   = auto()


class StateManager:
    """
    Thread-sicherer Zustandsautomat.
    Alle Module die den State lesen oder setzen wollen, nutzen diese Klasse.

    Verwendung:
        from user_interaction import state_manager, GameState
        if state_manager.is_running():
            ...
        state_manager.set(GameState.PAUSED)
    """

    def __init__(self):
        self._state = GameState.WAITING
        self._lock  = threading.Lock()
        # Event das andere Threads aufweckt wenn RUNNING gesetzt wird
        self.running_event = threading.Event()

    def get(self) -> GameState:
        with self._lock:
            return self._state

    def set(self, new_state: GameState) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
        print(f"[State] {old.name} → {new_state.name}")

        # running_event steuern damit wartende Threads aufgeweckt werden
        if new_state == GameState.RUNNING:
            self.running_event.set()
        else:
            self.running_event.clear()

    def is_running(self) -> bool:
        return self.get() == GameState.RUNNING

    def is_ended(self) -> bool:
        return self.get() == GameState.ENDED

    def wait_until_running(self) -> None:
        """Blockiert den aufrufenden Thread bis der State RUNNING wird."""
        self.running_event.wait()


# Globale Instanz – alle Module importieren dieses Objekt
state_manager = StateManager()


# ---------------------------------------------------------------------------
# Hotkey-Listener
# ---------------------------------------------------------------------------

class HotkeyListener:
    """
    Registriert globale Hotkeys (funktionieren auch wenn PvZ im Vordergrund).

    F5 → WAITING/PAUSED zu RUNNING
    F6 → Programm beenden (→ ENDED)
    """

    def __init__(self, on_start: callable, on_stop: callable):
        self._on_start = on_start
        self._on_stop  = on_stop

    def start(self) -> None:
        keyboard.add_hotkey(HOTKEY_START_GAME,   self._handle_start)
        keyboard.add_hotkey(HOTKEY_STOP_PROGRAM, self._handle_stop)
        print(f"[Hotkeys] {HOTKEY_START_GAME} = Spiel gestartet  |  "
              f"{HOTKEY_STOP_PROGRAM} = Programm beenden")

    def stop(self) -> None:
        keyboard.remove_all_hotkeys()

    def _handle_start(self) -> None:
        current = state_manager.get()
        if current in (GameState.WAITING, GameState.PAUSED):
            self._on_start()
        else:
            print(f"[Hotkeys] {HOTKEY_START_GAME} ignoriert – State ist {current.name}")

    def _handle_stop(self) -> None:
        self._on_stop()


# ---------------------------------------------------------------------------
# Startup-Dialog
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    """Hilfsfunktion: Eingabe mit optionalem Default-Wert."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Hilfsfunktion: Ja/Nein-Frage."""
    hint = "J/n" if default else "j/N"
    answer = input(f"{prompt} ({hint}): ").strip().lower()
    if answer in ("j", "ja", "y", "yes"):
        return True
    if answer in ("n", "nein", "no"):
        return False
    return default


def run_startup_dialog() -> None:
    """
    Führt den interaktiven Startup-Dialog durch und befüllt runtime.*
    Wird einmalig beim Programmstart von main.py aufgerufen.
    """
    print("=" * 60)
    print("  Plants vs. Zombies KI  –  Startmenü")
    print("=" * 60)

    # --- 1. Speicherpfad ---
    save_dir_input = _ask(
        "Speicherort für Modell und Logs",
        default=str(DEFAULT_SAVE_DIR),
    )
    save_dir = Path(save_dir_input).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    runtime.save_dir = save_dir
    print(f"  → Speicherort: {save_dir}")

    # --- 2. Bestehendes Modell laden? ---
    load_model = _ask_yes_no("Bestehendes RL-Modell laden?", default=False)
    if load_model:
        default_model = save_dir / MODEL_FILENAME
        model_path_input = _ask(
            "Pfad zum Modell",
            default=str(default_model),
        )
        model_path = Path(model_path_input).expanduser().resolve()
        if not model_path.exists() and not (model_path.parent / (model_path.name + ".zip")).exists():
            print(f"  [Warnung] Modell nicht gefunden: {model_path}")
            print("  → Starte mit neuem Modell.")
            runtime.load_existing_model = False
        else:
            runtime.load_existing_model = True
            runtime.model_load_path = model_path
            print(f"  → Modell wird geladen: {model_path}")
    else:
        runtime.load_existing_model = False
        print("  → Starte mit neuem Modell.")

    # --- 3. Aktive Pflanzen wählen ---
    print()
    print("Verfügbare Pflanzen:")
    for key, plant in PLANT_DATABASE.items():
        print(f"  [{key:15s}]  {plant.name:20s}  Kosten: {plant.sun_cost} Sonnen")

    print()
    print("Gib die Pflanzen in der Reihenfolge ihrer Slots in der Leiste ein.")
    print("Trenne mit Komma, z.B.:  sunflower, peashooter, wallnut, cherrybomb")
    print("(Reihenfolge = Slot 1 links → Slot N rechts)")

    active_plants: list[PlantConfig] = []
    while not active_plants:
        raw = _ask("Aktive Pflanzen").replace(" ", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        valid = True
        for key in keys:
            if key not in PLANT_DATABASE:
                print(f"  [Fehler] Unbekannte Pflanze: '{key}'")
                valid = False
                break
            active_plants.append(PLANT_DATABASE[key])
        if not valid:
            active_plants = []

    # plant_id neu nummerieren nach Slot-Reihenfolge
    for slot_index, plant in enumerate(active_plants):
        plant.plant_id = slot_index

    import config
    config.ACTIVE_PLANTS = active_plants

    print()
    print("  Aktive Pflanzen:")
    for i, p in enumerate(active_plants):
        print(f"    Slot {i}: {p.name} ({p.sun_cost} Sonnen)")

    # --- 4. Kalibrierung prüfen ---
    print()
    cal = load_calibration()
    if cal:
        print("  Gespeicherte Kalibrierung gefunden.")
        use_existing = _ask_yes_no("Bestehende Kalibrierung verwenden?", default=True)
        if use_existing:
            runtime.field_top_left     = tuple(cal["field_top_left"])
            runtime.field_bottom_right = tuple(cal["field_bottom_right"])
            runtime.cell_width         = cal["cell_width"]
            runtime.cell_height        = cal["cell_height"]
            runtime.field_rows         = cal["field_rows"]
            runtime.window_left        = cal.get("window_left", 0)
            runtime.window_top         = cal.get("window_top",  0)
            print("  → Kalibrierung geladen.")
        else:
            print("  → Neue Kalibrierung wird beim Spielstart durchgeführt.")
    else:
        print("  Keine Kalibrierung gefunden.")
        print("  → Neue Kalibrierung wird beim Spielstart durchgeführt.")

    # --- Abschluss ---
    print()
    print("=" * 60)
    print(f"  Bereit. Starte PvZ und drücke {HOTKEY_START_GAME} wenn das Spiel läuft.")
    print(f"  {HOTKEY_STOP_PROGRAM} beendet das Programm und speichert das Modell.")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Kalibrierungs-Assistent
# ---------------------------------------------------------------------------

def _wait_for_middle_click() -> tuple[int, int]:
    """
    Wartet bis der Benutzer die mittlere Maustaste drückt.
    PvZ bleibt dabei im Vordergrund – kein Enter in der Konsole nötig.
    Gibt (x, y) der Mausposition zum Zeitpunkt des Klicks zurück.
    """
    import mouse  # pip install mouse
    import pyautogui

    result: list[tuple[int, int]] = []
    done = threading.Event()

    def on_event(event):
        if isinstance(event, mouse.ButtonEvent) and            event.button == mouse.MIDDLE and            event.event_type == mouse.DOWN:
            result.append(pyautogui.position())
            done.set()

    mouse.hook(on_event)
    done.wait()
    mouse.unhook_all()
    return result[0]


def run_calibration_assistant() -> None:
    """
    Führt den Benutzer durch die manuelle Spielfeld-Kalibrierung.
    Positionierung per Mittelklick – PvZ bleibt im Vordergrund und pausiert nicht.
    """
    from config import save_calibration, FIELD_ROWS_DEFAULT, FIELD_COLS

    print()
    print("[Kalibrierung] Spielfeld-Koordinaten bestimmen")
    print("-" * 50)
    print("Klicke mit der MITTLEREN MAUSTASTE (Mausrad-Klick) auf die Position.")
    print("PvZ kann dabei im Vordergrund bleiben – das Spiel pausiert nicht.")
    print()

    # Fensterposition des PvZ-Fensters auslesen (für Frame-Koordinaten-Umrechnung)
    from screen_capture import find_pvz_window
    pvz_win = find_pvz_window()
    if pvz_win:
        runtime.window_left = pvz_win.left
        runtime.window_top  = pvz_win.top
        print(f"  Fenster-Ursprung: ({pvz_win.left}, {pvz_win.top})")
    else:
        print("  [Warnung] PvZ-Fenster nicht gefunden – window_left/top bleiben 0.")

    print("  1. Mittelklick auf die OBERE LINKE ECKE der ersten Gras-Zelle.")
    print("     → Das ist die Zelle direkt UNTERHALB der Pflanzen-Leiste, ganz links.")
    print("     → NICHT auf die Leiste selbst klicken.")
    x1, y1 = _wait_for_middle_click()
    print(f"     Gespeichert: ({x1}, {y1})")

    print("  2. Mittelklick auf die UNTERE RECHTE ECKE der letzten Zelle (ganz rechts unten).")
    x2, y2 = _wait_for_middle_click()
    print(f"     Gespeichert: ({x2}, {y2})")

    # Plausibilitätsprüfung: Punkt 1 muss oberhalb von Punkt 2 liegen
    # und das Spielfeld muss eine Mindestgröße haben
    if x2 <= x1 or y2 <= y1:
        print("  [Fehler] Ecke 2 muss rechts und unterhalb von Ecke 1 liegen.")
        print("           Starte die Kalibrierung neu (main.py neu starten).")
        return
    if (x2 - x1) < 200 or (y2 - y1) < 150:
        print("  [Warnung] Spielfeld erscheint sehr klein – stimmen die Klick-Positionen?")
        print("           Fahre trotzdem fort ...")

    # Reihenanzahl: einzige Konsolen-Eingabe – Screenshot ist nicht nötig
    print()
    rows_input = _ask(
        "  Anzahl Reihen (5 = normal/Roof, 6 = Pool)",
        default=str(FIELD_ROWS_DEFAULT),
    )
    try:
        rows = int(rows_input)
        if rows not in (5, 6):
            raise ValueError
    except ValueError:
        print("  [Warnung] Ungültige Eingabe, verwende 5 Reihen.")
        rows = FIELD_ROWS_DEFAULT

    cell_w = (x2 - x1) / FIELD_COLS
    cell_h = (y2 - y1) / rows

    runtime.field_top_left     = (x1, y1)
    runtime.field_bottom_right = (x2, y2)
    runtime.cell_width         = cell_w
    runtime.cell_height        = cell_h
    runtime.field_rows         = rows

    save_calibration({
        "field_top_left":     [x1, y1],
        "field_bottom_right": [x2, y2],
        "cell_width":         cell_w,
        "cell_height":        cell_h,
        "field_rows":         rows,
        "window_left":        runtime.window_left,
        "window_top":         runtime.window_top,
    })

    print(f"\n  → Zellgröße: {cell_w:.1f} × {cell_h:.1f} px")
    print(f"  → {rows} Reihen × {FIELD_COLS} Spalten")
    print("  Kalibrierung abgeschlossen.\n")


# ---------------------------------------------------------------------------
# Öffentliche API für main.py
# ---------------------------------------------------------------------------

def setup(on_game_start: callable, on_program_end: callable) -> HotkeyListener:
    """
    Führt Startup-Dialog durch und startet den Hotkey-Listener.

    on_game_start:  Callback wenn F5 gedrückt (State → RUNNING)
    on_program_end: Callback wenn F6 gedrückt (State → ENDED)

    Gibt den HotkeyListener zurück (zum späteren Stoppen).
    """
    run_startup_dialog()

    def _on_start():
        if not runtime.is_calibrated():
            run_calibration_assistant()
        state_manager.set(GameState.RUNNING)
        on_game_start()

    def _on_stop():
        state_manager.set(GameState.ENDED)
        on_program_end()

    listener = HotkeyListener(on_start=_on_start, on_stop=_on_stop)
    listener.start()
    return listener
