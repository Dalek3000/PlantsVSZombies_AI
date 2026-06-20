"""
clicker.py – Maussteuerung: Sonnen anklicken, Pflanzen setzen

Verantwortlichkeiten:
  - Sonnen anklicken (sofort, hohe Priorität)
  - Pflanze aus der Leiste auswählen
  - Pflanze auf Zielzelle platzieren
  - Sicherheitsprüfungen vor jedem Klick (State, Kalibrierung, gültige Position)

Abhängigkeiten: config.py, game_field.py, vision.py
Wird aufgerufen von: main.py (Sonnen), rl_agent.py (Pflanzen setzen)
"""

import time
import random
import threading

import pyautogui  # pip install pyautogui

import config
from config import runtime, FIELD_COLS, PlantConfig
from game_field import game_field, FieldCell
from vision import SunDetection
from user_interaction import state_manager, GameState


# ---------------------------------------------------------------------------
# pyautogui Sicherheitseinstellungen
# ---------------------------------------------------------------------------

# Fail-Safe: Maus in Ecke oben-links stoppt das Programm
pyautogui.FAILSAFE = True

# Minimale Pause zwischen pyautogui-Aktionen (Sekunden)
# Zu schnelle Klicks können vom Spiel ignoriert werden
pyautogui.PAUSE = 0.05


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _human_move_and_click(x: int, y: int, double: bool = False) -> None:
    """
    Bewegt die Maus menschlich (leichte Zufallsabweichung, sanfte Bewegung)
    und klickt dann.

    Zufallsabweichung verhindert dass PvZ exakt identische Klickmuster erkennt
    (relevant wenn Anti-Cheat irgendwann eingebaut wird – PvZ hat keins,
    aber es macht das Verhalten realistischer).
    """
    # Leichte Zufallsabweichung ±3 Pixel
    jitter_x = x + random.randint(-3, 3)
    jitter_y = y + random.randint(-3, 3)

    # Sanfte Bewegung mit zufälliger Dauer (0.05–0.15s)
    duration = random.uniform(0.05, 0.15)
    pyautogui.moveTo(jitter_x, jitter_y, duration=duration)

    if double:
        pyautogui.doubleClick()
    else:
        pyautogui.click()


def _is_safe_to_click() -> bool:
    """
    Gibt True zurück wenn ein Klick jetzt sicher ist.
    Prüft: State == RUNNING und Kalibrierung vorhanden.
    """
    if not state_manager.is_running():
        return False
    if not runtime.is_calibrated():
        return False
    return True


# ---------------------------------------------------------------------------
# Sonnen anklicken
# ---------------------------------------------------------------------------

class SunClicker:
    """
    Klickt erkannte Sonnen so schnell wie möglich an.

    Läuft in einem eigenen Thread mit hoher Priorität.
    Sonnen fallen nach ~10 Sekunden weg – wir haben also etwas Zeit,
    aber der SunClicker sollte nicht durch den langsameren RL-Loop blockiert werden.

    Verwendung:
        sun_clicker = SunClicker()
        sun_clicker.click_suns(sun_list)   # aus dem Vision-Loop aufrufen
    """

    def __init__(self):
        # Lock verhindert parallele Klicks (z.B. wenn Vision schneller liefert)
        self._lock = threading.Lock()

    def click_suns(self, suns: list[SunDetection]) -> None:
        """
        Klickt alle übergebenen Sonnen an.
        Wird direkt aus dem Vision-Loop aufgerufen (nicht in eigenem Thread).
        Ist schnell genug da Mausbewegungen sehr kurz sind.
        """
        if not _is_safe_to_click():
            return
        if not suns:
            return

        with self._lock:
            for sun in suns:
                if not state_manager.is_running():
                    break   # Abbrechen wenn Spiel zwischendurch endet
                try:
                    _human_move_and_click(sun.screen_x, sun.screen_y)
                    # Kurze Pause zwischen mehreren Sonnen
                    if len(suns) > 1:
                        time.sleep(0.05)
                except Exception as e:
                    print(f"[Clicker] Fehler beim Sonnen-Klick: {e}")


# Globale Instanz
sun_clicker = SunClicker()


# ---------------------------------------------------------------------------
# Pflanzen-Leiste: Slot-Koordinaten
# ---------------------------------------------------------------------------

def _get_slot_screen_position(slot_idx: int) -> tuple[int, int] | None:
    """
    Berechnet die absolute Bildschirmposition der Mitte eines Pflanzen-Slots.

    Die Leiste ist oben links im Fenster – Koordinaten werden aus den
    Fensterdimensionen geschätzt und stimmen mit _get_slot_region() in
    vision.py überein.

    Gibt (screen_x, screen_y) zurück oder None wenn Slot ungültig.
    """
    active_plants = config.ACTIVE_PLANTS
    if slot_idx < 0 or slot_idx >= len(active_plants):
        return None

    if not runtime.is_calibrated():
        return None

    # Fenstergröße aus Kalibrierung ableiten
    field_left, field_top = runtime.field_top_left
    field_right, field_bottom = runtime.field_bottom_right
    window_w = field_right - field_left + int(runtime.cell_width)   # Schätzung Gesamtbreite

    # Diese Werte müssen mit vision.py._get_slot_region() übereinstimmen
    # Leiste: obere ~12% des Fensters, oberhalb des Spielfelds
    field_height = field_bottom - field_top
    bar_y = field_top - int(field_height * 0.08)   # oberhalb des Spielfelds

    slot_start_x  = field_left + int(window_w * 0.08)
    slot_width     = int(window_w * 0.068)
    slot_gap       = int(window_w * 0.005)

    slot_x = slot_start_x + slot_idx * (slot_width + slot_gap) + slot_width // 2

    return (slot_x, bar_y)


# ---------------------------------------------------------------------------
# Pflanze setzen
# ---------------------------------------------------------------------------

class PlantPlacer:
    """
    Wählt eine Pflanze aus der Leiste und platziert sie auf dem Spielfeld.

    Ablauf:
      1. Slot in der Leiste anklicken (Pflanze auswählen)
      2. Kurze Pause (Spiel registriert Auswahl)
      3. Zielzelle auf dem Spielfeld anklicken
      4. Zelle in game_field als belegt markieren

    Sicherheitsprüfungen:
      - Pflanze muss baubar sein (plantable_slots[slot_idx] == True)
      - Zielzelle muss für diese Pflanze valid sein (is_plantable())
      - State muss RUNNING sein
    """

    def __init__(self):
        self._lock = threading.Lock()

    def place_plant(
        self,
        slot_idx:        int,
        target_row:      int,
        target_col:      int,
        plantable_slots: list[bool],
    ) -> bool:
        """
        Platziert die Pflanze aus Slot slot_idx auf Zelle (target_row, target_col).

        plantable_slots: aktuelles Ergebnis aus vision.detect_plantable_slots()

        Gibt True zurück wenn erfolgreich, False bei Fehler / ungültiger Aktion.
        """
        if not _is_safe_to_click():
            return False

        active_plants = config.ACTIVE_PLANTS
        if slot_idx < 0 or slot_idx >= len(active_plants):
            print(f"[Clicker] Ungültiger Slot-Index: {slot_idx}")
            return False

        plant = active_plants[slot_idx]

        # --- Sicherheitsprüfung 1: Pflanze baubar? ---
        if slot_idx < len(plantable_slots) and not plantable_slots[slot_idx]:
            print(f"[Clicker] {plant.name} nicht baubar (ausgegraut oder Cooldown).")
            return False

        # --- Sicherheitsprüfung 2: Zielzelle gültig? ---
        try:
            cell = game_field.get_cell(target_row, target_col)
        except (IndexError, RuntimeError) as e:
            print(f"[Clicker] Ungültige Zielzelle ({target_row}, {target_col}): {e}")
            return False

        if not cell.is_plantable(plant):
            reason = "belegt" if cell.occupied else f"falscher Feldtyp ({cell.field_type})"
            print(f"[Clicker] Zelle ({target_row},{target_col}) nicht bepflanzbar: {reason}")
            return False

        with self._lock:
            return self._execute_plant(slot_idx, plant, cell)

    def _execute_plant(
        self,
        slot_idx: int,
        plant:    PlantConfig,
        cell:     FieldCell,
    ) -> bool:
        """
        Führt die eigentliche Klick-Sequenz aus.
        """
        try:
            # Schritt 1: Slot in der Leiste anklicken
            slot_pos = _get_slot_screen_position(slot_idx)
            if slot_pos is None:
                print(f"[Clicker] Slot-Position nicht berechenbar für Slot {slot_idx}")
                return False

            print(f"[Clicker] Wähle {plant.name} (Slot {slot_idx}) → "
                  f"Zelle ({cell.row},{cell.col})")

            _human_move_and_click(slot_pos[0], slot_pos[1])

            # Schritt 2: Kurze Pause – Spiel muss Auswahl registrieren
            time.sleep(0.15)

            # Schritt 3: Abbrechen wenn State sich geändert hat
            if not state_manager.is_running():
                # Auswahl rückgängig machen: Rechtsklick
                pyautogui.rightClick()
                return False

            # Schritt 4: Zielzelle anklicken
            _human_move_and_click(cell.screen_x, cell.screen_y)

            # Schritt 5: Zelle in Matrix als belegt markieren
            game_field.update_cell(cell.row, cell.col, occupied=True, plant=plant)

            print(f"[Clicker] ✓ {plant.name} auf ({cell.row},{cell.col}) gesetzt.")
            return True

        except pyautogui.FailSafeException:
            print("[Clicker] FailSafe ausgelöst – Maus in Ecke. Programm stoppt.")
            state_manager.set(GameState.ENDED)
            return False

        except Exception as e:
            print(f"[Clicker] Fehler beim Pflanzen setzen: {e}")
            # Auswahl rückgängig machen
            try:
                pyautogui.rightClick()
            except Exception:
                pass
            return False

    def cancel_selection(self) -> None:
        """
        Bricht eine laufende Pflanzen-Auswahl ab (Rechtsklick).
        Nützlich wenn der RL-Agent eine Aktion ändert bevor sie ausgeführt wird.
        """
        try:
            pyautogui.rightClick()
        except Exception:
            pass


# Globale Instanz
plant_placer = PlantPlacer()


# ---------------------------------------------------------------------------
# Öffentliche API für rl_agent.py und main.py
# ---------------------------------------------------------------------------

def click_suns(suns: list[SunDetection]) -> None:
    """Shortcut: Sonnen anklicken. Direkt von main.py / Vision-Loop nutzbar."""
    sun_clicker.click_suns(suns)


def place_plant(
    slot_idx:        int,
    target_row:      int,
    target_col:      int,
    plantable_slots: list[bool],
) -> bool:
    """
    Shortcut: Pflanze setzen. Von rl_agent.py aufgerufen.

    slot_idx:        Index in config.ACTIVE_PLANTS (= Slot in der Leiste)
    target_row:      Zielreihe (0-basiert)
    target_col:      Zielspalte (0-basiert)
    plantable_slots: Aus vision.detect_plantable_slots()

    Gibt True zurück wenn Pflanze erfolgreich gesetzt wurde.
    """
    return plant_placer.place_plant(slot_idx, target_row, target_col, plantable_slots)


def cancel_plant_selection() -> None:
    """Bricht laufende Pflanzenauswahl ab."""
    plant_placer.cancel_selection()
