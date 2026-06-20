"""
vision.py – Bilderkennung: Sonnen, Zombies, Pflanzen-Leiste, Game-Over

Verantwortlichkeiten:
  - Sonnen auf dem Bildschirm erkennen (Position + Anzahl)
  - Zombie-Positionen pro Reihe erkennen
  - Baubare Pflanzen in der Leiste erkennen (ausgegraut = nicht baubar)
  - Game-Over / Sieg per Template Matching erkennen
  - Aktuellen Sonnenstand (Zahl) auslesen
  - Ergebnisse als strukturiertes GameState-Objekt zurückgeben

Abhängigkeiten: config.py, game_field.py, screen_capture.py
Wird aufgerufen von: main.py (Loop-Thread)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from config import (
    runtime,
    TEMPLATE_MATCH_THRESHOLD,
    SUN_TEMPLATE_PATH,
    GAME_OVER_TEMPLATE_PATH,
    VICTORY_TEMPLATE_PATH,
    SUN_HSV_RANGE,
    MIN_DETECTION_AREA,
    FIELD_COLS,
)
from game_field import game_field, FieldCell
from screen_capture import save_debug_screenshot


# ---------------------------------------------------------------------------
# Ergebnis-Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class SunDetection:
    """Eine erkannte Sonne auf dem Bildschirm."""
    screen_x: int   # Absolute Bildschirmkoordinate
    screen_y: int
    frame_x:  int   # Koordinate relativ zum PvZ-Fenster-Frame
    frame_y:  int


@dataclass
class VisionResult:
    """
    Vollständiges Erkennungsergebnis eines einzelnen Frames.
    Wird von vision_loop() produziert und an den RL-Agenten weitergegeben.
    """
    # Sonnen
    suns:            list[SunDetection] = field(default_factory=list)
    sun_count:       int = 0          # aktueller Sonnenstand (Zahl oben links)

    # Zombies: zombie_rows[r] = True wenn Zombie in Reihe r erkannt
    zombie_rows:     list[bool] = field(default_factory=list)

    # Pflanzen-Leiste: plantable[i] = True wenn Slot i baubar (nicht ausgegraut)
    plantable_slots: list[bool] = field(default_factory=list)

    # Spielende
    game_over:       bool = False
    victory:         bool = False

    def to_state_vector(self) -> np.ndarray:
        """
        Kombinierter State-Vektor für den RL-Agenten (nur Vision-Teil).
        [sun_count_norm, zombie_row_0, …, zombie_row_N, plantable_0, …, plantable_M]
        """
        sun_norm = min(self.sun_count / 9990.0, 1.0)  # PvZ max Sonnen = 9990
        return np.array(
            [sun_norm] + [1.0 if z else 0.0 for z in self.zombie_rows]
                       + [1.0 if p else 0.0 for p in self.plantable_slots],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Template-Cache
# ---------------------------------------------------------------------------

class TemplateCache:
    """
    Lädt Templates einmalig und hält sie im Speicher.
    Skaliert Templates auf die aktuelle Fensterauflösung wenn nötig.
    """

    def __init__(self):
        self._cache: dict[str, Optional[np.ndarray]] = {}

    def get(self, path: Path) -> Optional[np.ndarray]:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = self._load(path)
        return self._cache[key]

    def _load(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            print(f"[Vision] Template nicht gefunden: {path}")
            print("  → Erstelle Templates indem du Screenshots aus dem laufenden Spiel machst.")
            return None
        tmpl = cv2.imread(str(path))
        if tmpl is None:
            print(f"[Vision] Template konnte nicht geladen werden: {path}")
        return tmpl


_template_cache = TemplateCache()


# ---------------------------------------------------------------------------
# Sonnen-Erkennung
# ---------------------------------------------------------------------------

def detect_suns(frame: np.ndarray) -> list[SunDetection]:
    """
    Erkennt Sonnen auf dem Frame.

    Strategie: HSV-Farberkennung (gelb-orange) + Mindestfläche + Kreisförmigkeit.
    Template Matching als Fallback wenn keine HSV-Treffer.

    Gibt Liste von SunDetection zurück.
    """
    suns: list[SunDetection] = []

    # --- Methode 1: HSV-Farberkennung ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(SUN_HSV_RANGE[0], dtype=np.uint8)
    upper = np.array(SUN_HSV_RANGE[1], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Morphologisches Opening: kleines Rauschen entfernen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_DETECTION_AREA:
            continue

        # Kreisförmigkeit prüfen: Sonne ist rund
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.4:   # 1.0 = perfekter Kreis; 0.4 = großzügig
            continue

        # Mittelpunkt bestimmen
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        fx = int(M["m10"] / M["m00"])
        fy = int(M["m01"] / M["m00"])

        # Absolute Bildschirmkoordinaten
        sx = fx + runtime.field_top_left[0]
        sy = fy + runtime.field_top_left[1]

        suns.append(SunDetection(screen_x=sx, screen_y=sy, frame_x=fx, frame_y=fy))

    # --- Methode 2: Template Matching als Ergänzung ---
    # Nur wenn HSV keine Treffer liefert (z.B. andere Beleuchtung)
    if not suns:
        tmpl = _template_cache.get(SUN_TEMPLATE_PATH)
        if tmpl is not None:
            suns = _template_match_suns(frame, tmpl)

    return suns


def _template_match_suns(frame: np.ndarray, template: np.ndarray) -> list[SunDetection]:
    """Template Matching für Sonnen als Fallback."""
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= TEMPLATE_MATCH_THRESHOLD)

    suns = []
    h, w = template.shape[:2]

    # Duplikate zusammenfassen (NMS: Non-Maximum Suppression einfach)
    seen_positions: list[tuple[int, int]] = []
    for fy, fx in zip(*locations):
        cx = int(fx + w / 2)
        cy = int(fy + h / 2)

        # Zu nah an einem schon gesehenen Treffer? Überspringen.
        too_close = any(abs(cx - sx) < 30 and abs(cy - sy) < 30 for sx, sy in seen_positions)
        if too_close:
            continue

        seen_positions.append((cx, cy))
        sx = cx + runtime.field_top_left[0]
        sy = cy + runtime.field_top_left[1]
        suns.append(SunDetection(screen_x=sx, screen_y=sy, frame_x=cx, frame_y=cy))

    return suns


# ---------------------------------------------------------------------------
# Sonnenstand (Zahl) auslesen
# ---------------------------------------------------------------------------

def read_sun_count(frame: np.ndarray) -> int:
    """
    Liest den aktuellen Sonnenstand aus dem HUD aus.

    Der Sonnenstand steht unten links in der Pflanzen-Leiste.
    Wir nutzen OCR-freies Vorgehen: die Zahl wird aus einem bekannten
    Bereich ausgeschnitten und per Template Matching mit Zifferntemplates
    verglichen. Alternativ kann pytesseract verwendet werden.

    Gibt den erkannten Wert zurück (0 wenn nicht lesbar).
    """
    # Region der Sonnenanzeige – relativ zum Frame
    # Diese Koordinaten passen zu typischen PvZ-Auflösungen und müssen
    # ggf. nach der Kalibrierung angepasst werden (config.py).
    # Wir definieren die Region relativ zur Fensterbreite/-höhe.
    h, w = frame.shape[:2]

    # Sonnenzähler ist oben links im HUD (unter dem Samen-Auswahlbereich), ca. bei:
    sun_region = frame[
        int(h * 0.08) : int(h * 0.17),   # obere 8-17% des Fensters
        int(w * 0.01) : int(w * 0.09),   # ganz links
    ]

    if sun_region.size == 0:
        return 0

    # Pytesseract für OCR (optional – funktioniert gut für PvZ-Zahlen)
    try:
        import pytesseract  # pip install pytesseract
        gray = cv2.cvtColor(sun_region, cv2.COLOR_BGR2GRAY)
        # Schwellenwert für bessere OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        ).strip()
        if text.isdigit():
            return int(text)
    except ImportError:
        pass   # pytesseract nicht installiert → Standard-Fallback
    except Exception:
        pass

    # Fallback: 0 zurückgeben wenn OCR nicht verfügbar
    return 0


# ---------------------------------------------------------------------------
# Zombie-Erkennung
# ---------------------------------------------------------------------------

# HSV-Bereich für Zombies (bräunlich-grau-grüne Töne)
ZOMBIE_HSV_RANGES = [
    # Haut-Töne (Zombie-Körper)
    ((0, 20, 60), (20, 120, 200)),
    # Kleidung (variable Töne) – hier nur grau
    ((0, 0, 60), (179, 40, 160)),
]

def detect_zombies(frame: np.ndarray) -> list[bool]:
    """
    Erkennt ob Zombies in jeder Reihe vorhanden sind.

    Strategie: Für jede Reihe wird der rechte Bereich des Spielfelds
    (Spalten 5-9, wo Zombies hereinkommen) auf Zombie-Farben geprüft.
    Gibt eine Liste[bool] der Länge field_rows zurück.
    """
    rows = runtime.field_rows
    zombie_in_row = [False] * rows

    if not runtime.is_calibrated():
        return zombie_in_row

    # Für jede Reihe den rechten Teil des Spielfelds analysieren
    for row_idx in range(rows):
        # Bereich: gesamte Reihe aber nur rechte Hälfte (Zombies kommen von rechts)
        # Wir prüfen Spalten 0 bis FIELD_COLS (gesamte Breite für laufende Zombies)

        # Reihen-Y-Koordinaten relativ zum Frame
        top_y    = runtime.field_top_left[1] + int(row_idx * runtime.cell_height)
        bottom_y = top_y + int(runtime.cell_height)

        # Horizontaler Bereich: gesamtes Spielfeld
        left_x  = runtime.field_top_left[0]
        right_x  = runtime.field_bottom_right[0]

        # Koordinaten relativ zum Frame
        fy1 = top_y    - runtime.field_top_left[1]
        fy2 = bottom_y - runtime.field_top_left[1]
        fx1 = 0
        fx2 = right_x  - left_x

        h_frame, w_frame = frame.shape[:2]
        fy1 = max(0, fy1)
        fy2 = min(h_frame, fy2)
        fx2 = min(w_frame, fx2)

        row_strip = frame[fy1:fy2, fx1:fx2]
        if row_strip.size == 0:
            continue

        zombie_in_row[row_idx] = _has_zombie_colors(row_strip)

    return zombie_in_row


def _has_zombie_colors(crop: np.ndarray) -> bool:
    """
    Prüft ob in einem Bildausschnitt Zombie-typische Farben vorhanden sind.
    Gibt True zurück wenn genug Pixel die Kriterien erfüllen.
    """
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = crop.shape[0] * crop.shape[1]

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in ZOMBIE_HSV_RANGES:
        mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    ratio = np.count_nonzero(combined_mask) / total
    return ratio > 0.05   # Mindestens 5% der Pixel = Zombie-Farbe


# ---------------------------------------------------------------------------
# Pflanzen-Leiste: baubare Pflanzen erkennen
# ---------------------------------------------------------------------------

def detect_plantable_slots(frame: np.ndarray) -> list[bool]:
    """
    Erkennt welche Pflanzen-Slots in der Leiste baubar sind.

    Nicht baubare Pflanzen sind ausgegraut (niedrige Sättigung / Helligkeit).
    Wir prüfen jeden Slot auf durchschnittliche Sättigung im HSV-Raum.

    Gibt list[bool] zurück – True = baubar, False = ausgegraut / Cooldown.
    """
    active_plants = config.ACTIVE_PLANTS
    if not active_plants:
        return []

    h, w = frame.shape[:2]
    plantable = []

    for slot_idx, plant in enumerate(active_plants):
        region = _get_slot_region(frame, slot_idx, w, h)
        if region is None or region.size == 0:
            plantable.append(False)
            continue

        # Pflanze baubar = Region ist bunt (hohe Sättigung)
        # Ausgegraut    = niedrige Sättigung (S-Kanal im HSV)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])

        # Schwellenwert: experimentell bestimmt – ausgegraut ≈ S < 40
        is_plantable = avg_saturation > 40
        plantable.append(is_plantable)

    return plantable


def _get_slot_region(
    frame:    np.ndarray,
    slot_idx: int,
    frame_w:  int,
    frame_h:  int,
) -> Optional[np.ndarray]:
    """
    Gibt den Bildausschnitt eines Pflanzen-Slots in der Leiste zurück.

    Die Leiste ist oben links im Fenster. Slots sind gleichmäßig verteilt.
    Koordinaten müssen ggf. kalibriert werden – hier Schätzwerte für
    typische PvZ-Auflösungen die nach der Kalibrierung verfeinert werden.
    """
    # Leiste: obere ~12% des Fensters (Pflanzenauswahl oben links in PvZ)
    bar_top    = int(frame_h * 0.01)
    bar_bottom = int(frame_h * 0.12)

    # Erster Slot beginnt ca. bei 8% der Fensterbreite (nach dem Sonnen-Icon)
    # Slots sind ca. 7% der Fensterbreite breit
    slot_start_x = int(frame_w * 0.08)
    slot_width   = int(frame_w * 0.068)
    slot_gap     = int(frame_w * 0.005)   # kleiner Abstand zwischen Slots

    x1 = slot_start_x + slot_idx * (slot_width + slot_gap)
    x2 = x1 + slot_width

    if x2 > frame_w:
        return None

    return frame[bar_top:bar_bottom, x1:x2]


# ---------------------------------------------------------------------------
# Game-Over / Sieg erkennen
# ---------------------------------------------------------------------------

def detect_game_over(frame: np.ndarray) -> tuple[bool, bool]:
    """
    Prüft ob das Spiel beendet ist.
    Gibt (game_over, victory) zurück.

    Nutzt Template Matching auf dem Game-Over- und Sieg-Screen.
    """
    game_over = _match_template(frame, GAME_OVER_TEMPLATE_PATH)
    victory   = _match_template(frame, VICTORY_TEMPLATE_PATH)
    return game_over, victory


def _match_template(frame: np.ndarray, template_path: Path) -> bool:
    """
    Einfaches Template Matching: True wenn Template im Frame gefunden.
    """
    tmpl = _template_cache.get(template_path)
    if tmpl is None:
        return False

    # Template darf nicht größer als Frame sein
    fh, fw = frame.shape[:2]
    th, tw = tmpl.shape[:2]
    if th > fh or tw > fw:
        return False

    result  = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= TEMPLATE_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# Haupt-Erkennungsfunktion
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray) -> VisionResult:
    """
    Führt alle Erkennungsschritte auf einem Frame aus.
    Gibt ein VisionResult zurück.

    Diese Funktion wird im Vision-Loop von main.py aufgerufen.
    """
    result = VisionResult()

    # Spielfeld muss initialisiert sein
    if not game_field._initialized:
        return result

    # Sonnen
    result.suns      = detect_suns(frame)
    result.sun_count = read_sun_count(frame)

    # Zombies
    result.zombie_rows = detect_zombies(frame)

    # Pflanzen-Leiste
    result.plantable_slots = detect_plantable_slots(frame)

    # Spielende
    result.game_over, result.victory = detect_game_over(frame)

    return result


# ---------------------------------------------------------------------------
# Debug-Overlay
# ---------------------------------------------------------------------------

def draw_vision_overlay(frame: np.ndarray, result: VisionResult) -> np.ndarray:
    """
    Zeichnet alle erkannten Objekte als farbiges Overlay auf den Frame.
    Nützlich zur Überprüfung der Erkennung während der Entwicklung.
    """
    overlay = frame.copy()

    # Sonnen: gelbe Kreise
    for sun in result.suns:
        cv2.circle(overlay, (sun.frame_x, sun.frame_y), 20, (0, 215, 255), 2)
        cv2.putText(overlay, "SUN", (sun.frame_x - 15, sun.frame_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 215, 255), 1)

    # Zombie-Reihen: rote horizontale Linien
    for row_idx, has_zombie in enumerate(result.zombie_rows):
        if has_zombie and runtime.is_calibrated():
            y = runtime.field_top_left[1] + int((row_idx + 0.5) * runtime.cell_height)
            fy = y - runtime.field_top_left[1]
            cv2.line(overlay, (0, fy), (frame.shape[1], fy), (0, 0, 255), 2)
            cv2.putText(overlay, f"ZOMBIE R{row_idx}", (5, fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Sonnenstand
    cv2.putText(overlay, f"Suns: {result.sun_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    # Game-Over / Sieg
    if result.game_over:
        cv2.putText(overlay, "GAME OVER", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    if result.victory:
        cv2.putText(overlay, "VICTORY!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    return overlay
