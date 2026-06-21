"""
config.py – Zentrale Konfiguration für die PvZ-KI
Alle Parameter, Pfade und Konstanten werden hier definiert.
Andere Module importieren nur aus dieser Datei – nie hardcoded Werte.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

# Basisverzeichnis des Projekts (dort wo config.py liegt)
BASE_DIR = Path(__file__).parent

# Standardpfad für gespeicherte RL-Modelle und Logs
# Wird beim Start vom Benutzer überschrieben (user_interaction.py)
DEFAULT_SAVE_DIR = BASE_DIR / "saves"
DEFAULT_SAVE_DIR.mkdir(exist_ok=True)

# Dateiname des gespeicherten PPO-Modells
MODEL_FILENAME = "pvz_ppo_model"

# Dateiname der Kalibrierungsdaten (Spielfeld-Koordinaten)
CALIBRATION_FILE = BASE_DIR / "calibration.json"

# Dateiname für den Spielverlaufs-Log
LOG_FILENAME = "game_log.csv"


# ---------------------------------------------------------------------------
# Hotkeys
# ---------------------------------------------------------------------------

HOTKEY_START_GAME   = "F5"   # Teilt dem Programm mit: Spiel wurde gestartet
HOTKEY_STOP_PROGRAM = "F6"   # Programm sauber beenden und Modell speichern


# ---------------------------------------------------------------------------
# Screenshot & Performance
# ---------------------------------------------------------------------------

# Intervall zwischen Screenshots in Sekunden
SCREENSHOT_INTERVAL = 0.3   # 300ms → ~3 Bilder/Sekunde, ausreichend für PvZ

# Maximale Größe der Screenshot-Queue (verhindert Stau bei langsamer Verarbeitung)
SCREENSHOT_QUEUE_MAX = 5


# ---------------------------------------------------------------------------
# Spielfeld-Konfiguration
# ---------------------------------------------------------------------------

# Mögliche Feldtypen (bestimmt was darauf gepflanzt werden kann)
class FieldType:
    GRASS       = "grass"       # Normaler Rasen → alle Landpflanzen
    WATER       = "water"       # Wasser-Reihen im Pool-Level → nur Seerosenblatt + darauf
    ROOF        = "roof"        # Dach-Level → nur Blumentöpfe + darauf
    EMPTY       = "empty"       # Unbekannt / nicht erkannt
    OCCUPIED    = "occupied"    # Belegt durch Pflanze

# Spielfeld-Dimensionen (Reihen × Spalten)
# Standard: 5 Reihen × 9 Spalten; Pool: 6 Reihen × 9 Spalten
FIELD_ROWS_DEFAULT = 5
FIELD_ROWS_POOL    = 6
FIELD_COLS         = 9

# HSV-Farbbereiche für Feldtyp-Erkennung (OpenCV HSV: H=0-179, S=0-255, V=0-255)
# Format: (H_min, S_min, V_min), (H_max, S_max, V_max)
HSV_RANGES = {
    FieldType.GRASS: ((25, 40, 40),  (85, 255, 255)),   # Grüntöne
    FieldType.WATER: ((90, 80, 80),  (130, 255, 255)),   # Blautöne
    FieldType.ROOF:  ((10, 20, 80),  (30, 150, 200)),    # Braun/Grau-Töne
}


# ---------------------------------------------------------------------------
# Pflanzen-Konfiguration
# ---------------------------------------------------------------------------
# Wird beim Programmstart vom Benutzer befüllt (welche Pflanzen im Slot sind).
# Die Reihenfolge entspricht der Reihenfolge in der Pflanzen-Leiste im Spiel.

@dataclass
class PlantConfig:
    """Konfiguration einer einzelnen Pflanze."""
    name: str               # Anzeigename, z.B. "Sunflower"
    plant_id: int           # Eindeutige ID für State-Space / Action-Space
    sun_cost: int           # Sonnenkosten
    # Auf welchen Feldtypen kann die Pflanze gesetzt werden?
    valid_field_types: list = field(default_factory=lambda: [FieldType.GRASS])
    # Template-Bildpfad für Erkennung in der Leiste (relativ zu BASE_DIR)
    template_path: Optional[str] = None


# Vordefinierte Pflanzendatenbank – wird als Referenz genutzt
# Der Benutzer wählt beim Start welche davon im aktuellen Slot sind
PLANT_DATABASE: dict[str, PlantConfig] = {
    "sunflower": PlantConfig(
        name="Sunflower",
        plant_id=0,
        sun_cost=50,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/sunflower.png",
    ),
    "peashooter": PlantConfig(
        name="Peashooter",
        plant_id=1,
        sun_cost=100,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/peashooter.png",
    ),
    "wallnut": PlantConfig(
        name="Wall-nut",
        plant_id=2,
        sun_cost=50,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/wallnut.png",
    ),
    "snowpea": PlantConfig(
        name="Snow Pea",
        plant_id=3,
        sun_cost=175,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/snowpea.png",
    ),
    "cherrybomb": PlantConfig(
        name="Cherry Bomb",
        plant_id=4,
        sun_cost=150,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/cherrybomb.png",
    ),
    "lilypad": PlantConfig(
        name="Lily Pad",
        plant_id=5,
        sun_cost=25,
        valid_field_types=[FieldType.WATER],   # Nur auf Wasser!
        template_path="templates/plants/lilypad.png",
    ),
    "tallnut": PlantConfig(
        name="Tall-nut",
        plant_id=6,
        sun_cost=125,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/tallnut.png",
    ),
    "repeaterpea": PlantConfig(
        name="Repeater",
        plant_id=7,
        sun_cost=200,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/repeaterpea.png",
    ),
    "potatomine": PlantConfig(
        name="Potato Mine",
        plant_id=8,
        sun_cost=25,
        valid_field_types=[FieldType.GRASS],
        template_path="templates/plants/potatomine.png",
    ),
    "chomper": PlantConfig(
        name="Chomper",
        plant_id=9,
        sun_cost=150,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/chomper.png",
    ),
    "flowerspot": PlantConfig(
        name="Flower Pot",
        plant_id=10,
        sun_cost=25,
        valid_field_types=[FieldType.ROOF],   # Nur auf Dach!
        template_path="templates/plants/flowerspot.png",
    ),
    "kernel": PlantConfig(
        name="Kernel-Pult",
        plant_id=11,
        sun_cost=100,
        valid_field_types=[FieldType.GRASS, FieldType.ROOF],
        template_path="templates/plants/kernel.png",
    ),
}

# Aktive Pflanzen für die aktuelle Sitzung (wird von user_interaction.py gesetzt)
# Liste von PlantConfig-Objekten in Slot-Reihenfolge (Slot 0 = linker Slot in Leiste)
ACTIVE_PLANTS: list[PlantConfig] = []


# ---------------------------------------------------------------------------
# Vision / Template Matching
# ---------------------------------------------------------------------------

# Minimaler Konfidenz-Schwellwert für Template Matching (0.0 – 1.0)
TEMPLATE_MATCH_THRESHOLD = 0.75

# Pfad zum Template für die Sonne
SUN_TEMPLATE_PATH = BASE_DIR / "templates" / "sun.png"

# Pfad zum Template für den Game-Over-Screen
GAME_OVER_TEMPLATE_PATH = BASE_DIR / "templates" / "game_over.png"

# Pfad zum Template für den Runden-Abschluss (Wave-Complete / Siegesscreen)
VICTORY_TEMPLATE_PATH = BASE_DIR / "templates" / "victory.png"

# HSV-Bereich für Sonnen-Erkennung (gelb-oranges Leuchten)
SUN_HSV_RANGE = ((15, 150, 180), (35, 255, 255))

# Minimale Pixelfläche damit ein erkanntes Objekt als gültig gilt (Rauschen filtern)
MIN_DETECTION_AREA = 300   # Pixel²


# ---------------------------------------------------------------------------
# RL-Agent Konfiguration
# ---------------------------------------------------------------------------

# PPO-Hyperparameter (stable-baselines3 Defaults sind gute Startwerte)
PPO_CONFIG = {
    "learning_rate":  3e-4,
    "n_steps":        512,    # Schritte pro Update – niedrig wegen langer Spielzeit
    "batch_size":     64,
    "n_epochs":       10,
    "gamma":          0.99,   # Discount-Faktor: Zukunft zählt fast gleich viel
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "verbose":        1,
}

# Reward-Werte
REWARD_PER_SECOND_ALIVE   =  0.1
REWARD_ROUND_COMPLETE     = 50.0
REWARD_ZOMBIE_AT_BASE     = -10.0   # Zombie erreicht linken Rand
REWARD_INVALID_ACTION     =  -0.5   # Pflanze auf besetztes/falsches Feld

# Wie viele Sekunden "wartet" eine WAIT-Aktion (Sonne sammeln)?
WAIT_ACTION_DURATION = 2.0


# ---------------------------------------------------------------------------
# Kalibrierungs-Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_calibration() -> Optional[dict]:
    """
    Lädt gespeicherte Kalibrierungsdaten aus calibration.json.
    Gibt None zurück wenn keine Datei existiert.
    """
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return None


def save_calibration(data: dict) -> None:
    """Speichert Kalibrierungsdaten in calibration.json."""
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Config] Kalibrierung gespeichert: {CALIBRATION_FILE}")


# ---------------------------------------------------------------------------
# Laufzeit-State (wird von anderen Modulen gelesen/geschrieben)
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """
    Laufzeit-Konfiguration die sich während des Programms ändern kann.
    Wird als Singleton genutzt – import und direkt verwenden.
    """
    save_dir: Path = DEFAULT_SAVE_DIR
    load_existing_model: bool = False
    model_load_path: Optional[Path] = None

    # Kalibrierung: Koordinaten des Spielfelds im Bildschirm (Pixel)
    field_top_left:     Optional[tuple[int, int]] = None   # (x, y)
    field_bottom_right: Optional[tuple[int, int]] = None   # (x, y)
    cell_width:         Optional[float] = None
    cell_height:        Optional[float] = None
    field_rows:         int = FIELD_ROWS_DEFAULT

    # Fenster-Ursprung (obere linke Ecke des PvZ-Fensters inkl. Titelleiste)
    # Wird beim Kalibrieren automatisch aus pygetwindow befüllt.
    # Nötig um Bildschirmkoordinaten in Frame-Koordinaten umzurechnen.
    window_left: int = 0
    window_top:  int = 0

    def is_calibrated(self) -> bool:
        """Gibt True zurück wenn Spielfeld-Koordinaten bekannt sind."""
        return self.field_top_left is not None and self.field_bottom_right is not None

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        """
        Berechnet die Bildschirm-Pixelkoordinaten der Mitte einer Zelle.
        row: 0-basiert von oben, col: 0-basiert von links.
        """
        if not self.is_calibrated():
            raise RuntimeError("Spielfeld nicht kalibriert. Erst calibrate() aufrufen.")
        x = int(self.field_top_left[0] + (col + 0.5) * self.cell_width)
        y = int(self.field_top_left[1] + (row + 0.5) * self.cell_height)
        return (x, y)


# Globale Instanz – alle Module importieren dieses Objekt
runtime = RuntimeConfig()
