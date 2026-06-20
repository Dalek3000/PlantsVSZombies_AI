"""
game_field.py – Spielfeld-Matrix und Feldtypen-Erkennung

Verantwortlichkeiten:
  - Einmalige Analyse des Spielfelds zu Beginn jeder Runde
  - Feldtypen per HSV-Farberkennung bestimmen (Gras, Wasser, Dach)
  - Matrix[reihe][spalte] → FieldCell (Typ, belegt, Pflanze)
  - Belegte Felder aktualisieren während des Spiels (Pflanze gesetzt / entfernt)

Abhängigkeiten: config.py, screen_capture.py
Wird aufgerufen von: main.py, rl_agent.py, clicker.py
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from config import (
    runtime,
    FieldType,
    HSV_RANGES,
    FIELD_COLS,
    FIELD_ROWS_DEFAULT,
    PlantConfig,
)
from screen_capture import take_single_screenshot


# ---------------------------------------------------------------------------
# Datenstruktur einer einzelnen Zelle
# ---------------------------------------------------------------------------

@dataclass
class FieldCell:
    """
    Repräsentiert eine einzelne Zelle im Spielfeld.

    row, col        : Position (0-basiert)
    field_type      : FieldType.GRASS / WATER / ROOF / EMPTY
    occupied        : True wenn eine Pflanze darauf steht
    plant           : PlantConfig der Pflanze (oder None)
    screen_x/y      : Pixelkoordinaten der Zellmitte auf dem Bildschirm
    """
    row:        int
    col:        int
    field_type: str   = FieldType.EMPTY
    occupied:   bool  = False
    plant:      Optional[PlantConfig] = None
    screen_x:   int   = 0
    screen_y:   int   = 0

    def is_plantable(self, plant: PlantConfig) -> bool:
        """
        Gibt True zurück wenn diese Pflanze auf diese Zelle gesetzt werden kann.
        Bedingung: Zelle nicht belegt UND Feldtyp ist in valid_field_types der Pflanze.
        """
        if self.occupied:
            return False
        return self.field_type in plant.valid_field_types

    def to_state_vector(self) -> list[float]:
        """
        Wandelt die Zelle in einen numerischen Vektor für den RL-State um.
        Format: [field_type_id, occupied, plant_id]
        """
        type_map = {
            FieldType.GRASS: 1.0,
            FieldType.WATER: 2.0,
            FieldType.ROOF:  3.0,
            FieldType.EMPTY: 0.0,
        }
        plant_id = float(self.plant.plant_id + 1) if self.plant else 0.0
        return [
            type_map.get(self.field_type, 0.0),
            1.0 if self.occupied else 0.0,
            plant_id,
        ]


# ---------------------------------------------------------------------------
# Spielfeld-Matrix
# ---------------------------------------------------------------------------

class GameField:
    """
    Verwaltet die vollständige Spielfeld-Matrix.

    Verwendung:
        field = GameField()
        field.initialize()          # einmalig zu Spielbeginn
        field.update_cell(row, col, occupied=True, plant=sunflower_config)
        cell = field.get_cell(row, col)
        free = field.get_free_cells_for_plant(peashooter_config)
    """

    def __init__(self):
        self.rows:  int = runtime.field_rows
        self.cols:  int = FIELD_COLS
        self.grid:  list[list[FieldCell]] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Initialisierung
    # ------------------------------------------------------------------

    def initialize(self, frame: Optional[np.ndarray] = None) -> None:
        """
        Analysiert das Spielfeld einmalig zu Beginn einer Runde.
        Macht einen Screenshot (oder nutzt übergebenen Frame),
        bestimmt Feldtypen per HSV und baut die Matrix auf.
        """
        self.rows = runtime.field_rows

        if frame is None:
            frame = take_single_screenshot()
            if frame is None:
                print("[GameField] Kein Screenshot verfügbar – erstelle leeres Feld.")
                self._build_empty_grid()
                return

        print(f"[GameField] Analysiere Spielfeld ({self.rows}×{self.cols}) …")
        self.grid = []

        for row in range(self.rows):
            grid_row = []
            for col in range(self.cols):
                cell = self._analyze_cell(frame, row, col)
                grid_row.append(cell)
            self.grid.append(grid_row)

        self._initialized = True
        self._print_field_summary()

    def _analyze_cell(self, frame: np.ndarray, row: int, col: int) -> FieldCell:
        """
        Schneidet eine Zelle aus dem Frame aus und bestimmt ihren Typ per HSV.
        """
        # Bildschirmkoordinaten der Zellmitte
        sx, sy = runtime.cell_center(row, col)

        # Zelle ausschneiden: kleines Fenster um die Mitte (30% der Zellgröße)
        half_w = int(runtime.cell_width  * 0.3)
        half_h = int(runtime.cell_height * 0.3)

        # Grenzen sicherstellen (nicht außerhalb des Frames)
        h_frame, w_frame = frame.shape[:2]

        # Koordinaten relativ zum Frame
        # Frame-Ursprung = Fenster-Ursprung (window.left, window.top)
        fx = sx - runtime.window_left
        fy = sy - runtime.window_top

        x1 = max(0, fx - half_w)
        x2 = min(w_frame, fx + half_w)
        y1 = max(0, fy - half_h)
        y2 = min(h_frame, fy + half_h)

        cell_crop = frame[y1:y2, x1:x2]

        # Feldtyp bestimmen
        field_type = self._classify_cell_hsv(cell_crop)

        return FieldCell(
            row=row,
            col=col,
            field_type=field_type,
            occupied=False,
            plant=None,
            screen_x=sx,
            screen_y=sy,
        )

    def _classify_cell_hsv(self, crop: np.ndarray) -> str:
        """
        Bestimmt den Feldtyp eines Zell-Ausschnitts per HSV-Analyse.
        Gibt den Typ mit dem höchsten Anteil passender Pixel zurück.
        """
        if crop.size == 0:
            return FieldType.EMPTY

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        best_type  = FieldType.EMPTY
        best_ratio = 0.0

        for field_type, (lower, upper) in HSV_RANGES.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask  = cv2.inRange(hsv, lower_np, upper_np)
            ratio = np.count_nonzero(mask) / total_pixels
            if ratio > best_ratio:
                best_ratio = ratio
                best_type  = field_type

        # Mindestanteil: 15% der Pixel müssen die Farbe haben
        # sonst gilt das Feld als EMPTY (z.B. Übergangsbereich)
        if best_ratio < 0.15:
            return FieldType.EMPTY

        return best_type

    def _build_empty_grid(self) -> None:
        """Erstellt eine leere Gras-Matrix als Fallback."""
        self.grid = [
            [
                FieldCell(
                    row=row, col=col,
                    field_type=FieldType.GRASS,
                    screen_x=runtime.cell_center(row, col)[0],
                    screen_y=runtime.cell_center(row, col)[1],
                )
                for col in range(self.cols)
            ]
            for row in range(self.rows)
        ]
        self._initialized = True

    # ------------------------------------------------------------------
    # Zugriff und Updates
    # ------------------------------------------------------------------

    def get_cell(self, row: int, col: int) -> FieldCell:
        """Gibt die Zelle an Position (row, col) zurück."""
        self._check_initialized()
        return self.grid[row][col]

    def update_cell(
        self,
        row:      int,
        col:      int,
        occupied: bool,
        plant:    Optional[PlantConfig] = None,
    ) -> None:
        """
        Markiert eine Zelle als belegt oder frei.
        Wird von clicker.py aufgerufen nachdem eine Pflanze gesetzt wurde.
        """
        self._check_initialized()
        cell = self.grid[row][col]
        cell.occupied = occupied
        cell.plant    = plant if occupied else None

    def get_free_cells_for_plant(self, plant: PlantConfig) -> list[FieldCell]:
        """
        Gibt alle Zellen zurück auf die die angegebene Pflanze gesetzt werden kann.
        Berücksichtigt Feldtyp und ob die Zelle belegt ist.
        """
        self._check_initialized()
        return [
            cell
            for row in self.grid
            for cell in row
            if cell.is_plantable(plant)
        ]

    def get_all_cells(self) -> list[FieldCell]:
        """Gibt alle Zellen als flache Liste zurück."""
        self._check_initialized()
        return [cell for row in self.grid for cell in row]

    def reset(self) -> None:
        """
        Setzt alle Belegungs-Infos zurück (Pflanze entfernt).
        Wird zu Beginn einer neuen Runde aufgerufen.
        Feldtypen bleiben erhalten wenn die Map gleich bleibt.
        """
        for row in self.grid:
            for cell in row:
                cell.occupied = False
                cell.plant    = None

    # ------------------------------------------------------------------
    # State-Vektor für RL-Agent
    # ------------------------------------------------------------------

    def to_state_vector(self) -> np.ndarray:
        """
        Wandelt die gesamte Matrix in einen flachen numpy-Vektor um.
        Format: [cell_0_type, cell_0_occupied, cell_0_plant, cell_1_type, …]
        Länge: rows × cols × 3
        """
        self._check_initialized()
        vec = []
        for row in self.grid:
            for cell in row:
                vec.extend(cell.to_state_vector())
        return np.array(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    # Debug-Hilfsmittel
    # ------------------------------------------------------------------

    def _print_field_summary(self) -> None:
        """Gibt eine ASCII-Übersicht des erkannten Spielfelds aus."""
        type_symbols = {
            FieldType.GRASS: "G",
            FieldType.WATER: "W",
            FieldType.ROOF:  "R",
            FieldType.EMPTY: "?",
        }
        print("[GameField] Erkanntes Spielfeld:")
        print("  " + "  ".join(f"C{c}" for c in range(self.cols)))
        for r, row in enumerate(self.grid):
            symbols = "  ".join(type_symbols.get(cell.field_type, "?") for cell in row)
            print(f"R{r} {symbols}")
        print()

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Zeichnet das Gitter und Feldtypen als farbiges Overlay auf einen Frame.
        Nützlich zur visuellen Überprüfung der Kalibrierung und Erkennung.

        Rückgabe: Frame mit Overlay (Original bleibt unverändert)
        """
        overlay = frame.copy()
        type_colors = {
            FieldType.GRASS: (0, 200, 0),    # grün
            FieldType.WATER: (200, 100, 0),  # blau (BGR)
            FieldType.ROOF:  (0, 140, 200),  # orange
            FieldType.EMPTY: (100, 100, 100),
        }

        for row in self.grid:
            for cell in row:
                color = type_colors.get(cell.field_type, (100, 100, 100))

                # Rahmen um die Zelle
                half_w = int(runtime.cell_width  / 2)
                half_h = int(runtime.cell_height / 2)

                # Koordinaten relativ zum Frame
                # Frame-Ursprung = Fenster-Ursprung (keine Titelleisten-Korrektur nötig)
                # screen_capture.py erfasst das gesamte Fenster
                fx = cell.screen_x - runtime.window_left
                fy = cell.screen_y - runtime.window_top

                pt1 = (fx - half_w, fy - half_h)
                pt2 = (fx + half_w, fy + half_h)
                cv2.rectangle(overlay, pt1, pt2, color, 2)

                # Feldtyp-Buchstabe in der Mitte
                label = cell.field_type[0].upper()
                cv2.putText(
                    overlay, label,
                    (fx - 5, fy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1,
                )

                # Belegte Zellen mit X markieren
                if cell.occupied:
                    cv2.line(overlay, pt1, pt2, (0, 0, 255), 2)
                    cv2.line(overlay, (pt1[0], pt2[1]), (pt2[0], pt1[1]), (0, 0, 255), 2)

        return overlay

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "GameField nicht initialisiert. Erst initialize() aufrufen."
            )


# ---------------------------------------------------------------------------
# Globale Instanz
# ---------------------------------------------------------------------------

# Alle Module importieren diese Instanz
game_field = GameField()
