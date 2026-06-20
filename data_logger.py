"""
data_logger.py – Spielverlauf-Logging

Verantwortlichkeiten:
  - Jeden Schritt (Aktion, Reward, State-Info) in CSV schreiben
  - Episode-Zusammenfassung nach jeder Runde speichern
  - Einfache Laufzeit-Statistiken ausgeben (Durchschnitts-Reward, beste Runde)

Abhängigkeiten: config.py
Wird aufgerufen von: main.py
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import runtime, LOG_FILENAME


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    """Daten eines einzelnen Agenten-Schritts."""
    episode:     int
    step:        int
    timestamp:   float
    action:      int
    action_name: str
    reward:      float
    sun_count:   int
    zombie_rows: str    # z.B. "0,0,1,0,0" – eine Zahl pro Reihe
    n_suns:      int    # Anzahl Sonnen auf dem Bildschirm


@dataclass
class EpisodeLog:
    """Zusammenfassung einer abgeschlossenen Episode."""
    episode:       int
    start_time:    float
    end_time:      float
    duration_sec:  float
    total_steps:   int
    total_reward:  float
    victory:       bool
    n_plants_set:  int


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class DataLogger:
    """
    Schreibt Schritt- und Episode-Daten in zwei separate CSV-Dateien.

    step_log.csv    – jeder einzelne Agenten-Schritt
    episode_log.csv – Zusammenfassung jeder Runde

    Verwendung:
        logger = DataLogger()
        logger.start_episode(episode_num)
        logger.log_step(step_log)
        logger.end_episode(victory)
        logger.print_stats()
    """

    STEP_FIELDS = [
        "episode", "step", "timestamp", "action",
        "action_name", "reward", "sun_count", "zombie_rows", "n_suns",
    ]
    EPISODE_FIELDS = [
        "episode", "start_time", "end_time", "duration_sec",
        "total_steps", "total_reward", "victory", "n_plants_set",
    ]

    def __init__(self):
        self._step_path:    Optional[Path] = None
        self._episode_path: Optional[Path] = None
        self._step_writer:    Optional[csv.DictWriter] = None
        self._episode_writer: Optional[csv.DictWriter] = None
        self._step_file    = None
        self._episode_file = None

        # Laufzeit-Statistiken
        self._episode_num:    int   = 0
        self._episode_start:  float = 0.0
        self._step_count:     int   = 0
        self._reward_sum:     float = 0.0
        self._plants_set:     int   = 0
        self._all_rewards:    list[float] = []
        self._best_reward:    float = float("-inf")
        self._best_episode:   int   = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Öffnet die CSV-Dateien zum Schreiben.
        Existierende Dateien werden mit neuen Daten fortgesetzt (append).
        Muss einmalig aufgerufen werden bevor Logging beginnt.
        """
        save_dir = runtime.save_dir
        self._step_path    = save_dir / "step_log.csv"
        self._episode_path = save_dir / "episode_log.csv"

        # Schritt-Log
        step_exists = self._step_path.exists()
        self._step_file = open(self._step_path, "a", newline="", encoding="utf-8")
        self._step_writer = csv.DictWriter(self._step_file, fieldnames=self.STEP_FIELDS)
        if not step_exists:
            self._step_writer.writeheader()

        # Episode-Log
        ep_exists = self._episode_path.exists()
        self._episode_file = open(self._episode_path, "a", newline="", encoding="utf-8")
        self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=self.EPISODE_FIELDS)
        if not ep_exists:
            self._episode_writer.writeheader()

        print(f"[Logger] Logging gestartet:")
        print(f"  Schritte:  {self._step_path}")
        print(f"  Episoden:  {self._episode_path}")

    def close(self) -> None:
        """Schließt alle offenen Dateien sauber."""
        for f in (self._step_file, self._episode_file):
            if f:
                try:
                    f.close()
                except Exception:
                    pass
        print("[Logger] Dateien geschlossen.")

    # ------------------------------------------------------------------
    # Episode-Steuerung
    # ------------------------------------------------------------------

    def start_episode(self, episode_num: int) -> None:
        """Wird zu Beginn jeder Spielrunde aufgerufen."""
        self._episode_num   = episode_num
        self._episode_start = time.time()
        self._step_count    = 0
        self._reward_sum    = 0.0
        self._plants_set    = 0

    def end_episode(self, victory: bool) -> None:
        """
        Schreibt die Episode-Zusammenfassung in die CSV.
        Aktualisiert Statistiken.
        """
        end_time = time.time()
        duration = end_time - self._episode_start

        log = EpisodeLog(
            episode=self._episode_num,
            start_time=round(self._episode_start, 2),
            end_time=round(end_time, 2),
            duration_sec=round(duration, 1),
            total_steps=self._step_count,
            total_reward=round(self._reward_sum, 3),
            victory=victory,
            n_plants_set=self._plants_set,
        )

        if self._episode_writer:
            self._episode_writer.writerow(vars(log))
            self._episode_file.flush()

        # Statistiken aktualisieren
        self._all_rewards.append(self._reward_sum)
        if self._reward_sum > self._best_reward:
            self._best_reward  = self._reward_sum
            self._best_episode = self._episode_num

        print(f"[Logger] Episode {self._episode_num} geloggt: "
              f"{duration:.0f}s, Reward {self._reward_sum:.1f}, "
              f"{'SIEG' if victory else 'Niederlage'}")

    # ------------------------------------------------------------------
    # Schritt-Logging
    # ------------------------------------------------------------------

    def log_step(self, log: StepLog, plant_was_set: bool = False) -> None:
        """
        Schreibt einen einzelnen Schritt in die CSV.
        plant_was_set=True wenn in diesem Schritt eine Pflanze gesetzt wurde.
        """
        self._step_count += 1
        self._reward_sum += log.reward
        if plant_was_set:
            self._plants_set += 1

        if self._step_writer:
            self._step_writer.writerow({
                "episode":     log.episode,
                "step":        log.step,
                "timestamp":   round(log.timestamp, 3),
                "action":      log.action,
                "action_name": log.action_name,
                "reward":      round(log.reward, 4),
                "sun_count":   log.sun_count,
                "zombie_rows": log.zombie_rows,
                "n_suns":      log.n_suns,
            })
            # Flush alle 10 Schritte damit Daten nicht verloren gehen
            if self._step_count % 10 == 0:
                self._step_file.flush()

    # ------------------------------------------------------------------
    # Statistiken
    # ------------------------------------------------------------------

    def print_stats(self) -> None:
        """Gibt eine Übersicht aller bisherigen Episoden auf der Konsole aus."""
        n = len(self._all_rewards)
        if n == 0:
            print("[Logger] Noch keine Episoden abgeschlossen.")
            return

        avg   = sum(self._all_rewards) / n
        worst = min(self._all_rewards)

        print()
        print("=" * 50)
        print(f"  Statistik nach {n} Episode(n)")
        print(f"  Ø Reward:       {avg:.1f}")
        print(f"  Bester Reward:  {self._best_reward:.1f}  (Episode {self._best_episode})")
        print(f"  Schlechtester:  {worst:.1f}")
        print("=" * 50)
        print()

    def make_step_log(
        self,
        action:      int,
        action_name: str,
        reward:      float,
        vision,                 # VisionResult (kein Import nötig durch lazy typing)
    ) -> StepLog:
        """
        Hilfsmethode: erstellt ein StepLog-Objekt aus den aktuellen Werten.
        Erspart main.py das manuelle Befüllen aller Felder.
        """
        zombie_str = ",".join(
            "1" if z else "0"
            for z in (vision.zombie_rows if vision else [])
        )
        return StepLog(
            episode=self._episode_num,
            step=self._step_count + 1,
            timestamp=time.time(),
            action=action,
            action_name=action_name,
            reward=reward,
            sun_count=vision.sun_count if vision else 0,
            zombie_rows=zombie_str,
            n_suns=len(vision.suns) if vision else 0,
        )


# ---------------------------------------------------------------------------
# Globale Instanz
# ---------------------------------------------------------------------------

logger = DataLogger()
