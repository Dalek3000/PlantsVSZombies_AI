"""
rl_agent.py – Reinforcement Learning Agent (PPO)

Verantwortlichkeiten:
  - State-Space aus GameField + VisionResult aufbauen
  - Action-Space definieren (Pflanze × Zelle + Warten)
  - PPO-Modell erstellen, laden, speichern
  - Aktionen ausführen und an clicker.py weitergeben
  - Reward berechnen und Episode-Daten sammeln
  - Nach jeder Runde lernen (PPO-Update)

Abhängigkeiten: config.py, game_field.py, vision.py, clicker.py
Wird aufgerufen von: main.py
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional

import gymnasium as gym                          # pip install gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO               # pip install stable-baselines3
from stable_baselines3.common.callbacks import BaseCallback

import config
from config import (
    runtime,
    PPO_CONFIG,
    REWARD_PER_SECOND_ALIVE,
    REWARD_ROUND_COMPLETE,
    REWARD_ZOMBIE_AT_BASE,
    REWARD_INVALID_ACTION,
    WAIT_ACTION_DURATION,
    FIELD_COLS,
    MODEL_FILENAME,
)
from game_field import game_field
from vision import VisionResult
from clicker import place_plant, cancel_plant_selection


# ---------------------------------------------------------------------------
# Gymnasium-Environment
# ---------------------------------------------------------------------------

class PvZEnvironment(gym.Env):
    """
    Gymnasium-kompatibles Environment für Plants vs. Zombies.

    Der Agent bekommt pro Schritt einen State-Vektor und wählt eine Aktion.
    Aktionen werden über clicker.py ausgeführt.
    Rewards kommen aus Spielereignissen (Zeit, Game-Over, Sieg).

    State-Space:
        Flacher Float32-Vektor:
        [field_matrix (rows×cols×3)] + [sun_count_norm] +
        [zombie_rows (rows)] + [plantable_slots (n_plants)]

    Action-Space:
        Diskret: n_plants × (rows×cols) + 1 (Warten)
        Aktion i = Pflanze (i // n_cells) auf Zelle (i % n_cells)
        Letzte Aktion = Warten
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.n_plants = len(config.ACTIVE_PLANTS)
        self.n_rows   = runtime.field_rows
        self.n_cols   = FIELD_COLS
        self.n_cells  = self.n_rows * self.n_cols

        # Action-Space: n_plants × n_cells + 1 (Warten)
        self.n_actions = self.n_plants * self.n_cells + 1
        self.WAIT_ACTION = self.n_actions - 1

        # State-Space Dimension berechnen
        field_vec_size   = self.n_rows * self.n_cols * 3   # field matrix
        vision_vec_size  = 1 + self.n_rows + self.n_plants # sun + zombies + slots
        obs_size         = field_vec_size + vision_vec_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Laufzeit-State
        self._last_vision:   Optional[VisionResult] = None
        self._episode_start: float = 0.0
        self._step_count:    int   = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Wird zu Beginn jeder Episode aufgerufen.
        Setzt den internen State zurück – das Spiel selbst wird NICHT
        zurückgesetzt (das macht der Benutzer manuell).
        """
        super().reset(seed=seed)
        self._episode_start = time.time()
        self._step_count    = 0
        self._last_vision   = None

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        """
        Führt eine Aktion aus und gibt (obs, reward, terminated, truncated, info) zurück.

        Da PvZ asynchron läuft (Spiel läuft immer), wird hier eine kurze
        Wartezeit eingebaut damit das Spiel Zeit hat zu reagieren.
        """
        self._step_count += 1
        reward      = 0.0
        terminated  = False
        truncated   = False

        # --- Aktion ausführen ---
        if action == self.WAIT_ACTION:
            # Warten: Pause damit Sonnen gesammelt werden können
            time.sleep(WAIT_ACTION_DURATION)
            reward += REWARD_PER_SECOND_ALIVE * WAIT_ACTION_DURATION
        else:
            reward += self._execute_plant_action(action)
            # Kurze Pause nach Pflanzaktion
            time.sleep(0.5)
            reward += REWARD_PER_SECOND_ALIVE * 0.5

        # --- Vision-State prüfen (wird von außen gesetzt) ---
        if self._last_vision is not None:
            # Zombie hat Basis erreicht?
            if self._check_zombie_at_base():
                reward    += REWARD_ZOMBIE_AT_BASE

            # Spielende?
            if self._last_vision.game_over:
                terminated = True
                print(f"[RLAgent] Episode beendet: Game Over nach "
                      f"{self._get_elapsed():.0f}s, {self._step_count} Schritten")

            elif self._last_vision.victory:
                terminated = True
                reward    += REWARD_ROUND_COMPLETE
                print(f"[RLAgent] Episode beendet: SIEG nach "
                      f"{self._get_elapsed():.0f}s, {self._step_count} Schritten")

        obs  = self._get_observation()
        info = {
            "elapsed_seconds": self._get_elapsed(),
            "step_count":      self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass   # Rendering läuft über vision.draw_vision_overlay() in main.py

    # ------------------------------------------------------------------
    # State aufbauen
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """
        Baut den State-Vektor aus GameField + VisionResult zusammen.
        Gibt nullen zurück wenn noch kein VisionResult vorhanden.
        """
        # Spielfeld-Matrix
        if game_field._initialized:
            field_vec = game_field.to_state_vector()
        else:
            field_vec = np.zeros(self.n_rows * self.n_cols * 3, dtype=np.float32)

        # Vision-Ergebnis
        if self._last_vision is not None:
            vision_vec = self._last_vision.to_state_vector()
        else:
            vision_vec = np.zeros(1 + self.n_rows + self.n_plants, dtype=np.float32)

        return np.concatenate([field_vec, vision_vec]).astype(np.float32)

    def update_vision(self, vision_result: VisionResult) -> None:
        """
        Aktualisiert den internen Vision-State.
        Wird vom Vision-Loop in main.py aufgerufen.
        """
        self._last_vision = vision_result

    # ------------------------------------------------------------------
    # Aktionen ausführen
    # ------------------------------------------------------------------

    def _execute_plant_action(self, action: int) -> float:
        """
        Dekodiert eine diskrete Aktion und ruft clicker.place_plant() auf.
        Gibt den Reward für diese Aktion zurück.
        """
        if self.n_plants == 0:
            return REWARD_INVALID_ACTION

        slot_idx   = action // self.n_cells
        cell_idx   = action  % self.n_cells
        target_row = cell_idx // self.n_cols
        target_col = cell_idx  % self.n_cols

        # Plantable-Slots aus letztem Vision-Ergebnis
        plantable_slots = (
            self._last_vision.plantable_slots
            if self._last_vision else []
        )

        success = place_plant(slot_idx, target_row, target_col, plantable_slots)

        if success:
            return 0.0   # Neutraler Reward für gültige Pflanzaktion
                         # Der echte Reward kommt durch Überleben (Zeit)
        else:
            return REWARD_INVALID_ACTION

    def _check_zombie_at_base(self) -> bool:
        """
        Prüft ob ein Zombie die Basis (linker Rand) erreicht hat.
        Heuristik: Zombie in Spalte 0 erkannt in irgendeiner Reihe.
        Genauer Mechanismus wird über Game-Over erkannt – dies ist ein
        Frühwarnsignal für negativen Reward.
        """
        if self._last_vision is None:
            return False
        # Vereinfachung: Game-Over ist das klare Signal
        # Zombie-at-base ist hier als Frühindikator gedacht (optional)
        return False   # Kann später verfeinert werden

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _get_elapsed(self) -> float:
        return time.time() - self._episode_start

    def decode_action(self, action: int) -> str:
        """Gibt eine lesbare Beschreibung der Aktion zurück (für Logging)."""
        if action == self.WAIT_ACTION:
            return "WARTEN"
        slot_idx   = action // self.n_cells
        cell_idx   = action  % self.n_cells
        row        = cell_idx // self.n_cols
        col        = cell_idx  % self.n_cols
        plants     = config.ACTIVE_PLANTS
        plant_name = plants[slot_idx].name if slot_idx < len(plants) else "?"
        return f"{plant_name} → Reihe {row}, Spalte {col}"


# ---------------------------------------------------------------------------
# PPO-Agent
# ---------------------------------------------------------------------------

class PvZAgent:
    """
    Verwaltet das PPO-Modell und die Trainings-Episoden.

    Der Agent wird einmalig erstellt und lernt über mehrere Spielrunden.
    Nach jeder Runde wird das Modell gespeichert.

    Verwendung (in main.py):
        agent = PvZAgent()
        agent.initialize()
        action = agent.predict(vision_result)
        agent.record_step(reward, terminated)
        agent.end_episode()   # nach Game-Over / Sieg
    """

    def __init__(self):
        self.env:   Optional[PvZEnvironment] = None
        self.model: Optional[PPO]            = None
        self._episode_rewards: list[float]   = []
        self._episode_count:   int           = 0

    def initialize(self) -> None:
        """
        Erstellt das Environment und das PPO-Modell (oder lädt ein bestehendes).
        Muss nach dem Startup-Dialog aufgerufen werden damit ACTIVE_PLANTS bekannt ist.
        """
        self.env = PvZEnvironment()

        if runtime.load_existing_model and runtime.model_load_path:
            self._load_model()
        else:
            self._create_new_model()

        print(f"[RLAgent] Initialisiert.")
        print(f"  State-Space:  {self.env.observation_space.shape[0]} Dimensionen")
        print(f"  Action-Space: {self.env.n_actions} Aktionen "
              f"({self.env.n_plants} Pflanzen × {self.env.n_cells} Zellen + Warten)")

    def _create_new_model(self) -> None:
        """Erstellt ein neues PPO-Modell mit den Einstellungen aus config.py."""
        print("[RLAgent] Erstelle neues PPO-Modell …")
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=PPO_CONFIG["learning_rate"],
            n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"],
            n_epochs=PPO_CONFIG["n_epochs"],
            gamma=PPO_CONFIG["gamma"],
            gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"],
            verbose=PPO_CONFIG["verbose"],
        )

    def _load_model(self) -> None:
        """Lädt ein bestehendes PPO-Modell von Disk."""
        path = runtime.model_load_path
        print(f"[RLAgent] Lade Modell von {path} …")
        try:
            self.model = PPO.load(str(path), env=self.env)
            print("[RLAgent] Modell erfolgreich geladen.")
        except Exception as e:
            print(f"[RLAgent] Fehler beim Laden: {e}")
            print("  → Erstelle neues Modell.")
            self._create_new_model()

    # ------------------------------------------------------------------
    # Vorhersage (Inference)
    # ------------------------------------------------------------------

    def predict(self, vision_result: VisionResult) -> int:
        """
        Wählt die nächste Aktion basierend auf dem aktuellen State.
        Aktualisiert zuerst den Vision-State im Environment.

        Gibt den Action-Index zurück.
        """
        if self.model is None or self.env is None:
            return self.env.WAIT_ACTION if self.env else 0

        self.env.update_vision(vision_result)
        obs = self.env._get_observation()

        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)

    def decode_action(self, action: int) -> str:
        """Lesbare Aktionsbeschreibung."""
        if self.env:
            return self.env.decode_action(action)
        return str(action)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Wird zu Beginn jeder Spielrunde aufgerufen."""
        if self.env:
            self.env.reset()
        self._episode_rewards = []
        self._episode_count  += 1
        print(f"[RLAgent] Episode {self._episode_count} gestartet.")

    def record_step(self, reward: float) -> None:
        """Reward eines Schritts aufzeichnen."""
        self._episode_rewards.append(reward)

    def end_episode(self, victory: bool) -> None:
        """
        Wird nach Game-Over oder Sieg aufgerufen.
        Führt einen PPO-Lernschritt durch und speichert das Modell.
        """
        total_reward = sum(self._episode_rewards)
        print(f"[RLAgent] Episode {self._episode_count} beendet.")
        print(f"  Schritte:      {len(self._episode_rewards)}")
        print(f"  Total Reward:  {total_reward:.2f}")
        print(f"  Ergebnis:      {'SIEG' if victory else 'NIEDERLAGE'}")

        # PPO-Training: n_steps Schritte sammeln dann updaten
        # stable-baselines3 sammelt intern – wir rufen learn() mit
        # reset_num_timesteps=False auf damit der Fortschritt erhalten bleibt
        if self.model is not None and self.env is not None:
            try:
                print("[RLAgent] Starte PPO-Lernschritt …")
                self.model.learn(
                    total_timesteps=PPO_CONFIG["n_steps"],
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                print("[RLAgent] Lernschritt abgeschlossen.")
            except Exception as e:
                print(f"[RLAgent] Fehler beim Lernen: {e}")

        # Modell speichern
        self.save()

    def save(self) -> None:
        """Speichert das aktuelle Modell auf Disk."""
        if self.model is None:
            return
        path = runtime.save_dir / MODEL_FILENAME
        try:
            self.model.save(str(path))
            print(f"[RLAgent] Modell gespeichert: {path}.zip")
        except Exception as e:
            print(f"[RLAgent] Fehler beim Speichern: {e}")


# ---------------------------------------------------------------------------
# Globale Instanz
# ---------------------------------------------------------------------------

agent = PvZAgent()
