"""
main.py – Einstiegspunkt und Haupt-Loop der PvZ-KI

Orchestriert alle Module:
  1. Startup-Dialog + Hotkeys         (user_interaction)
  2. Screenshot-Thread starten        (screen_capture)
  3. Spielfeld initialisieren         (game_field)
  4. Vision-Loop: Erkennen + Klicken  (vision, clicker)
  5. RL-Agent: Entscheiden + Lernen   (rl_agent)
  6. Logging                          (data_logger)

Thread-Struktur:
  Main-Thread      – Startup, wartet auf ENDED
  ScreenCapture    – Screenshot alle 300ms → Queue
  VisionLoop       – liest Queue, erkennt, klickt Sonnen, updated Agent
  AgentLoop        – fragt Agent, führt Pflanz-Aktionen aus
"""

import sys
import time
import threading
from queue import Queue, Empty

import numpy as np

# --- Eigene Module ---
import config
from config import runtime
from user_interaction import setup, state_manager, GameState
from screen_capture import create_capture_pipeline, take_single_screenshot, save_debug_screenshot
from game_field import game_field
from vision import analyze_frame, draw_vision_overlay, VisionResult
from clicker import click_suns, cancel_plant_selection
from rl_agent import agent
from data_logger import logger


# ---------------------------------------------------------------------------
# Gemeinsamer State zwischen Vision- und Agent-Loop
# ---------------------------------------------------------------------------

# Letztes VisionResult – Vision schreibt, Agent liest
_latest_vision: VisionResult | None = None
_vision_lock = threading.Lock()

# Signalisiert dem Agent-Loop dass ein neues VisionResult bereit ist
_vision_event = threading.Event()


def _set_vision(result: VisionResult) -> None:
    global _latest_vision
    with _vision_lock:
        _latest_vision = result
    _vision_event.set()


def _get_vision() -> VisionResult | None:
    with _vision_lock:
        return _latest_vision


# ---------------------------------------------------------------------------
# Vision-Loop
# ---------------------------------------------------------------------------

def vision_loop(screenshot_queue: Queue) -> None:
    """
    Läuft in eigenem Thread.
    Liest Screenshots aus der Queue, analysiert sie und:
      - klickt Sonnen sofort an
      - updated den Agent mit dem neuesten VisionResult
      - erkennt Game-Over / Sieg und setzt State auf PAUSED
    """
    print("[VisionLoop] Thread gestartet.")

    while not state_manager.is_ended():

        # Warten bis Spiel läuft
        if not state_manager.is_running():
            state_manager.running_event.wait(timeout=0.5)
            continue

        # Screenshot aus Queue holen (max 1s warten)
        try:
            frame = screenshot_queue.get(timeout=1.0)
        except Empty:
            continue

        # Frame analysieren
        try:
            result = analyze_frame(frame)
        except Exception as e:
            print(f"[VisionLoop] Fehler bei Analyse: {e}")
            continue

        # Sonnen sofort anklicken (höchste Priorität)
        if result.suns:
            try:
                click_suns(result.suns)
            except Exception as e:
                print(f"[VisionLoop] Fehler beim Sonnen-Klick: {e}")

        # VisionResult für Agent bereitstellen
        _set_vision(result)

        # Spielende erkennen
        if result.game_over or result.victory:
            if state_manager.is_running():
                print(f"[VisionLoop] Spielende erkannt: "
                      f"{'Sieg' if result.victory else 'Game Over'}")
                state_manager.set(GameState.PAUSED)

    print("[VisionLoop] Thread beendet.")


# ---------------------------------------------------------------------------
# Agent-Loop
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    """
    Läuft in eigenem Thread.
    Wartet auf neue VisionResults, fragt den RL-Agenten nach der nächsten
    Aktion und führt sie über clicker.py aus.
    """
    print("[AgentLoop] Thread gestartet.")
    step = 0

    while not state_manager.is_ended():

        # Warten bis Spiel läuft
        if not state_manager.is_running():
            state_manager.running_event.wait(timeout=0.5)
            _vision_event.clear()
            continue

        # Auf neues VisionResult warten (max 2s)
        got_new = _vision_event.wait(timeout=2.0)
        _vision_event.clear()

        if not got_new or not state_manager.is_running():
            continue

        vision = _get_vision()
        if vision is None:
            continue

        # Spielende: kein Agenten-Schritt mehr
        if vision.game_over or vision.victory:
            continue

        step += 1

        # Agent fragen
        try:
            action      = agent.predict(vision)
            action_name = agent.decode_action(action)
        except Exception as e:
            print(f"[AgentLoop] Fehler bei Vorhersage: {e}")
            continue

        # Aktion ausführen und Reward sammeln
        try:
            _, reward, terminated, _, _ = agent.env.step(action)
        except Exception as e:
            print(f"[AgentLoop] Fehler bei step(): {e}")
            reward     = 0.0
            terminated = False

        # Logging
        try:
            plant_was_set = (action != agent.env.WAIT_ACTION and reward >= 0)
            step_log = logger.make_step_log(action, action_name, reward, vision)
            logger.log_step(step_log, plant_was_set=plant_was_set)
        except Exception as e:
            print(f"[AgentLoop] Logging-Fehler: {e}")

        agent.record_step(reward)

    print("[AgentLoop] Thread beendet.")


# ---------------------------------------------------------------------------
# Runden-Steuerung
# ---------------------------------------------------------------------------

def on_game_start() -> None:
    """
    Callback wenn F5 gedrückt wird (State → RUNNING).
    Initialisiert Spielfeld und startet eine neue Episode.
    """
    print("\n[Main] Neue Runde startet …")

    # Spielfeld einmalig analysieren
    try:
        frame = take_single_screenshot()
        if frame is not None:
            game_field.initialize(frame)
        else:
            print("[Main] Kein Screenshot für Spielfeld-Initialisierung.")
            game_field.initialize()   # leeres Fallback-Feld
    except Exception as e:
        print(f"[Main] Fehler bei Spielfeld-Initialisierung: {e}")

    # Neue Episode starten
    agent.start_episode()
    logger.start_episode(agent._episode_count)


def on_round_end(victory: bool) -> None:
    """
    Wird aufgerufen wenn Vision Game-Over oder Sieg erkennt.
    Lässt den Agent lernen und wartet auf den nächsten F5-Druck.
    """
    print(f"\n[Main] Runde beendet ({'Sieg' if victory else 'Niederlage'}).")

    # Spielfeld zurücksetzen für nächste Runde
    game_field.reset()

    # Agent: lernen + Modell speichern
    try:
        agent.end_episode(victory=victory)
    except Exception as e:
        print(f"[Main] Fehler beim Lernen: {e}")

    # Logger
    try:
        logger.end_episode(victory=victory)
        logger.print_stats()
    except Exception as e:
        print(f"[Main] Logging-Fehler: {e}")

    print(f"\n[Main] Drücke {config.HOTKEY_START_GAME} um die nächste Runde zu starten.")
    print(f"       Drücke {config.HOTKEY_STOP_PROGRAM} um das Programm zu beenden.")


def on_program_end() -> None:
    """Callback wenn F6 gedrückt wird – räumt auf und beendet."""
    print("\n[Main] Programm wird beendet …")
    try:
        agent.save()
    except Exception as e:
        print(f"[Main] Fehler beim Speichern des Modells: {e}")
    try:
        logger.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Überwachungs-Loop im Main-Thread
# ---------------------------------------------------------------------------

def monitor_loop() -> None:
    """
    Läuft im Main-Thread.
    Beobachtet den State und ruft on_round_end() auf wenn PAUSED erkannt wird.
    Blockiert bis ENDED gesetzt wird.
    """
    was_running = False

    while not state_manager.is_ended():
        current = state_manager.get()

        if current == GameState.RUNNING:
            was_running = True

        elif current == GameState.PAUSED and was_running:
            # Runde gerade beendet
            was_running = False
            vision = _get_vision()
            victory = vision.victory if vision else False
            on_round_end(victory)

        time.sleep(0.2)   # 200ms Polling – genügt da State-Änderungen selten sind


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════╗")
    print("║     Plants vs. Zombies KI v1.0       ║")
    print("╚══════════════════════════════════════╝")
    print()

    # --- 1. Startup-Dialog + Hotkeys ---
    hotkey_listener = setup(
        on_game_start=on_game_start,
        on_program_end=on_program_end,
    )

    # --- 2. RL-Agent initialisieren (nach Startup-Dialog da ACTIVE_PLANTS jetzt bekannt) ---
    try:
        agent.initialize()
    except Exception as e:
        print(f"[Main] Fehler bei Agent-Initialisierung: {e}")
        sys.exit(1)

    # --- 3. Logger öffnen ---
    try:
        logger.open()
    except Exception as e:
        print(f"[Main] Fehler beim Öffnen des Loggers: {e}")

    # --- 4. Screenshot-Pipeline starten ---
    capture_thread, screenshot_queue = create_capture_pipeline()
    capture_thread.start()

    # --- 5. Vision-Loop starten ---
    vision_thread = threading.Thread(
        target=vision_loop,
        args=(screenshot_queue,),
        name="VisionLoop",
        daemon=True,
    )
    vision_thread.start()

    # --- 6. Agent-Loop starten ---
    agent_thread = threading.Thread(
        target=agent_loop,
        name="AgentLoop",
        daemon=True,
    )
    agent_thread.start()

    # --- 7. Main-Thread: Überwachung ---
    print("[Main] Alle Threads gestartet. Warte auf Spielstart …\n")
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt – beende …")
        state_manager.set(GameState.ENDED)
        on_program_end()

    # --- 8. Aufräumen ---
    hotkey_listener.stop()
    print("[Main] Programm beendet.")


if __name__ == "__main__":
    main()
