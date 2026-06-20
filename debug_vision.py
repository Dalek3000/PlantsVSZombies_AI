"""
debug_vision.py – Visueller Test von Kalibrierung und Erkennung

Führe dieses Script aus WÄHREND PvZ läuft und das Spielfeld sichtbar ist.
Es macht einen Screenshot, zeichnet alle Erkennungen als Overlay drauf
und speichert das Ergebnis als PNG – so siehst du sofort ob alles stimmt.

Aufruf:
    python debug_vision.py
"""

import cv2
import numpy as np
from pathlib import Path

# --- Kalibrierung laden (muss vorher einmalig gemacht worden sein) ---
from config import runtime, load_calibration, FIELD_ROWS_DEFAULT, FIELD_COLS
from screen_capture import take_single_screenshot, save_debug_screenshot
from game_field import game_field
from vision import analyze_frame, draw_vision_overlay


def run_debug() -> None:
    print("=" * 55)
    print("  PvZ Debug-Overlay")
    print("=" * 55)

    # --- Schritt 1: Kalibrierung laden ---
    cal = load_calibration()
    if cal is None:
        print("[!] Keine Kalibrierung gefunden.")
        print("    Starte zuerst main.py und führe die Kalibrierung durch.")
        return

    runtime.field_top_left     = tuple(cal["field_top_left"])
    runtime.field_bottom_right = tuple(cal["field_bottom_right"])
    runtime.cell_width         = cal["cell_width"]
    runtime.cell_height        = cal["cell_height"]
    runtime.field_rows         = cal["field_rows"]
    runtime.window_left        = cal.get("window_left", 0)
    runtime.window_top         = cal.get("window_top",  0)
    print(f"[✓] Kalibrierung geladen: {runtime.field_rows} Reihen × {FIELD_COLS} Spalten")
    print(f"    Oben-links:   {runtime.field_top_left}")
    print(f"    Unten-rechts: {runtime.field_bottom_right}")
    print(f"    Zellgröße:    {runtime.cell_width:.1f} × {runtime.cell_height:.1f} px")

    # --- Schritt 2: Warten bis Spiel wieder läuft ---
    print()
    print("  Wechsle jetzt zu PvZ und stelle sicher dass das Spiel läuft (nicht pausiert).")
    print("  Drücke dann F8 um den Screenshot auszulösen.")
    import keyboard
    keyboard.wait("F8")

    # --- Schritt 3: Screenshot machen ---
    print("\n[...] Mache Screenshot von PvZ-Fenster ...")
    frame = take_single_screenshot()
    if frame is None:
        print("[!] Kein Screenshot möglich – ist PvZ geöffnet?")
        return
    print(f"[✓] Screenshot: {frame.shape[1]}×{frame.shape[0]} px")

    # Rohes Bild speichern
    save_debug_screenshot(frame, "debug_raw.png")
    print(f"[✓] Rohbild gespeichert: saves/debug_raw.png")

    # --- Schritt 4: Spielfeld analysieren ---
    print("\n[...] Analysiere Spielfeld ...")
    game_field.initialize(frame)

    # Spielfeld-Overlay
    field_overlay = game_field.draw_debug_overlay(frame)
    save_debug_screenshot(field_overlay, "debug_field.png")
    print("[✓] Spielfeld-Overlay gespeichert: saves/debug_field.png")
    print("    Legende: G=Gras  W=Wasser  R=Dach  ?=Unbekannt")
    print("             X auf Zelle = als belegt markiert")

    # --- Schritt 5: Vision-Erkennung ---
    print("\n[...] Führe Bilderkennung aus ...")
    result = analyze_frame(frame)

    print(f"\n  Erkannte Sonnen:      {len(result.suns)}")
    for i, sun in enumerate(result.suns):
        print(f"    Sonne {i}: Bildschirm ({sun.screen_x}, {sun.screen_y})")

    print(f"  Sonnenstand (OCR):    {result.sun_count}")

    print(f"  Zombies pro Reihe:    ", end="")
    print(" ".join(f"R{i}={'JA' if z else 'nein'}" for i, z in enumerate(result.zombie_rows)))

    print(f"  Baubare Slots:        ", end="")
    print(" ".join(f"S{i}={'JA' if p else 'nein'}" for i, p in enumerate(result.plantable_slots)))

    print(f"  Game-Over erkannt:    {'JA' if result.game_over else 'nein'}")
    print(f"  Sieg erkannt:         {'JA' if result.victory else 'nein'}")

    # Vision-Overlay
    vision_overlay = draw_vision_overlay(frame, result)
    save_debug_screenshot(vision_overlay, "debug_vision.png")
    print("\n[✓] Vision-Overlay gespeichert: saves/debug_vision.png")

    # --- Schritt 6: Kombiniertes Overlay ---
    # Spielfeld + Vision zusammen
    combined = game_field.draw_debug_overlay(vision_overlay)

    # HUD-Bereiche markieren (Sonnenstand + Slots)
    h, w = frame.shape[:2]

    # Sonnenstand-Region (gelber Rahmen)
    cv2.rectangle(combined,
        (int(w * 0.01), int(h * 0.08)),
        (int(w * 0.09), int(h * 0.17)),
        (0, 215, 255), 2)
    cv2.putText(combined, "Sonnen-OCR", (int(w * 0.01), int(h * 0.08) - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    # Slot-Regionen (lila Rahmen)
    import config
    n_slots = len(config.ACTIVE_PLANTS) if config.ACTIVE_PLANTS else 6
    slot_start_x = int(w * 0.08)
    slot_width   = int(w * 0.068)
    slot_gap     = int(w * 0.005)
    bar_top      = int(h * 0.01)
    bar_bottom   = int(h * 0.12)

    for i in range(n_slots):
        x1 = slot_start_x + i * (slot_width + slot_gap)
        x2 = x1 + slot_width
        cv2.rectangle(combined, (x1, bar_top), (x2, bar_bottom), (255, 0, 255), 2)
        cv2.putText(combined, f"S{i}", (x1 + 3, bar_top + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    save_debug_screenshot(combined, "debug_combined.png")
    print("[✓] Kombiniertes Overlay gespeichert: saves/debug_combined.png")

    # --- Zusammenfassung ---
    print()
    print("=" * 55)
    print("  Dateien zum Prüfen:")
    print(f"  {runtime.save_dir / 'debug_raw.png'}      ← Rohbild")
    print(f"  {runtime.save_dir / 'debug_field.png'}    ← Spielfeld-Raster")
    print(f"  {runtime.save_dir / 'debug_vision.png'}   ← Erkennungen")
    print(f"  {runtime.save_dir / 'debug_combined.png'} ← Alles zusammen")
    print()
    print("  Was du prüfen solltest:")
    print("  [1] debug_field.png  – Liegt das Raster NUR auf dem Grasfeld (nicht auf der Leiste)?")
    print("      Falls nicht: Kalibrierung wiederholen – Punkt 1 muss auf die")
    print("      obere linke Ecke der ersten GRAS-Zelle, direkt unter der Leiste.")
    print("  [2] debug_combined.png – Sonnen gelb markiert? Slots lila umrandet?")
    print("  [3] Sonnenstand korrekt? (OCR-Wert oben in der Konsole)")
    print("  [4] Slots richtig als baubar/nicht-baubar erkannt?")
    print()
    print("  Falls etwas nicht stimmt:")
    print("  → Raster falsch:  Kalibrierung wiederholen (main.py starten)")
    print("  → Farben falsch:  HSV_RANGES in config.py anpassen")
    print("  → Slots falsch:   slot_start_x / slot_width in vision.py anpassen")
    print("=" * 55)


if __name__ == "__main__":
    # Speicherpfad setzen (gleicher Default wie in config.py)
    from config import DEFAULT_SAVE_DIR
    runtime.save_dir = DEFAULT_SAVE_DIR
    DEFAULT_SAVE_DIR.mkdir(exist_ok=True)

    run_debug()
