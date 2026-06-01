"""
Interactive script to add or update subject metadata for Phase 2B.
Stores: subject_id, body_weight_kg, hand_dominant.

Usage:
  python scripts/imu/add_subject_metadata.py
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "with_hardware"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = METADATA_DIR / "subjects.csv"


def load_existing() -> pd.DataFrame:
    if METADATA_CSV.exists():
        return pd.read_csv(METADATA_CSV)
    return pd.DataFrame(columns=["subject_id", "body_weight_kg", "hand_dominant"])


def save(df: pd.DataFrame):
    df.to_csv(METADATA_CSV, index=False)
    print(f"\n✓ Saved to: {METADATA_CSV}")


def add_or_update_subject():
    df = load_existing()
    
    print("=" * 60)
    print("ADD / UPDATE SUBJECT METADATA")
    print("=" * 60)
    
    if len(df) > 0:
        print("\nExisting subjects:")
        print(df.to_string(index=False))
        print()
    
    subject_id = input("Subject ID (e.g., subject01): ").strip()
    if not subject_id:
        print("Cancelled.")
        return
    
    existing_row = df[df["subject_id"] == subject_id]
    if len(existing_row) > 0:
        print(f"\nSubject '{subject_id}' already exists:")
        print(existing_row.to_string(index=False))
        confirm = input("Update? (y/n): ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return
    
    while True:
        try:
            weight_str = input("Body weight (kg): ").strip()
            body_weight_kg = float(weight_str)
            if 20 <= body_weight_kg <= 200:
                break
            print("  Body weight should be between 20 and 200 kg. Try again.")
        except ValueError:
            print("  Please enter a number.")
    
    while True:
        hand = input("Dominant hand (left/right): ").strip().lower()
        if hand in ("left", "right"):
            hand_dominant = hand
            break
        print("  Please enter 'left' or 'right'.")
    
    new_row = {
        "subject_id": subject_id,
        "body_weight_kg": body_weight_kg,
        "hand_dominant": hand_dominant,
    }
    
    if len(existing_row) > 0:
        df = df[df["subject_id"] != subject_id]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values("subject_id").reset_index(drop=True)
    
    save(df)
    
    print("\nFinal subject list:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    add_or_update_subject()