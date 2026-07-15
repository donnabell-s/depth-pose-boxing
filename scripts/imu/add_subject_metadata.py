"""
Interactive script to add or update subject metadata for Phase 2B.
Stores: subject_id, person_id, body_weight_kg, hand_dominant.

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
        df = pd.read_csv(METADATA_CSV)
        # Backwards-compatibility: add person_id column if missing
        if "person_id" not in df.columns:
            df["person_id"] = ""
        return df
    return pd.DataFrame(columns=["subject_id", "person_id", "body_weight_kg", "hand_dominant"])


def save(df: pd.DataFrame):
    # Ensure column order
    df = df[["subject_id", "person_id", "body_weight_kg", "hand_dominant"]]
    df.to_csv(METADATA_CSV, index=False)
    print(f"\n✓ Saved to: {METADATA_CSV}")


def suggest_person_id(df: pd.DataFrame) -> str:
    """Suggest the next available person_id (e.g., person01, person02)."""
    existing_persons = sorted(set(df["person_id"].dropna().tolist()) - {""})
    if not existing_persons:
        return "person01"
    
    # Try to find next person number
    max_num = 0
    for p in existing_persons:
        if p.startswith("person") and p[6:].isdigit():
            num = int(p[6:])
            max_num = max(max_num, num)
    return f"person{max_num + 1:02d}"


def add_or_update_subject():
    df = load_existing()
    
    print("=" * 60)
    print("ADD / UPDATE SUBJECT METADATA")
    print("=" * 60)
    
    if len(df) > 0:
        print("\nExisting subjects:")
        print(df.to_string(index=False))
        print()
        
        # Show known persons
        known_persons = sorted(set(df["person_id"].dropna().tolist()) - {""})
        if known_persons:
            print(f"Known persons: {', '.join(known_persons)}")
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
    
    # Person ID prompt
    suggested = suggest_person_id(df)
    print(f"\nPerson ID identifies the real human participant.")
    print(f"  - For a NEW person: use a new ID (suggested: {suggested})")
    print(f"  - For a RETURNING person: use their existing person_id (e.g., person01)")
    
    while True:
        person_id = input(f"Person ID [default: {suggested}]: ").strip()
        if not person_id:
            person_id = suggested
        
        # Validate: only alphanumeric, no spaces
        if person_id.replace("_", "").isalnum():
            break
        print("  Person ID should be alphanumeric only (e.g., person01).")
    
    # If this person_id already exists, show their info for consistency check
    same_person_rows = df[df["person_id"] == person_id]
    if len(same_person_rows) > 0:
        print(f"\nOther subjects for {person_id}:")
        print(same_person_rows.to_string(index=False))
        print(f"(Make sure body_weight and hand_dominant match for the same person)")
        print()
    
    # Body weight
    while True:
        try:
            # If person exists, suggest their existing weight as default
            default_weight = None
            if len(same_person_rows) > 0:
                default_weight = same_person_rows["body_weight_kg"].iloc[0]
                weight_str = input(f"Body weight (kg) [default: {default_weight}]: ").strip()
                if not weight_str:
                    body_weight_kg = float(default_weight)
                    break
            else:
                weight_str = input("Body weight (kg): ").strip()
            
            body_weight_kg = float(weight_str)
            if 20 <= body_weight_kg <= 200:
                break
            print("  Body weight should be between 20 and 200 kg. Try again.")
        except ValueError:
            print("  Please enter a number.")
    
    # Dominant hand
    while True:
        if len(same_person_rows) > 0:
            default_hand = same_person_rows["hand_dominant"].iloc[0]
            hand = input(f"Dominant hand (left/right) [default: {default_hand}]: ").strip().lower()
            if not hand:
                hand_dominant = default_hand
                break
        else:
            hand = input("Dominant hand (left/right): ").strip().lower()
        
        if hand in ("left", "right"):
            hand_dominant = hand
            break
        print("  Please enter 'left' or 'right'.")
    
    new_row = {
        "subject_id": subject_id,
        "person_id": person_id,
        "body_weight_kg": body_weight_kg,
        "hand_dominant": hand_dominant,
    }
    
    if len(existing_row) > 0:
        df = df[df["subject_id"] != subject_id]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values(["person_id", "subject_id"]).reset_index(drop=True)
    
    save(df)
    
    print("\nFinal subject list:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    add_or_update_subject()