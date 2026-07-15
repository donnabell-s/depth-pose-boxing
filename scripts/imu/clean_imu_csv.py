"""
Post-process IMU CSV files:
1. Sort by timestamp (fixes out-of-order batches from network jitter)
2. Detect and remove duplicate timestamps
3. Identify sync clap events automatically (first and last claps)
4. Add sync_offset column relative to first clap (clap = 0)
5. Report sample rate statistics with quality verdict
6. Report duration between claps for video sync verification

Handles both:
- Wireless UDP format: gyro columns named gx_raw, gy_raw, gz_raw (raw int16)
- Wired serial format: gyro columns named gx_dps, gy_dps, gz_dps (scaled to deg/s)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


CLAP_DETECTION = {
    "min_accel_magnitude_g": 8.0,
    "gyro_saturation_threshold_raw": 30000,
    "gyro_saturation_threshold_dps": 1800,
    "min_clap_time_ms": 100,
}



def get_gyro_columns(df: pd.DataFrame) -> tuple[list[str], float]:
    if all(c in df.columns for c in ["gx_raw", "gy_raw", "gz_raw"]):
        return ["gx_raw", "gy_raw", "gz_raw"], CLAP_DETECTION["gyro_saturation_threshold_raw"]
    if all(c in df.columns for c in ["gx_dps", "gy_dps", "gz_dps"]):
        return ["gx_dps", "gy_dps", "gz_dps"], CLAP_DETECTION["gyro_saturation_threshold_dps"]
    raise ValueError(
        "CSV does not contain expected gyro columns. "
        "Expected (gx_raw, gy_raw, gz_raw) or (gx_dps, gy_dps, gz_dps)."
    )


def detect_all_claps(df: pd.DataFrame) -> list[int]:
    """Find first and last clap events only.
    
    Logic: claps bookend the recording. The first significant impact event
    is the start clap, the last is the end clap. Everything in between is
    treated as data (punches, etc).
    """
    accel_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    gyro_cols, gyro_threshold = get_gyro_columns(df)
    gyro_max = df[gyro_cols].abs().max(axis=1)
    
    is_impact_event = (
        (df["timestamp_ms"] > CLAP_DETECTION["min_clap_time_ms"]) &
        (
            (accel_mag > CLAP_DETECTION["min_accel_magnitude_g"]) |
            (gyro_max > gyro_threshold)
        )
    )
    
    candidate_indices = df.index[is_impact_event].tolist()
    if not candidate_indices:
        return []
    
    # Group consecutive samples (within 500ms) into single events
    grouped = [candidate_indices[0]]
    for idx in candidate_indices[1:]:
        time_since_last = df.loc[idx, "timestamp_ms"] - df.loc[grouped[-1], "timestamp_ms"]
        if time_since_last >= 500:  # 500ms minimum between distinct events
            grouped.append(idx)
    
    # Return only the first and last grouped events
    if len(grouped) >= 2:
        return [grouped[0], grouped[-1]]
    elif len(grouped) == 1:
        return [grouped[0]]
    return []

def assess_quality(stats: dict) -> tuple[str, list[str]]:
    issues = []
    
    rate = stats["actual_rate_hz"]
    mean_interval = stats["mean_interval_ms"]
    largest_gap = stats["largest_gap_ms"]
    gap_count = stats["gaps_detected"]
    duration_s = stats["duration_s"]
    
    gaps_per_10s = (gap_count / duration_s) * 10 if duration_s > 0 else 0
    
    if rate >= 180 and mean_interval <= 6 and largest_gap < 20 and gaps_per_10s <= 2:
        verdict = "EXCELLENT (production-ready)"
    elif rate >= 100 and mean_interval <= 15 and largest_gap < 50 and gaps_per_10s <= 10:
        verdict = "ACCEPTABLE (usable for data collection)"
    elif rate >= 50 and mean_interval <= 30 and largest_gap < 100 and gaps_per_10s <= 15:
        verdict = "MARGINAL (document limitations)"
    else:
        verdict = "UNACCEPTABLE (fix network before recording)"
    
    if rate < 150:
        issues.append(f"Effective rate {rate:.1f} Hz is below target 150 Hz")
    if mean_interval > 10:
        issues.append(f"Mean interval {mean_interval:.1f} ms is high (target <10 ms)")
    if largest_gap > 50:
        issues.append(f"Largest gap {largest_gap:.0f} ms could miss punch peaks (target <50 ms)")
    if gaps_per_10s > 5:
        issues.append(f"{gaps_per_10s:.1f} gaps per 10 seconds (target <5)")
    
    return verdict, issues


def clean_imu_csv(input_path: Path, output_path: Path | None = None,
                  detect_sync: bool = True) -> dict:
    df = pd.read_csv(input_path)
    n_original = len(df)
    
    if len(df) == 0:
        raise ValueError(f"Input file is empty: {input_path}")
    
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp_ms"], keep="first").reset_index(drop=True)
    n_duplicates = n_before_dedup - len(df)
    
    # Detect all clap events
    clap_indices = []
    first_clap_time_ms = None
    last_clap_time_ms = None
    clap_to_clap_duration_ms = None
    
    if detect_sync:
        try:
            clap_indices = detect_all_claps(df)
            if clap_indices:
                first_clap_time_ms = float(df.loc[clap_indices[0], "timestamp_ms"])
                df["sync_offset_ms"] = df["timestamp_ms"] - first_clap_time_ms
                if len(clap_indices) >= 2:
                    last_clap_time_ms = float(df.loc[clap_indices[-1], "timestamp_ms"])
                    clap_to_clap_duration_ms = last_clap_time_ms - first_clap_time_ms
            else:
                df["sync_offset_ms"] = np.nan
        except ValueError as e:
            print(f"  Warning: skipping clap detection ({e})")
            df["sync_offset_ms"] = np.nan
    
    # Rate statistics
    if len(df) > 1:
        intervals = df["timestamp_ms"].diff().dropna()
        median_interval_ms = intervals.median()
        mean_interval_ms = intervals.mean()
        std_interval_ms = intervals.std()
        actual_rate_hz = 1000.0 / median_interval_ms if median_interval_ms > 0 else 0
        gap_threshold = median_interval_ms * 2.5
        n_gaps = (intervals > gap_threshold).sum()
        largest_gap_ms = intervals.max()
        total_duration_s = (df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0
    else:
        median_interval_ms = mean_interval_ms = std_interval_ms = actual_rate_hz = 0
        n_gaps = 0
        largest_gap_ms = 0
        total_duration_s = 0
    
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_cleaned.csv")
    df.to_csv(output_path, index=False)
    
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "original_rows": n_original,
        "duplicates_removed": n_duplicates,
        "final_rows": len(df),
        "duration_s": total_duration_s,
        "median_interval_ms": median_interval_ms,
        "mean_interval_ms": mean_interval_ms,
        "std_interval_ms": std_interval_ms,
        "actual_rate_hz": actual_rate_hz,
        "gaps_detected": int(n_gaps),
        "largest_gap_ms": largest_gap_ms,
        "n_claps_detected": len(clap_indices),
        "first_clap_time_ms": first_clap_time_ms,
        "last_clap_time_ms": last_clap_time_ms,
        "clap_to_clap_duration_ms": clap_to_clap_duration_ms,
    }
    return stats


def print_stats(stats: dict):
    print(f"\nInput:  {stats['input']}")
    print(f"Output: {stats['output']}\n")
    print(f"Rows: {stats['original_rows']} → {stats['final_rows']}  "
          f"(removed {stats['duplicates_removed']} duplicates)")
    print(f"Duration: {stats['duration_s']:.2f} seconds")
    print(f"Median sample interval: {stats['median_interval_ms']:.2f} ms")
    print(f"Mean sample interval:   {stats['mean_interval_ms']:.2f} ms")
    print(f"Std of intervals:       {stats['std_interval_ms']:.2f} ms")
    print(f"Effective sample rate:  {stats['actual_rate_hz']:.1f} Hz")
    print(f"Network gaps detected:  {stats['gaps_detected']}")
    print(f"Largest gap:            {stats['largest_gap_ms']:.1f} ms")
    
    # Sync information
    n_claps = stats["n_claps_detected"]
    if n_claps == 0:
        print(f"\nNo clap events detected.")
    elif n_claps == 1:
        print(f"\n1 clap detected — using single-clap sync.")
        print(f"  First clap at:  {stats['first_clap_time_ms']:.1f} ms")
        print(f"  → sync_offset_ms column added (first clap = 0)")
    else:
        print(f"\n{n_claps} clap events detected — using two-clap sync.")
        print(f"  First clap at:  {stats['first_clap_time_ms']:.1f} ms")
        print(f"  Last clap at:   {stats['last_clap_time_ms']:.1f} ms")
        print(f"  Duration between claps: {stats['clap_to_clap_duration_ms']:.1f} ms")
        print(f"  → sync_offset_ms column added (first clap = 0)")
        print(f"\nTo verify sync: find both claps in video, compute video clap-to-clap duration.")
        print(f"If it matches {stats['clap_to_clap_duration_ms']:.1f} ms (within ~30 ms), sync is good.")
    
    verdict, issues = assess_quality(stats)
    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print("=" * 60)
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues detected.")


def main():
    parser = argparse.ArgumentParser(description="Clean and sync-process IMU CSV files")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-sync", action="store_true")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        return
    
    try:
        stats = clean_imu_csv(args.input, args.output, detect_sync=not args.no_sync)
        print_stats(stats)
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()