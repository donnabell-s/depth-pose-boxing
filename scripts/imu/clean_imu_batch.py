"""
Batch version of clean_imu_csv.py.
Processes all IMU CSV files in a directory and reports a summary.


# Process all IMU CSVs in a folder recursively
python clean_imu_batch.py data/raw/with_hardware

# Or for just one subject
python clean_imu_batch.py data/raw/with_hardware/test

# Non-recursive
python clean_imu_batch.py data/raw/with_hardware --no-recursive
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_imu_csv import clean_imu_csv, assess_quality


def batch_clean(input_dir: Path, recursive: bool = True, detect_sync: bool = True,
                skip_cleaned: bool = True) -> list[dict]:
    """Clean every IMU CSV in input_dir. Returns a list of stats per file."""
    pattern = "**/*_imu.csv" if recursive else "*_imu.csv"
    csv_files = sorted(input_dir.glob(pattern))
    
    if skip_cleaned:
        # Don't process files already named *_cleaned.csv
        csv_files = [f for f in csv_files if not f.stem.endswith("_cleaned")]
    
    if not csv_files:
        print(f"No IMU CSV files found in {input_dir}")
        return []
    
    print(f"Found {len(csv_files)} IMU CSV files to process\n")
    
    all_stats = []
    for i, csv_path in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing: {csv_path.name}")
        try:
            stats = clean_imu_csv(csv_path, output_path=None, detect_sync=detect_sync)
            verdict, issues = assess_quality(stats)
            stats["verdict"] = verdict
            stats["issues"] = issues
            all_stats.append(stats)
            
            # Brief inline result
            print(f"  Rate: {stats['actual_rate_hz']:.1f} Hz | "
                  f"Mean interval: {stats['mean_interval_ms']:.1f} ms | "
                  f"Gaps: {stats['gaps_detected']} | "
                  f"Largest gap: {stats['largest_gap_ms']:.0f} ms")
            print(f"  → {verdict}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue
    
    return all_stats


def print_summary(all_stats: list[dict]):
    if not all_stats:
        return
    
    print("=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    
    # Verdict counts
    verdict_counts = {}
    for s in all_stats:
        v = s["verdict"].split(" (")[0]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    
    print("\nVerdict distribution:")
    for verdict, count in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {verdict:<15}: {count} files")
    
    # Files needing attention
    problem_files = [s for s in all_stats if "UNACCEPTABLE" in s["verdict"] or "MARGINAL" in s["verdict"]]
    if problem_files:
        print(f"\nFiles below ACCEPTABLE ({len(problem_files)}):")
        for s in problem_files:
            print(f"  - {Path(s['input']).name}: {s['verdict']}")
    
    # Aggregate statistics
    print(f"\nAggregate stats across {len(all_stats)} files:")
    rates = [s["actual_rate_hz"] for s in all_stats]
    means = [s["mean_interval_ms"] for s in all_stats]
    gaps = [s["gaps_detected"] for s in all_stats]
    durations = [s["duration_s"] for s in all_stats]
    
    print(f"  Total duration recorded: {sum(durations):.1f} seconds")
    print(f"  Mean rate across files:  {sum(rates)/len(rates):.1f} Hz")
    print(f"  Mean interval (avg):     {sum(means)/len(means):.1f} ms")
    print(f"  Total gaps across files: {sum(gaps)}")


def export_summary_csv(all_stats: list[dict], output_csv: Path):
    """Save a CSV summary of all processed files."""
    rows = []
    for s in all_stats:
        clap_duration = s["clap_to_clap_duration_ms"]
        rows.append({
            "file": Path(s["input"]).name,
            "rows": s["final_rows"],
            "duration_s": round(s["duration_s"], 2),
            "median_interval_ms": round(s["median_interval_ms"], 2),
            "mean_interval_ms": round(s["mean_interval_ms"], 2),
            "std_interval_ms": round(s["std_interval_ms"], 2),
            "effective_rate_hz": round(s["actual_rate_hz"], 1),
            "gaps_detected": s["gaps_detected"],
            "largest_gap_ms": round(s["largest_gap_ms"], 1),
            "n_claps_detected": s["n_claps_detected"],
            "clap_to_clap_duration_ms": round(clap_duration, 1) if clap_duration is not None else None,
            "verdict": s["verdict"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSummary CSV saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Batch cleanup of IMU CSV files")
    parser.add_argument("input_dir", type=Path, help="Directory containing IMU CSV files")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Only process top-level directory, not subfolders")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip clap detection")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="Path to save batch summary CSV")
    parser.add_argument("--include-cleaned", action="store_true",
                        help="Also process files already ending in _cleaned.csv")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: directory not found: {args.input_dir}")
        return
    
    all_stats = batch_clean(
        args.input_dir,
        recursive=not args.no_recursive,
        detect_sync=not args.no_sync,
        skip_cleaned=not args.include_cleaned,
    )
    
    print_summary(all_stats)
    
    if args.summary_csv:
        export_summary_csv(all_stats, args.summary_csv)
    else:
        # Default summary location
        default_csv = args.input_dir / "imu_quality_summary.csv"
        export_summary_csv(all_stats, default_csv)


if __name__ == "__main__":
    main()