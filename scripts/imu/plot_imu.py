"""
Plot IMU acceleration and gyro signals to visually verify a recording.
Shows:
- Acceleration magnitude over time (claps and punches as spikes)
- Per-axis acceleration
- Gyro magnitude
- Markers for detected clap events

Usage:
  python scripts/plot_imu.py path/to/imu_cleaned.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_gyro_columns(df: pd.DataFrame) -> list[str]:
    if all(c in df.columns for c in ["gx_raw", "gy_raw", "gz_raw"]):
        return ["gx_raw", "gy_raw", "gz_raw"]
    if all(c in df.columns for c in ["gx_dps", "gy_dps", "gz_dps"]):
        return ["gx_dps", "gy_dps", "gz_dps"]
    raise ValueError("CSV does not contain expected gyro columns.")


def plot_imu(csv_path: Path, output_path: Path | None = None):
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print(f"Error: file is empty: {csv_path}")
        return
    
    # Use sync_offset_ms if available (relative to clap), otherwise timestamp_ms
    if "sync_offset_ms" in df.columns and not df["sync_offset_ms"].isna().all():
        time_col = "sync_offset_ms"
        time_label = "Time relative to first clap (ms)"
    else:
        time_col = "timestamp_ms"
        time_label = "Time from recording start (ms)"
    
    time = df[time_col].values
    
    # Acceleration magnitude
    accel_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    
    # Gyro magnitude
    gyro_cols = get_gyro_columns(df)
    gyro_mag = np.sqrt(
        df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2
    )
    
    # Detect claps using the same logic as cleanup script
    # Claps bookend the recording: first impact and last impact
    accel_threshold = 8.0  # g  (matches cleanup script CLAP_DETECTION)
    is_impact = accel_mag > accel_threshold
    impact_times = time[is_impact]

    # Group consecutive samples into events
    event_times = []
    min_separation_ms = 500
    last_event_time = -float("inf")
    for t in impact_times:
        if t - last_event_time >= min_separation_ms:
            event_times.append(t)
            last_event_time = t

    # Take only first and last (the bookend claps)
    if len(event_times) >= 2:
        clap_groups = [event_times[0], event_times[-1]]
    elif len(event_times) == 1:
        clap_groups = event_times
    else:
        clap_groups = []
    
    # Create the figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Acceleration magnitude
    axes[0].plot(time, accel_mag, linewidth=0.8, color="navy")
    axes[0].axhline(accel_threshold, color="red", linestyle="--", alpha=0.5,
                    label=f"Clap threshold ({accel_threshold}g)")
    axes[0].axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="1g (gravity)")
    
    for ct in clap_groups:
        axes[0].axvline(ct, color="orange", linestyle="-", alpha=0.6, linewidth=1)
    
    axes[0].set_ylabel("Accel magnitude (g)")
    axes[0].set_title(f"IMU recording: {csv_path.name}")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Per-axis acceleration
    axes[1].plot(time, df["ax"], linewidth=0.6, label="ax", alpha=0.8)
    axes[1].plot(time, df["ay"], linewidth=0.6, label="ay", alpha=0.8)
    axes[1].plot(time, df["az"], linewidth=0.6, label="az", alpha=0.8)
    axes[1].set_ylabel("Per-axis accel (g)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Gyro magnitude
    axes[2].plot(time, gyro_mag, linewidth=0.8, color="darkgreen")
    axes[2].set_ylabel(f"Gyro magnitude ({gyro_cols[0].split('_')[1]})")
    axes[2].grid(alpha=0.3)
    
    # Plot 4: Per-axis gyro
    axes[3].plot(time, df[gyro_cols[0]], linewidth=0.6, label=gyro_cols[0], alpha=0.8)
    axes[3].plot(time, df[gyro_cols[1]], linewidth=0.6, label=gyro_cols[1], alpha=0.8)
    axes[3].plot(time, df[gyro_cols[2]], linewidth=0.6, label=gyro_cols[2], alpha=0.8)
    axes[3].set_ylabel(f"Per-axis gyro ({gyro_cols[0].split('_')[1]})")
    axes[3].set_xlabel(time_label)
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(alpha=0.3)
    
    # Annotate clap markers on the top plot
    for i, ct in enumerate(clap_groups):
        label = "Clap 1 (sync start)" if i == 0 else f"Clap {i+1}"
        axes[0].annotate(label, xy=(ct, accel_mag.max() * 0.95),
                         xytext=(5, 0), textcoords="offset points",
                         fontsize=9, color="orange",
                         rotation=0, va="top")
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_plot.png")
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    
    # Also display interactive
    plt.show()
    
    # Summary
    print(f"\nDetected {len(clap_groups)} clap events at times (ms):")
    for i, ct in enumerate(clap_groups):
        print(f"  Clap {i+1}: {ct:.1f} ms")
    if len(clap_groups) >= 2:
        duration = clap_groups[-1] - clap_groups[0]
        print(f"\nDuration between first and last clap: {duration:.1f} ms ({duration/1000:.2f} s)")


def main():
    parser = argparse.ArgumentParser(description="Plot IMU signals from a CSV file")
    parser.add_argument("input", type=Path, help="Path to cleaned IMU CSV")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: input_plot.png)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: file not found: {args.input}")
        return
    
    plot_imu(args.input, args.output)


if __name__ == "__main__":
    main()