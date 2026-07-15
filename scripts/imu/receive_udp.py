"""
Read IMU data from ESP/Feather over WiFi UDP.

Produces output identical to receive_serial.py (the validated wired reference)
so both can be processed by the same downstream pipeline
(clean_imu_csv.py, plot_imu.py, notebooks/regression preprocessing).
"""

import argparse
import csv
import socket
import time
from pathlib import Path

import keyboard


# Network configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 4210
SOCKET_TIMEOUT = 0.1

# imu_wifi.ino configures the gyro for ±2000dps (16.4 LSB/dps), same as the
# wired firmware (imu.ino). It sends raw int16 counts instead of pre-converted
# floats to keep the batched UDP packet small, so the conversion happens here.
GYRO_SENSITIVITY_LSB_PER_DPS = 16.4

# Both firmwares sample at 200Hz (sampleInterval = 5000us)
EXPECTED_RATE_HZ = 200

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/imu/ → project root
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "with_hardware"

# Session metadata — set these before each recording (or override via CLI flags)
SUBJECT_ID = "test"
PUNCH_TYPE = "test"
DISTANCE_M = 1
HAND = "left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive IMU data over WiFi UDP")
    parser.add_argument("--subject", default=SUBJECT_ID)
    parser.add_argument("--punch-type", default=PUNCH_TYPE)
    parser.add_argument("--distance", type=int, default=DISTANCE_M)
    parser.add_argument("--hand", default=HAND)
    parser.add_argument("--output", type=Path, default=None,
                         help="Explicit output CSV path (overrides --subject/--punch-type/--distance/--hand naming)")
    return parser.parse_args()


def print_reliability_stats(sample_timestamps_ms: list[float], total_time_s: float) -> None:
    expected_samples = int(total_time_s * EXPECTED_RATE_HZ)
    actual_samples = len(sample_timestamps_ms)
    delivery_rate = (actual_samples / expected_samples * 100.0) if expected_samples > 0 else 0.0

    print("\n" + "=" * 60)
    print("WIFI RELIABILITY STATS")
    print("=" * 60)
    print(f"  Expected samples (@{EXPECTED_RATE_HZ}Hz): {expected_samples}")
    print(f"  Actual samples received:     {actual_samples}")
    print(f"  Delivery rate:                {delivery_rate:.1f}%")

    if actual_samples > 1:
        ordered = sorted(sample_timestamps_ms)
        intervals = [b - a for a, b in zip(ordered, ordered[1:])]
        avg_interval = sum(intervals) / len(intervals)
        max_gap = max(intervals)
        print(f"  Average sample interval:     {avg_interval:.2f} ms")
        print(f"  Maximum gap between samples:  {max_gap:.2f} ms")
    print("=" * 60)


def main():
    args = parse_args()

    if args.output is not None:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        session_name = f"{args.punch_type}_{args.distance}m_{args.hand}"
        session_dir = DATA_DIR / args.subject
        session_dir.mkdir(parents=True, exist_ok=True)
        output_path = session_dir / f"{session_name}_imu.csv"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(SOCKET_TIMEOUT)

    print(f"Recording session: {args.punch_type}_{args.distance}m_{args.hand}")
    print(f"Subject:           {args.subject}")
    print(f"Output file:       {output_path}")
    print(f"\nWaiting for IMU data on port {UDP_PORT}...")
    print("Press Q to stop recording.\n")

    sample_count = 0
    packet_count = 0
    laptop_start_time = None
    window_start = None
    window_samples = 0
    sample_timestamps_ms: list[float] = []

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp_ms",
            "ax", "ay", "az",
            "gx_dps", "gy_dps", "gz_dps",
            "mx_raw", "my_raw", "mz_raw",
        ])

        try:
            while True:
                if keyboard.is_pressed("q"):
                    print("\nStopping recording...")
                    break

                try:
                    data, addr = sock.recvfrom(2048)
                    decoded = data.decode(errors="ignore").strip()
                    if not decoded:
                        continue

                    packet_count += 1
                    lines = [line for line in decoded.split("\n") if line.strip()]
                    if not lines:
                        continue

                    now = time.perf_counter()
                    if laptop_start_time is None:
                        laptop_start_time = now
                        window_start = now
                        print("First packet received — recording started.")

                    # Parse every sample in the batch first so we know how long
                    # the batch spans (ESP's relMicros clock), then anchor the
                    # LAST sample to the packet's arrival time — the ESP sends
                    # the batch immediately after collecting that last sample,
                    # so arrival time is the best estimate of when it was taken.
                    parsed = []
                    for line in lines:
                        parts = line.strip().split(",")
                        if len(parts) != 10:
                            continue
                        try:
                            rel_micros = int(parts[0])
                            values = [float(p) for p in parts[1:]]
                            parsed.append((rel_micros, values))
                        except (ValueError, IndexError):
                            continue

                    if not parsed:
                        continue

                    arrival_ms = (now - laptop_start_time) * 1000.0
                    batch_duration_ms = parsed[-1][0] / 1000.0
                    batch_anchor_ms = arrival_ms - batch_duration_ms

                    for rel_micros, values in parsed:
                        sample_ts = batch_anchor_ms + (rel_micros / 1000.0)
                        ax, ay, az, gx_raw, gy_raw, gz_raw, mx, my, mz = values
                        gx_dps = round(gx_raw / GYRO_SENSITIVITY_LSB_PER_DPS, 2)
                        gy_dps = round(gy_raw / GYRO_SENSITIVITY_LSB_PER_DPS, 2)
                        gz_dps = round(gz_raw / GYRO_SENSITIVITY_LSB_PER_DPS, 2)
                        writer.writerow([
                            f"{sample_ts:.3f}",
                            round(ax, 2), round(ay, 2), round(az, 2),
                            gx_dps, gy_dps, gz_dps,
                            mx, my, mz,
                        ])
                        sample_count += 1
                        window_samples += 1
                        sample_timestamps_ms.append(sample_ts)

                    if now - window_start >= 2.0:
                        rate = window_samples / (now - window_start)
                        elapsed = now - laptop_start_time
                        print(f"  Samples: {sample_count} | "
                              f"Packets: {packet_count} | "
                              f"Elapsed: {elapsed:.1f}s | "
                              f"Instant rate: {rate:.1f} Hz")
                        window_start = now
                        window_samples = 0

                except socket.timeout:
                    pass
                except UnicodeDecodeError:
                    continue
        finally:
            sock.close()

    if laptop_start_time is not None:
        total_time = time.perf_counter() - laptop_start_time
        actual_rate = sample_count / total_time if total_time > 0 else 0
        print(f"\nRecording complete.")
        print(f"  Total samples:  {sample_count}")
        print(f"  Total packets:  {packet_count}")
        print(f"  Total duration: {total_time:.2f} seconds")
        print(f"  Actual rate:    {actual_rate:.1f} Hz")
        print(f"  Saved to:       {output_path}")

        print_reliability_stats(sample_timestamps_ms, total_time)
    else:
        print("\nNo data received.")


if __name__ == "__main__":
    main()
