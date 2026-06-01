import socket
import keyboard
import csv
import time
from pathlib import Path

# Network configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 4210

# Storage configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/imu/ → project root
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "with_hardware"

# Session metadata — set these before each recording
SUBJECT_ID = "test"
PUNCH_TYPE = "test"
DISTANCE_M = 1
HAND = "left"

# Build session name and output path
session_name = f"{PUNCH_TYPE}_{DISTANCE_M}m_{HAND}"
session_dir = DATA_DIR / SUBJECT_ID
session_dir.mkdir(parents=True, exist_ok=True)
output_path = session_dir / f"{session_name}_imu.csv"


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.1)

    print(f"Recording session: {session_name}")
    print(f"Subject:           {SUBJECT_ID}")
    print(f"Output file:       {output_path}")
    print(f"\nWaiting for IMU data on port {UDP_PORT}...")
    print("Press Q to stop recording.\n")

    sample_count = 0
    packet_count = 0
    start_time = None
    laptop_start_time = None
    window_start = None
    window_samples = 0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp_ms",
            "ax", "ay", "az",
            "gx_raw", "gy_raw", "gz_raw",
            "mx_raw", "my_raw", "mz_raw",
        ])

        try:
            while True:
                if keyboard.is_pressed("q"):
                    print("\nStopping recording...")
                    break

                try:
                    data, addr = sock.recvfrom(2048)
                    decoded = data.decode().strip()
                    if not decoded:
                        continue
                    
                    packet_count += 1
                    lines = decoded.split("\n")
                    
                    # First packet anchors the timeline
                    now = time.perf_counter()
                    if laptop_start_time is None:
                        laptop_start_time = now
                        window_start = now
                        print("First packet received — recording started.")
                    
                    # Parse each sample in the batch
                    # First line uses ESP's relative microseconds within batch
                    # We anchor batch start time to laptop's perf_counter
                    batch_anchor_ms = (now - laptop_start_time) * 1000.0
                    
                    for line in lines:
                        parts = line.strip().split(",")
                        if len(parts) != 10:
                            continue
                        
                        try:
                            rel_micros = int(parts[0])
                            rel_ms = rel_micros / 1000.0
                            timestamp_ms = batch_anchor_ms + rel_ms - (rel_ms if line == lines[0] else 0)
                            
                            # Simpler: just use batch_anchor as start of batch
                            # and add the relative offset
                            sample_ts = batch_anchor_ms + rel_ms
                            
                            values = [float(p) for p in parts[1:]]
                            writer.writerow([f"{sample_ts:.3f}"] + values)
                            sample_count += 1
                            window_samples += 1
                        except (ValueError, IndexError):
                            continue
                    
                    # Report instantaneous rate every 2 seconds
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
    else:
        print("\nNo data received.")


if __name__ == "__main__":
    main()