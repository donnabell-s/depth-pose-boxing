"""
Read IMU data from ESP/Feather via USB serial connection.
Replaces the UDP-based receive.py for wired data collection.

Features:
- Pre-flight check to confirm IMU is responding before recording starts
- Live frozen-sensor detection during recording (warns if sensor stops responding)
- Auto-detects ESP/Feather COM port
"""

import csv
import time
from pathlib import Path

import serial
import serial.tools.list_ports
import keyboard


# Serial configuration
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.1

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/imu/ → project root
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "with_hardware"

# Session metadata — set these before each recording
SUBJECT_ID = "subject04"
PUNCH_TYPE = "uppercut"
DISTANCE_M = 1
HAND = "left"

# Pre-flight check parameters
PREFLIGHT_DURATION_S = 5
PREFLIGHT_MIN_VARIATION = 0.1  # minimum range in any axis to consider IMU responsive

# Frozen sensor detection during recording
FROZEN_THRESHOLD = 10  # number of consecutive identical samples before warning


def find_serial_port() -> str | None:
    """Auto-detect the ESP/Feather COM port."""
    ports = serial.tools.list_ports.comports()
    
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device} - {p.description}")
    
    for p in ports:
        desc = (p.description or "").lower()
        if any(keyword in desc for keyword in ["esp", "feather", "ch340", "cp210", "ftdi", "huzzah"]):
            return p.device
    
    if ports:
        return ports[0].device
    return None


def preflight_check(ser: serial.Serial) -> bool:
    """Run a pre-flight check to confirm IMU is responsive.
    
    Reads samples for PREFLIGHT_DURATION_S seconds while the user waves the IMU,
    then verifies sufficient variation across axes.
    
    Returns True if IMU is healthy, False otherwise.
    """
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK")
    print("=" * 60)
    print(f"Wave the IMU around for {PREFLIGHT_DURATION_S} seconds...")
    print("(Move it in all directions to verify all axes are working)\n")
    
    samples = []
    start = time.perf_counter()
    
    while time.perf_counter() - start < PREFLIGHT_DURATION_S:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 9:
            continue
        try:
            values = [float(p) for p in parts]
            samples.append(values)
        except ValueError:
            continue
        
        # Live countdown
        elapsed = time.perf_counter() - start
        remaining = PREFLIGHT_DURATION_S - elapsed
        if int(elapsed * 2) != int((elapsed - 0.1) * 2):  # update every 0.5s
            print(f"  {remaining:.1f}s remaining... ({len(samples)} samples)", end="\r")
    
    print("\n")
    
    if len(samples) < 50:
        print(f"  ❌ FAILED: Only received {len(samples)} samples in {PREFLIGHT_DURATION_S}s.")
        print(f"  → ESP may not be sending data. Check connection and firmware.")
        return False
    
    # Compute per-axis ranges to check for movement
    ax_vals = [s[0] for s in samples]
    ay_vals = [s[1] for s in samples]
    az_vals = [s[2] for s in samples]
    
    ax_range = max(ax_vals) - min(ax_vals)
    ay_range = max(ay_vals) - min(ay_vals)
    az_range = max(az_vals) - min(az_vals)
    
    print(f"  Samples collected:  {len(samples)}")
    print(f"  ax range: [{min(ax_vals):.2f}, {max(ax_vals):.2f}] (span {ax_range:.2f}g)")
    print(f"  ay range: [{min(ay_vals):.2f}, {max(ay_vals):.2f}] (span {ay_range:.2f}g)")
    print(f"  az range: [{min(az_vals):.2f}, {max(az_vals):.2f}] (span {az_range:.2f}g)")
    
    max_range = max(ax_range, ay_range, az_range)
    
    if max_range < PREFLIGHT_MIN_VARIATION:
        print(f"\n  ❌ FAILED: IMU appears frozen (max variation {max_range:.3f}g).")
        print(f"  → Unplug ESP, wait 5 seconds, replug, and try again.")
        return False
    
    print(f"\n  ✓ IMU is healthy (max variation {max_range:.2f}g across axes)")
    return True


def main():
    session_name = f"{PUNCH_TYPE}_{DISTANCE_M}m_{HAND}"
    session_dir = DATA_DIR / SUBJECT_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    output_path = session_dir / f"{session_name}_imu.csv"
    
    # Find serial port
    port = find_serial_port()
    if port is None:
        print("No serial port found. Is the ESP/Feather plugged in?")
        return
    print(f"\nUsing serial port: {port}")
    
    # Open serial connection
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")
        return
    
    # Give ESP a moment to settle after serial connection (it might reset)
    time.sleep(2)
    ser.reset_input_buffer()
    
    # Pre-flight check
    if not preflight_check(ser):
        ser.close()
        print("\nAborting. Fix the IMU issue and run again.")
        return
    
    # Wait for user to confirm
    print("\n" + "=" * 60)
    print(f"READY TO RECORD: {session_name}")
    print("=" * 60)
    print(f"Subject:           {SUBJECT_ID}")
    print(f"Output file:       {output_path}")
    print("\nPress ENTER to start recording, or Ctrl+C to cancel.")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        ser.close()
        return
    
    # Clear any buffered data before starting fresh
    ser.reset_input_buffer()
    
    print(f"\nReading from serial at {BAUD_RATE} baud...")
    print("Press Q to stop recording.\n")
    
    sample_count = 0
    start_time = None
    window_start = None
    window_samples = 0
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp_ms",
            "ax", "ay", "az",
            "gx_dps", "gy_dps", "gz_dps",
            "mx_raw", "my_raw", "mz_raw",
        ])
        
        try:
            previous_values = None
            frozen_count = 0
            warned_frozen = False
            
            while True:
                if keyboard.is_pressed("q"):
                    print("\nStopping recording...")
                    break
                
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                
                parts = line.split(",")
                if len(parts) != 9:
                    continue
                
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    continue
                
                now = time.perf_counter()
                if start_time is None:
                    start_time = now
                    window_start = now
                    print("First sample received — recording started.")
                
                # Detect frozen sensor (consecutive identical readings)
                if previous_values is not None and values == previous_values:
                    frozen_count += 1
                    if frozen_count >= FROZEN_THRESHOLD and not warned_frozen:
                        print(f"\n  ⚠️  WARNING: IMU appears frozen "
                              f"({frozen_count} identical samples).")
                        print(f"  → Stop recording (Q), unplug+replug ESP, restart.")
                        warned_frozen = True
                else:
                    if warned_frozen and frozen_count > 0:
                        print(f"  ✓ IMU recovered (was frozen for {frozen_count} samples)")
                    frozen_count = 0
                    warned_frozen = False
                
                previous_values = values
                
                timestamp_ms = (now - start_time) * 1000.0
                writer.writerow([f"{timestamp_ms:.3f}"] + values)
                sample_count += 1
                window_samples += 1
                
                if now - window_start >= 2.0:
                    rate = window_samples / (now - window_start)
                    elapsed = now - start_time
                    health = "✓" if not warned_frozen else "⚠️  FROZEN"
                    print(f"  Samples: {sample_count} | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Instant rate: {rate:.1f} Hz | {health}")
                    window_start = now
                    window_samples = 0
        finally:
            ser.close()
    
    if start_time is not None:
        total_time = time.perf_counter() - start_time
        actual_rate = sample_count / total_time if total_time > 0 else 0
        print(f"\nRecording complete.")
        print(f"  Total samples:  {sample_count}")
        print(f"  Total duration: {total_time:.2f} seconds")
        print(f"  Actual rate:    {actual_rate:.1f} Hz")
        print(f"  Saved to:       {output_path}")
    else:
        print("\nNo data received.")
        print("Check that the ESP is running and outputting on the correct port.")


if __name__ == "__main__":
    main()