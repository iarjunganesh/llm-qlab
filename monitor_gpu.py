"""
monitor_gpu.py — Lightweight GPU stats logger using nvidia-smi.

Polls nvidia-smi at a configurable interval and logs:
  - GPU utilization (%)
  - VRAM used / total (MB)
  - Temperature (°C)
  - Power draw (W)

Output is saved to a CSV file (default: results/gpu_log.csv).
Run this in a separate terminal alongside benchmark.py.

Usage:
    python monitor_gpu.py
    python monitor_gpu.py --interval 2 --output results/my_gpu_log.csv
"""

import argparse
import csv
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# nvidia-smi query
# ---------------------------------------------------------------------------

QUERY_FIELDS = [
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]

CSV_FIELDS = ["timestamp", "gpu_util_pct", "vram_used_mb", "vram_total_mb", "temp_c", "power_w"]


def query_gpu() -> dict | None:
    """Query nvidia-smi and return a dict of stats, or None on failure."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        values = [v.strip() for v in output.strip().splitlines()[0].split(",")]
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "gpu_util_pct": float(values[0]),
            "vram_used_mb": float(values[1]),
            "vram_total_mb": float(values[2]),
            "temp_c": float(values[3]),
            "power_w": float(values[4]),
        }
    except Exception as exc:
        print(f"[warn] nvidia-smi query failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


def monitor(interval: float, output_path: Path) -> None:
    output_path.parent.mkdir(exist_ok=True)
    write_header = not output_path.exists()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"Monitoring GPU every {interval}s — saving to {output_path}")
    print("Press Ctrl+C to stop.\n")

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        while not _stop:
            stats = query_gpu()
            if stats:
                writer.writerow(stats)
                f.flush()
                print(
                    f"[{stats['timestamp']}] "
                    f"GPU: {stats['gpu_util_pct']:.0f}%  "
                    f"VRAM: {stats['vram_used_mb']:.0f}/{stats['vram_total_mb']:.0f} MB  "
                    f"Temp: {stats['temp_c']:.0f}°C  "
                    f"Power: {stats['power_w']:.1f} W"
                )
            time.sleep(interval)

    print(f"\nStopped. Log saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll nvidia-smi and log GPU stats to a CSV file."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="results/gpu_log.csv",
        help="Output CSV file path (default: results/gpu_log.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor(interval=args.interval, output_path=Path(args.output))


if __name__ == "__main__":
    main()
