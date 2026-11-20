"""
impact.py -- sample laptop impact while a training run is going.

Usage:
  .venv/bin/python impact.py <pidfile> <out.csv> [interval_s]

Samples every interval: GPU (nvidia-smi), system CPU %, train-process CPU %,
RAM available, CPU package temp (coretemp). Stops when the train pid is gone.
"""

import csv
import os
import subprocess
import sys
import time


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        u, m, t, p = [float(x) for x in out.split(", ")]
        return u, m, t, p
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def _proc_stat_total():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)
    return total, idle


def _proc_ticks(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().split()
    return int(parts[13]) + int(parts[14])


def cpu_temp_c():
    for d in sorted(os.listdir("/sys/class/hwmon")):
        p = f"/sys/class/hwmon/{d}"
        try:
            if open(p + "/name").read().strip() == "coretemp":
                return int(open(p + "/temp1_input").read()) / 1000.0
        except Exception:
            pass
    return -1.0


def mem_avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


def main():
    pidfile, out = sys.argv[1], sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    pid = int(open(pidfile).read().strip())

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gpu_util", "gpu_mem_mib", "gpu_temp_c", "gpu_power_w",
                    "cpu_pct", "train_cpu_pct", "ram_avail_mb", "cpu_temp_c"])
        prev_total, prev_idle = _proc_stat_total()
        prev_ticks = _proc_ticks(pid)
        t0 = time.time()
        last = t0
        while os.path.exists(f"/proc/{pid}"):
            time.sleep(interval)
            now = time.time()
            u, m, gt, gp = gpu()
            total, idle = _proc_stat_total()
            dt = now - last
            cpu = 100.0 * ((total - prev_total) - (idle - prev_idle)) / max(1, total - prev_total)
            try:
                ticks = _proc_ticks(pid)
                tcpu = min(100.0 * (ticks - prev_ticks) / (100.0 * dt), 400.0)
            except Exception:
                tcpu = -1.0
            w.writerow([round(now - t0, 1), round(u, 1), m, gt, round(gp, 1),
                        round(cpu, 1), round(tcpu, 1), mem_avail_mb(), cpu_temp_c()])
            f.flush()
            prev_total, prev_idle = total, idle
            prev_ticks = ticks if tcpu >= 0 else prev_ticks
            last = now


if __name__ == "__main__":
    main()
