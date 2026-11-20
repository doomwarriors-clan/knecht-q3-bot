"""
summarize_impact.py -- float-correct summary of an impact CSV.

Usage: .venv/bin/python summarize_impact.py impact_run1.csv
"""

import csv
import sys


def main():
    path = sys.argv[1]
    rows = list(csv.DictReader(open(path)))
    n = len(rows)

    def col(k):
        return [float(r[k]) for r in rows]

    def s(k):
        v = col(k)
        return "mean %.1f | min %.1f | max %.1f" % (
            sum(v) / len(v), min(v), max(v))

    print("file: %s" % path)
    print("samples: %d | duration: %s s" % (n, rows[-1]["t_s"]))
    print("gpu_util %:   " + s("gpu_util"))
    print("gpu_mem MiB:  " + s("gpu_mem_mib"))
    print("gpu_temp C:   " + s("gpu_temp_c"))
    print("gpu_power W:  " + s("gpu_power_w"))
    print("sys_cpu %:    " + s("cpu_pct"))
    print("train_cpu %:  " + s("train_cpu_pct"))
    print("ram_avail MiB:" + s("ram_avail_mb"))
    print("cpu_temp C:   " + s("cpu_temp_c"))


if __name__ == "__main__":
    main()
