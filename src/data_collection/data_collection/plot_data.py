#!/usr/bin/env python3
import glob
import math
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

files = glob.glob("crazyflie_hover_*.csv")
if len(sys.argv) == 1 and not files:
    raise FileNotFoundError("No crazyflie_hover_*.csv file was found")
csv_path = sys.argv[1] if len(sys.argv) > 1 else max(files, key=os.path.getmtime)
data = pd.read_csv(csv_path)
numeric = data.apply(pd.to_numeric, errors="coerce")
time_s = numeric["time_s"]
columns = [name for name in numeric.columns if name != "time_s" and numeric[name].notna().any()]
per_page = 18
for start in range(0, len(columns), per_page):
    page = columns[start:start + per_page]
    rows = math.ceil(len(page) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 3.2 * rows), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, name in zip(axes, page):
        axis.plot(time_s, numeric[name], linewidth=1.0)
        axis.set_title(name)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)
    for axis in axes[len(page):]:
        axis.remove()
    fig.supxlabel("Time [s]")
    fig.suptitle(f"Crazyflie Hover Data - {os.path.basename(csv_path)}", fontweight="bold", y=0.998)
    fig.tight_layout()
    output = f"{os.path.splitext(csv_path)[0]}_plots_{start // per_page + 1:02d}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")
plt.show()