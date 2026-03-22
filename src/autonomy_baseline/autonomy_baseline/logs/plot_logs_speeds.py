#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

LOG_DIR = Path(__file__).resolve().parent

CONTROLLERS = [
    ("figure8_log_PID.csv", "PID"),
    ("figure8_log_Mellinger.csv", "Mellinger"),
    ("figure8_log_INDI.csv", "INDI"),
    ("figure8_log_Brescianini.csv", "Brescianini"),
    ("figure8_log_Lee.csv", "Lee"),
]

# Font and style settings
FIGSIZE = (14, 18)
TITLE_FONTSIZE = 16
SUBTITLE_FONTSIZE = 13
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 9
LINEWIDTH = 1.8
BAR_HEIGHT = 0.5

SAVE_FIGURE = True
OUTPUT_NAME = "all_controllers_speeds_and_total_time.pdf"
DPI = 300


# ============================================================
# FUNCTIONS
# ============================================================

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "t_wall",
        "vx_meas", "vy_meas", "vz_meas",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df["t"] = df["t_wall"] - df["t_wall"].iloc[0]
    df["speed_norm"] = (df["vx_meas"]**2 + df["vy_meas"]**2 + df["vz_meas"]**2) ** 0.5
    return df


def total_time(df: pd.DataFrame) -> float:
    return float(df["t_wall"].iloc[-1] - df["t_wall"].iloc[0])


# ============================================================
# MAIN
# ============================================================

def main():
    n = len(CONTROLLERS)

    # First pass: load all data and compute max total time for common x-axis
    loaded = []
    max_total_t = 0.0

    for csv_name, controller_name in CONTROLLERS:
        csv_path = LOG_DIR / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = load_csv(csv_path)
        T = total_time(df)
        max_total_t = max(max_total_t, T)
        loaded.append((df, controller_name, T))

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=FIGSIZE)

    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Measured Speeds and Total Experiment Time Across Controllers",
        fontsize=TITLE_FONTSIZE
    )

    for i, (df, controller_name, T) in enumerate(loaded):
        ax_speed = axes[i][0]
        ax_time = axes[i][1]

        # ----------------------------------------------------
        # Left column: measured speeds
        # ----------------------------------------------------
        ax_speed.plot(df["t"], df["vx_meas"], label=r"$v_x$", linewidth=LINEWIDTH)
        ax_speed.plot(df["t"], df["vy_meas"], label=r"$v_y$", linewidth=LINEWIDTH)
        ax_speed.plot(df["t"], df["vz_meas"], label=r"$v_z$", linewidth=LINEWIDTH)
        ax_speed.plot(df["t"], df["speed_norm"], label=r"$\|v\|$", linewidth=LINEWIDTH)

        ax_speed.set_title(f"{controller_name} - Speeds", fontsize=SUBTITLE_FONTSIZE)
        ax_speed.set_xlabel("time [s]", fontsize=LABEL_FONTSIZE)
        ax_speed.set_ylabel("velocity [m/s]", fontsize=LABEL_FONTSIZE)
        ax_speed.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax_speed.grid(True)
        ax_speed.legend(fontsize=LEGEND_FONTSIZE)

        # ----------------------------------------------------
        # Right column: total experiment time
        # ----------------------------------------------------
        ax_time.barh(
            [0],
            [T],
            height=BAR_HEIGHT,
            label="total time"
        )

        ax_time.set_title(f"{controller_name} - Total Time", fontsize=SUBTITLE_FONTSIZE)
        ax_time.set_xlabel("total time [s]", fontsize=LABEL_FONTSIZE)
        ax_time.set_yticks([])
        ax_time.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax_time.grid(True, axis="x")
        ax_time.set_xlim(0.0, 1.10 * max_total_t)

        # Put the numeric value on the bar
        ax_time.text(
            T,
            0,
            f"  {T:.2f} s",
            va="center",
            ha="left",
            fontsize=LABEL_FONTSIZE
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if SAVE_FIGURE:
        output_path = LOG_DIR / OUTPUT_NAME
        if output_path.suffix.lower() == ".pdf":
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()