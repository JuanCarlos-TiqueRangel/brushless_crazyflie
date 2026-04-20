#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

# Folder where the CSV files are located
LOG_DIR = Path(__file__).resolve().parent

# List of CSV files and display names
CONTROLLERS = [
    ("figure8_log_PID.csv", "PID"),
    ("figure8_log_Mellinger.csv", "Mellinger"),
    ("figure8_log_INDI.csv", "INDI"),
    ("figure8_log_Brescianini.csv", "Brescianini"),
    ("figure8_log_Lee.csv", "Lee"),
]

# Font and style settings
FIGSIZE = (13, 18)
TITLE_FONTSIZE = 16
SUBTITLE_FONTSIZE = 10
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 9
LINEWIDTH_REF = 2.2
LINEWIDTH_MEAS = 2.0
LINEWIDTH_ERR = 1.8

# Save figure
SAVE_FIGURE = True
OUTPUT_NAME = "all_controllers_figure8_and_errors.png"
DPI = 300


# ============================================================
# FUNCTIONS
# ============================================================

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "t_wall",
        "x_ref", "y_ref", "z_ref",
        "x_meas", "y_meas", "z_meas",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df["t"] = df["t_wall"] - df["t_wall"].iloc[0]
    df["ex"] = df["x_ref"] - df["x_meas"]
    df["ey"] = df["y_ref"] - df["y_meas"]
    df["ez"] = df["z_ref"] - df["z_meas"]

    return df


def compute_rmse(df: pd.DataFrame):
    rmse_x = (df["ex"] ** 2).mean() ** 0.5
    rmse_y = (df["ey"] ** 2).mean() ** 0.5
    rmse_z = (df["ez"] ** 2).mean() ** 0.5
    return rmse_x, rmse_y, rmse_z


# ============================================================
# MAIN
# ============================================================

def main():
    n = len(CONTROLLERS)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=FIGSIZE)

    # If there is only one controller, axes is not 2D by default
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Figure-8 Trajectory Tracking Comparison Across Controllers",
        fontsize=TITLE_FONTSIZE
    )

    for i, (csv_name, controller_name) in enumerate(CONTROLLERS):
        csv_path = LOG_DIR / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = load_csv(csv_path)
        rmse_x, rmse_y, rmse_z = compute_rmse(df)

        ax_xy = axes[i][0]
        ax_err = axes[i][1]

        # ----------------------------------------------------
        # Left column: XY trajectory
        # ----------------------------------------------------
        ax_xy.plot(
            df["x_ref"], df["y_ref"],
            label="Reference",
            linewidth=LINEWIDTH_REF
        )
        ax_xy.plot(
            df["x_meas"], df["y_meas"],
            label="Measured",
            linewidth=LINEWIDTH_MEAS
        )

        ax_xy.set_title(
            f"{controller_name} - Figure 8",
            fontsize=SUBTITLE_FONTSIZE
        )
        # ax_xy.set_xlabel("x [m]", fontsize=LABEL_FONTSIZE)
        ax_xy.set_ylabel("y [m]", fontsize=LABEL_FONTSIZE)
        ax_xy.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax_xy.grid(True)
        ax_xy.axis("equal")
        ax_xy.legend(fontsize=LEGEND_FONTSIZE)

        # ----------------------------------------------------
        # Right column: tracking errors
        # ----------------------------------------------------
        ax_err.plot(
            df["t"], df["ex"],
            label=r"$e_x$",
            linewidth=LINEWIDTH_ERR
        )
        ax_err.plot(
            df["t"], df["ey"],
            label=r"$e_y$",
            linewidth=LINEWIDTH_ERR
        )
        ax_err.plot(
            df["t"], df["ez"],
            label=r"$e_z$",
            linewidth=LINEWIDTH_ERR
        )

        ax_err.set_title(
            f"{controller_name} - Errors (RMSE: x={rmse_x:.3f}, y={rmse_y:.3f}, z={rmse_z:.3f})",
            fontsize=SUBTITLE_FONTSIZE
        )
        # ax_err.set_xlabel("time [s]", fontsize=LABEL_FONTSIZE)
        ax_err.set_ylabel("error [m]", fontsize=LABEL_FONTSIZE)
        ax_err.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax_err.grid(True)
        ax_err.legend(fontsize=LEGEND_FONTSIZE)

    ax_xy.set_xlabel("x [m]", fontsize=LABEL_FONTSIZE)
    ax_err.set_xlabel("time [s]", fontsize=LABEL_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if SAVE_FIGURE:
        output_path = LOG_DIR / OUTPUT_NAME
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()