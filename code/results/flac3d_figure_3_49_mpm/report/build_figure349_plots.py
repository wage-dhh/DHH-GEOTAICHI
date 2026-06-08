from pathlib import Path
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"F:\dhh\GeoTaichi-dhh")
RESULT_DIR = ROOT / "code/results/flac3d_figure_3_49_mpm"
REPORT_DIR = RESULT_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

G0 = 9.80665


def manual_accel_g(time):
    return math.sqrt(0.375 * math.exp(-2.2 * time) * time**8) * math.sin(6.0 * math.pi * time)


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=float)


motion = read_csv(RESULT_DIR / "base_motion.csv")
hist = read_csv(RESULT_DIR / "histories.csv")
equiv = read_csv(RESULT_DIR / "equivalent_1d_response.csv")

time = motion["time"]
manual_g = np.array([manual_accel_g(float(t)) for t in time])
piple_g = motion["base_acceleration_g"]
err = piple_g - manual_g

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=220)
ax.plot(time, manual_g, color="#111111", lw=1.25, label="FLAC3D manual formula")
ax.plot(time, piple_g, color="#B23A2E", lw=0.9, ls="--", label="PIPLE.py generated input")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Acceleration [g]")
ax.set_title("Figure 3.49 Input acceleration at bottom of model")
ax.grid(True, color="#D7DDE5", lw=0.45)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(REPORT_DIR / "fig1_manual_vs_piple_figure349_input.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 3.3), dpi=220)
ax.plot(time, err, color="#1F6F8B", lw=1.0)
ax.axhline(0, color="#555555", lw=0.7)
ax.set_xlabel("Time [s]")
ax.set_ylabel("PIPLE - manual [g]")
ax.set_title(f"Input acceleration reproduction error, max |error| = {np.max(np.abs(err)):.3e} g")
ax.grid(True, color="#D7DDE5", lw=0.45)
fig.tight_layout()
fig.savefig(REPORT_DIR / "fig2_figure349_input_error.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), dpi=220, sharex=True)
axes[0].plot(time, motion["integrated_base_velocity"], color="#1F6F8B", lw=1.0)
axes[0].set_ylabel("Velocity [m/s]")
axes[0].grid(True, color="#D7DDE5", lw=0.45)
axes[0].set_title("Boundary histories derived from Figure 3.49 input")
axes[1].plot(time, motion["integrated_base_displacement"], color="#2D7D46", lw=1.0)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Displacement [m]")
axes[1].grid(True, color="#D7DDE5", lw=0.45)
fig.tight_layout()
fig.savefig(REPORT_DIR / "fig3_integrated_boundary_histories.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=220)
ax.plot(hist["time"], hist["input_base_ax_g"], color="#111111", lw=0.9, label="input acceleration")
ax.plot(hist["time"], hist["base_vx"], color="#B23A2E", lw=0.8, label="bottom layer vx")
ax.plot(hist["time"], hist["top_vx"], color="#2D7D46", lw=0.8, label="top vx diagnostic")
ax.set_xlabel("Time [s]")
ax.set_ylabel("g or m/s")
ax.set_title("GeoTaichi run diagnostic histories")
ax.grid(True, color="#D7DDE5", lw=0.45)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(REPORT_DIR / "fig4_geotaichi_run_diagnostics.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), dpi=220, sharex=True)
axes[0].plot(equiv["time"], equiv["input_base_ax_g"], color="#111111", lw=0.9, label="base input")
axes[0].plot(equiv["time"], equiv["top_ax_g"], color="#B23A2E", lw=0.9, label="equivalent 1D top response")
axes[0].set_ylabel("Acceleration [g]")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, color="#D7DDE5", lw=0.45)
axes[0].set_title("Approximate full dynamic response: acceleration histories")
axes[1].plot(equiv["time"], equiv["input_base_vx"], color="#111111", lw=0.8, label="base velocity")
axes[1].plot(equiv["time"], equiv["top_vx"], color="#1F6F8B", lw=0.9, label="top velocity")
axes[1].set_ylabel("Velocity [m/s]")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].grid(True, color="#D7DDE5", lw=0.45)
axes[2].plot(equiv["time"], equiv["stress_point_shear_stress"], color="#2D7D46", lw=0.9)
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Shear stress [Pa]")
axes[2].grid(True, color="#D7DDE5", lw=0.45)
fig.tight_layout()
fig.savefig(REPORT_DIR / "fig5_equivalent_1d_dynamic_histories.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=220)
ax.plot(equiv["stress_point_strain"], equiv["stress_point_shear_stress"], color="#B23A2E", lw=0.9)
ax.axhline(0, color="#666666", lw=0.6)
ax.axvline(0, color="#666666", lw=0.6)
ax.set_xlabel("Shear strain")
ax.set_ylabel("Shear stress [Pa]")
ax.set_title("Equivalent 1D stress-strain response near z = 37 m")
ax.grid(True, color="#D7DDE5", lw=0.45)
fig.tight_layout()
fig.savefig(REPORT_DIR / "fig6_equivalent_1d_stress_strain.png", bbox_inches="tight")
plt.close(fig)

summary = [
    "figure = Figure 3.49 Input acceleration at bottom of model",
    f"points = {len(time)}",
    f"manual_min_g = {manual_g.min():.12g}",
    f"manual_max_g = {manual_g.max():.12g}",
    f"piple_min_g = {piple_g.min():.12g}",
    f"piple_max_g = {piple_g.max():.12g}",
    f"max_abs_error_g = {np.max(np.abs(err)):.12g}",
    f"rms_error_g = {math.sqrt(float(np.mean(err * err))):.12g}",
    f"equivalent_top_ax_g_min = {equiv['top_ax_g'].min():.12g}",
    f"equivalent_top_ax_g_max = {equiv['top_ax_g'].max():.12g}",
    f"equivalent_top_vx_min = {equiv['top_vx'].min():.12g}",
    f"equivalent_top_vx_max = {equiv['top_vx'].max():.12g}",
    f"equivalent_stress_min = {equiv['stress_point_shear_stress'].min():.12g}",
    f"equivalent_stress_max = {equiv['stress_point_shear_stress'].max():.12g}",
    f"equivalent_strain_min = {equiv['stress_point_strain'].min():.12g}",
    f"equivalent_strain_max = {equiv['stress_point_strain'].max():.12g}",
]
(REPORT_DIR / "figure349_reproduction_metrics.txt").write_text("\n".join(summary), encoding="utf-8")
