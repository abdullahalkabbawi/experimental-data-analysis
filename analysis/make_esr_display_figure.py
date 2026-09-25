"""Schematic of the two ESR oscilloscope displays.

The NMR half of the report already has a figure of its oscilloscope displays
(fig_nmr_displays_wiggles.pdf, made by generate_esr_nmr.py). This script draws the
matching pair of panels for the ESR half:

    (a) the two-channel time display, with the 50 Hz modulated field on CH I and the
        two absorption signals per modulation cycle on CH II, equally spaced when the
        steady current is set so that B_0 sits on the resonance field;
    (b) the XY (Lissajous) display, with the PHASE control set so that the two
        half-wave traversals coincide, and the uncompensated loop shown for comparison.

This is a SCHEMATIC of the instrument display. It is drawn from the representative
parameters of the 50 MHz line-width measurement, it is not a captured trace and it is
not a record of measured results.

Inputs : reports/data/esr_linewidth.csv
Output : reports/figures/fig_esr_displays.pdf

Run with:
    python analysis/make_esr_display_figure.py
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(REPORTS, "data")
FIG_DIR = os.path.join(REPORTS, "figures")

# Okabe-Ito colourblind-safe palette, same as generate_esr_nmr.py
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_BLACK = "#000000"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "text.usetex": False,
    }
)

# Coil constant from the Helmholtz calibration fit reported in Section 4.1.
S_COIL_mT_per_A = 4.2073
F_MOD_Hz = 50.0


def read_linewidth():
    """Pull B_0, the RMS modulation current and the FWHM out of the line-width file."""
    values = {}
    with open(os.path.join(DATA_DIR, "esr_linewidth.csv"), newline="") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if row and row[0] != "quantity":
                values[row[0]] = float(row[1])
    return (
        values["resonance field B0"],
        values["modulation current I_mod (RMS)"],
        values["line width delta_B0 (FWHM)"],
    )


B0_mT, I_MOD_RMS_A, FWHM_mT = read_linewidth()

# Peak modulation field: the peak-to-peak swing is s * 2 * sqrt(2) * I_mod (script
# equation 7), so the amplitude is half of that.
B_MOD_mT = S_COIL_mT_per_A * np.sqrt(2.0) * I_MOD_RMS_A

# Detuning used only for the grey "not yet tuned" trace in panel (a).
DETUNE_mT = 0.25
PHASE_ERR_rad = 0.08


def absorption(field_mT):
    """Lorentzian absorption chi'' against field, unit height, FWHM as measured."""
    return 1.0 / (1.0 + ((field_mT - B0_mT) / (0.5 * FWHM_mT)) ** 2)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.4, 2.6),
                               gridspec_kw={"wspace": 0.30})

# ---- (a) two-channel display -----------------------------------------------------
period_ms = 1.0e3 / F_MOD_Hz
t_ms = np.linspace(0.0, 2.0 * period_ms, 6000)
sweep = np.sin(2.0 * np.pi * F_MOD_Hz * t_ms * 1.0e-3)

field_tuned = B0_mT + B_MOD_mT * sweep
field_detuned = B0_mT - DETUNE_mT + B_MOD_mT * sweep

axa.plot(t_ms, sweep * 0.45 - 0.75, "-", color=C_ORANGE, lw=0.9,
         label="B SIGNAL, CH I")
axa.plot(t_ms, absorption(field_detuned), "--", color="0.62", lw=0.8,
         label="CH II, $B_0$ not yet set")
axa.plot(t_ms, absorption(field_tuned), "-", color=C_BLUE, lw=0.9,
         label="ESR SIGNAL, CH II")

# Mark the equal spacing of the tuned signals: one half modulation period apart.
# On resonance the field crosses B_res where the modulation is zero, so the absorptions
# sit at the zero crossings of the sweep, not at its turning points.
peaks_ms = [0.5 * k * period_ms for k in range(5)]
axa.plot(peaks_ms, [1.0] * len(peaks_ms), "o", ms=3, color=C_BLACK, zorder=5,
         clip_on=False)
axa.annotate("", xy=(peaks_ms[2], 1.16), xytext=(peaks_ms[3], 1.16),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color=C_BLACK))
axa.text(0.5 * (peaks_ms[2] + peaks_ms[3]), 1.20, "10 ms", ha="center", va="bottom",
         fontsize=6.5)

axa.set_xlabel("Time (ms)")
axa.set_ylabel("Signal (arb.)")
axa.set_title("(a) two-channel display", fontsize=8)
axa.set_xlim(0.0, 2.0 * period_ms)
axa.set_ylim(-2.05, 1.5)
axa.legend(loc="lower center", frameon=False, fontsize=6.2, handlelength=1.4,
           borderpad=0.1, labelspacing=0.25)

# ---- (b) XY display ---------------------------------------------------------------
# The X deflection is the field reference. A residual phase lag between the true field
# at the sample and that reference displaces the two traversals horizontally by
# 2 B_mod sin(phi), which opens the trace into a loop; the PHASE control removes it.
x_axis = B_MOD_mT * sweep
field_lagged = B0_mT + B_MOD_mT * np.sin(2.0 * np.pi * F_MOD_Hz * t_ms * 1.0e-3
                                         - PHASE_ERR_rad)

axb.plot(x_axis, absorption(field_lagged), "--", color="0.62", lw=0.8,
         label="PHASE not set")
axb.plot(x_axis, absorption(field_tuned), "-", color=C_BLUE, lw=0.9,
         label="PHASE set, traces coincide")

# Half-height width of the coincident line, the quantity read off this display.
half = 0.5
edge = 0.5 * FWHM_mT
axb.plot([-edge, edge], [half, half], "-", color=C_BLACK, lw=0.7)
axb.plot([-edge, edge], [half, half], "|", color=C_BLACK, ms=4)
axb.text(0.0, half - 0.16, r"$\delta B_0$", ha="center", va="top", fontsize=7)

axb.set_xlabel("B SIGNAL (mT about $B_0$), field axis")
axb.set_ylabel("ESR signal (arb.)")
axb.set_title("(b) XY display", fontsize=8)
axb.set_xlim(-B_MOD_mT * 1.05, B_MOD_mT * 1.05)
axb.set_ylim(-0.12, 1.30)
axb.legend(loc="upper right", frameon=False, fontsize=6.2, handlelength=1.4,
           borderpad=0.1, labelspacing=0.25)

os.makedirs(FIG_DIR, exist_ok=True)
# Every other figure in the report carries an in-image flag, put there by save() in
# generate_esr_nmr.py, so that a figure lifted out of the PDF on its own still says what
# it is. This panel pair is a schematic rather than data, so it says that instead.
fig.text(0.99, -0.34 / fig.get_figheight(), "Schematic, not measured data",
         ha="right", va="bottom", fontsize=7, color="0.45")
out = os.path.join(FIG_DIR, "fig_esr_displays.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
plt.close(fig)

print("B_0            = %.3f mT" % B0_mT)
print("I_mod (RMS)    = %.3f A" % I_MOD_RMS_A)
print("B_mod (peak)   = %.3f mT  (peak to peak %.3f mT)" % (B_MOD_mT, 2 * B_MOD_mT))
print("FWHM           = %.3f mT" % FWHM_mT)
print("wrote %s" % out)
