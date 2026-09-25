"""
generate_hall.py
================

Representative-data generator and analysis pipeline for the report
"The Hall Effect in a Two-Dimensional Electron Gas"
(University of Glasgow, PHYS4009 Physics Group Project, script "Hall Effect 2023 v1").

IMPORTANT, PLEASE READ
----------------------
The data produced by this script are REPRESENTATIVE. They are simulated values built
forward from physically accepted parameters for a GaAs/AlGaAs modulation-doped
heterostructure, with realistic instrument scatter and realistic systematic offsets
added on top. They are NOT a record of measured results, and every table, figure and
data file produced here is labelled as representative.

Reproducibility
---------------
The random number generator is seeded with the fixed integer SEED = 4009 (chosen after
the course code). Re-running this script on any machine reproduces every number, every
CSV and every figure exactly. Nothing in this script depends on the current date or
time.

Method
------
1. GROUND TRUTH is chosen first: sheet density, mobility, the true magnet response,
   contact misalignment, meter noise floors. These are the "true" values of the
   simulated apparatus and they are never used in the analysis.
2. The IDEAL observable is computed from that ground truth (Ohm's law, the 2D Hall
   relation, the true magnet response curve).
3. REALISTIC SCATTER AND SYSTEMATICS are added: Gaussian meter noise with a floor plus
   a percentage of reading, a slow thermal-EMF drift across each sweep, an iron-core
   hysteresis branch offset, and a contact-misalignment voltage that leaks a fraction
   of the longitudinal voltage into the Hall contact pair.
4. The generated data are then ANALYSED exactly as a student would analyse real data.
   In particular the magnet field used in the analysis is the field read off the
   CALIBRATION POLYNOMIAL, not the true field used to generate the Hall voltages, so
   the calibration error propagates into the answer just as it would in the laboratory.
   Fits are weighted least squares, uncertainties come from the covariance matrix, and
   they are propagated into the final physical quantities.
5. The fitted numbers are therefore NOT the ground truth and differ from it. Those
   differences are reported honestly in results_hall.md.

Outputs
-------
CSV data   -> reports/data/
Figures    -> reports/figures/   (vector PDF)
Results    -> reports/data/results_hall.md
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.optimize as op

# ----------------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------------

SEED = 4009
rng = np.random.default_rng(SEED)

BASE = r"C:\Users\Abdullah\.gemini\antigravity\scratch\jobs&CVs\reports"
DATA_DIR = os.path.join(BASE, "data")
FIG_DIR = os.path.join(BASE, "figures")
for d in (DATA_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# Okabe-Ito colourblind-safe palette
CB = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "verm": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}
SERIES = [CB["blue"], CB["orange"], CB["green"], CB["verm"], CB["purple"]]
MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 7.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "text.usetex": False,
    }
)

# ----------------------------------------------------------------------------------
# 1. Physical constants (CODATA)
# ----------------------------------------------------------------------------------

E_CHARGE = 1.602176634e-19      # C
HBAR = 1.054571817e-34          # J s
M0 = 9.1093837015e-31           # kg
KB = 1.380649e-23               # J / K
M_STAR = 0.067 * M0             # GaAs Gamma-valley effective mass, kg

# ----------------------------------------------------------------------------------
# 2. GROUND TRUTH of the simulated apparatus and sample
# ----------------------------------------------------------------------------------

# --- sample geometry, from the departmental script -------------------------------
L_ACTIVE = 300e-6               # m, voltage-probe separation, contacts 1 to 4
W_ACTIVE = 100e-6               # m, Hall bar width
N_SQUARES = L_ACTIVE / W_ACTIVE  # = 3.0 squares

# --- 2DEG transport parameters ----------------------------------------------------
TRUE = {
    "300K": {"n_sq": 5.00e15, "mu": 0.800},   # m^-2, m^2 V^-1 s^-1
    "77K": {"n_sq": 4.40e15, "mu": 12.00},
}
for T, p in TRUE.items():
    p["R_H2D"] = 1.0 / (p["n_sq"] * E_CHARGE)                 # ohm / T
    p["R_sq"] = 1.0 / (p["n_sq"] * E_CHARGE * p["mu"])        # ohm / square
    p["R_L"] = N_SQUARES * p["R_sq"]                          # ohm, contacts 1 to 4

# --- drive circuit ----------------------------------------------------------------
R_SERIES = 300.0e3              # two 150 kohm protection resistors in series
I_CAP = 100e-6                  # A, software current limit on the Hall bar

# --- contact misalignment ---------------------------------------------------------
# Hall arms 1 and 2 are displaced by DELTA_MISALIGN along the current direction, so
# the pair straddles a short length of bar and picks up a fraction delta/w of the
# longitudinal voltage. This term is proportional to I and independent of B.
DELTA_MISALIGN = 0.80e-6        # m

# --- static instrumental / thermal offset -----------------------------------------
V_OFFSET_STATIC = {"300K": 0.25e-3, "77K": 0.12e-3}    # V, Hall channel
V_OFFSET_STATIC_L = {"300K": 0.15e-3, "77K": 0.04e-3}  # V, longitudinal channel

# --- thermal EMF drift across one sweep (peak to peak), worse at 77 K -------------
DRIFT_PP = {"300K": 0.030e-3, "77K": 0.060e-3}   # V

# --- electromagnet ----------------------------------------------------------------
R_COIL = 10.2                   # ohm
MAG_POLY_TRUE = (0.2050, -0.0350, 0.0040)   # B = a1*I + a2*I^2 + a3*I^3, tesla
B_REMANENT_UP = 0.0015          # T, field at zero coil current on the rising branch
HYST_GAP0 = 0.0080              # T, branch separation extrapolated to zero current
HYST_EXP = 2.5                  # branches converge at the turning point
MAG_V_MAX = 24.0                # V
I_COIL_MAX = MAG_V_MAX / R_COIL  # A

# --- meter noise floors -----------------------------------------------------------
SIG_VH_FLOOR = 0.050e-3         # V
SIG_VH_FRAC = 0.0015
SIG_VL_FLOOR = {"300K": 0.200e-3, "77K": 0.050e-3}
SIG_VL_FRAC = {"300K": 0.0010, "77K": 0.0015}
SIG_I_FLOOR = 0.10e-6           # A
SIG_I_FRAC = 0.0010
SIG_BPROBE = 0.0025             # T, Hall probe magnetometer repeatability
SIG_ICOIL = 0.0050              # A
SIG_B_BRANCH = 0.0020           # T, branch reproducibility between runs

# --- systematic uncertainties quoted at the end -----------------------------------
SYST_B_FRAC = 0.030             # 3 per cent on B (probe accuracy, positioning)
SYST_B_FLOOR = 0.0050           # T
SYST_GEOM_FRAC = 0.010          # 1 per cent on l/w (etch bias, edge depletion)

# --- 77 K thermometry -------------------------------------------------------------
T_TRUE_77 = 78.6                # K, sample mount sits above the liquid
SIG_TC = 0.8                    # K, thermocouple reading scatter


# ----------------------------------------------------------------------------------
# 3. Small helpers
# ----------------------------------------------------------------------------------

def b_true_rising(i_coil):
    """True gap field on the rising branch, tesla, for coil current in amps."""
    a1, a2, a3 = MAG_POLY_TRUE
    return a1 * i_coil + a2 * i_coil**2 + a3 * i_coil**3 + B_REMANENT_UP


def b_true_falling(i_coil):
    """True gap field on the falling branch. Sits above the rising branch."""
    frac = np.clip(i_coil / I_COIL_MAX, 0.0, 1.0)
    gap = HYST_GAP0 * (1.0 - frac**HYST_EXP)
    return b_true_rising(i_coil) + gap


def sigma_vh(v):
    return np.sqrt(SIG_VH_FLOOR**2 + (SIG_VH_FRAC * np.abs(v)) ** 2)


def sigma_vl(v, T):
    return np.sqrt(SIG_VL_FLOOR[T] ** 2 + (SIG_VL_FRAC[T] * np.abs(v)) ** 2)


def sigma_i(i):
    return np.sqrt(SIG_I_FLOOR**2 + (SIG_I_FRAC * np.abs(i)) ** 2)


def wls(X, y, sig):
    """Weighted linear least squares. Returns beta, covariance, chi2, dof."""
    w = 1.0 / sig
    Xw = X * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.inv(Xw.T @ Xw)
    resid = y - X @ beta
    chi2 = float(np.sum((resid / sig) ** 2))
    dof = len(y) - X.shape[1]
    return beta, cov, chi2, dof


def polyfit_w(x, y, deg, sig):
    """Weighted polynomial fit. Coefficients highest power first, as np.polyval."""
    X = np.vander(np.asarray(x, dtype=float), deg + 1)
    return wls(X, np.asarray(y, dtype=float), np.asarray(sig, dtype=float))


def drift_profile(n, pp, sign):
    """Linear thermal-EMF drift across a sweep, peak to peak amplitude pp."""
    return sign * pp * (np.linspace(0.0, 1.0, n) - 0.5)


def pct_diff(a, b):
    """Percentage difference of a relative to reference b."""
    return 100.0 * (a - b) / b


def quad(*terms):
    return float(np.sqrt(np.sum(np.array(terms, dtype=float) ** 2)))


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


# ==================================================================================
# PART (a) -- ELECTROMAGNET CALIBRATION
# ==================================================================================

print("=" * 78)
print("REPRESENTATIVE DATA GENERATOR: Hall effect in a 2DEG")
print("Seed =", SEED, " (fixed, fully reproducible)")
print("=" * 78)

mag_supply = np.arange(0.0, MAG_V_MAX + 1e-9, 1.0)      # V, 25 settings
n_mag = mag_supply.size
i_nom = mag_supply / R_COIL

# Rising sweep, then falling sweep back down through the same supply settings.
i_coil_up = np.maximum(i_nom + rng.normal(0.0, SIG_ICOIL, n_mag), 0.0)
i_coil_dn = np.maximum(i_nom + rng.normal(0.0, SIG_ICOIL, n_mag), 0.0)

b_meas_up = b_true_rising(i_nom) + rng.normal(0.0, SIG_BPROBE, n_mag)
b_meas_dn = b_true_falling(i_nom) + rng.normal(0.0, SIG_BPROBE, n_mag)

# Effective y-uncertainty on a calibration point: probe scatter, plus the coil-current
# reading uncertainty folded in through the local slope, plus branch reproducibility.
dbdi_nom = np.polyval(
    np.polyder(np.array([MAG_POLY_TRUE[2], MAG_POLY_TRUE[1], MAG_POLY_TRUE[0], 0.0])),
    i_nom,
)
sig_b_up = np.sqrt(SIG_BPROBE**2 + (dbdi_nom * SIG_ICOIL) ** 2 + SIG_B_BRANCH**2)
sig_b_dn = sig_b_up.copy()

# --- fit the rising branch with a cubic (order 3 > 1, as the script requires) -----
mag_c3, mag_cov3, chi2_3, dof_3 = polyfit_w(i_coil_up, b_meas_up, 3, sig_b_up)
mag_err3 = np.sqrt(np.diag(mag_cov3))
res3 = b_meas_up - np.polyval(mag_c3, i_coil_up)

# --- quadratic, kept only to show that it is not good enough ---------------------
mag_c2, mag_cov2, chi2_2, dof_2 = polyfit_w(i_coil_up, b_meas_up, 2, sig_b_up)
res2 = b_meas_up - np.polyval(mag_c2, i_coil_up)

# --- linear, kept to show how badly a straight line fails ------------------------
mag_c1, mag_cov1, chi2_1, dof_1 = polyfit_w(i_coil_up, b_meas_up, 1, sig_b_up)
res1 = b_meas_up - np.polyval(mag_c1, i_coil_up)

# --- falling branch, cubic, used only to quantify the hysteresis smoothly --------
mag_c3d, _, _, _ = polyfit_w(i_coil_dn, b_meas_dn, 3, sig_b_dn)

igrid = np.linspace(0.0, I_COIL_MAX, 400)
gap_fit = np.polyval(mag_c3d, igrid) - np.polyval(mag_c3, igrid)
gap_points = b_meas_dn - b_meas_up
gap_mean = float(np.mean(gap_points))
gap_at_zero = float(gap_fit[0])
gap_max_fit = float(np.max(gap_fit))
remanence_up_fit = float(np.polyval(mag_c3, 0.0))
remanence_dn_fit = float(np.polyval(mag_c3d, 0.0))


def b_from_calibration(i_coil):
    """Field from the rising-branch cubic, with its statistical uncertainty."""
    i_coil = np.atleast_1d(np.asarray(i_coil, dtype=float))
    b = np.polyval(mag_c3, i_coil)
    var = np.empty_like(b)
    for k, ic in enumerate(i_coil):
        g = np.array([ic**3, ic**2, ic, 1.0])
        var[k] = g @ mag_cov3 @ g                      # polynomial covariance
    dbdi = np.polyval(np.polyder(mag_c3), i_coil)
    var += (dbdi * SIG_ICOIL) ** 2                     # coil-current reading
    var += SIG_B_BRANCH**2                             # branch reproducibility
    return b, np.sqrt(var)


# --- the five magnet settings used for every Hall measurement --------------------
em_volts = np.array([6.0, 11.0, 16.0, 20.5, 24.0])
i_coil_set = em_volts / R_COIL
B_SET, SIG_B_SET = b_from_calibration(i_coil_set)     # what the student believes
B_SET_TRUE = b_true_rising(i_coil_set)                # what the magnet actually does
B_CAL_ERR = B_SET - B_SET_TRUE

print("\n(a) Magnet calibration")
print("    cubic fit B(I) = %+.5f I^3 %+.5f I^2 %+.5f I %+.5f  (T, I in A)"
      % (mag_c3[0], mag_c3[1], mag_c3[2], mag_c3[3]))
print("    chi2/dof: cubic = %.2f, quadratic = %.2f, linear = %.2f"
      % (chi2_3 / dof_3, chi2_2 / dof_2, chi2_1 / dof_1))
print("    mean branch gap = %.1f mT, gap at zero coil current = %.1f mT"
      % (gap_mean * 1e3, gap_at_zero * 1e3))
print("    working fields from the calibration (T): "
      + ", ".join("%.4f" % b for b in B_SET))
print("    true fields at those settings      (T): "
      + ", ".join("%.4f" % b for b in B_SET_TRUE))
print("    calibration error                 (mT): "
      + ", ".join("%+.2f" % (b * 1e3) for b in B_CAL_ERR))

# --- save calibration CSV ---------------------------------------------------------
with open(os.path.join(DATA_DIR, "hall_magnet_calibration.csv"), "w") as f:
    f.write("# REPRESENTATIVE DATA. Not a record of measured results.\n")
    f.write("# Electromagnet calibration against a Hall probe magnetometer,\n")
    f.write("# sample stand removed. Rising sweep then falling sweep.\n")
    f.write("branch,magnet_supply_V,coil_current_A,B_measured_T,sigma_B_T\n")
    for k in range(n_mag):
        f.write("rising,%.2f,%.4f,%.5f,%.5f\n"
                % (mag_supply[k], i_coil_up[k], b_meas_up[k], sig_b_up[k]))
    for k in range(n_mag - 1, -1, -1):
        f.write("falling,%.2f,%.4f,%.5f,%.5f\n"
                % (mag_supply[k], i_coil_dn[k], b_meas_dn[k], sig_b_dn[k]))


# ==================================================================================
# PART (b), (e) -- HALL SWEEPS AT 300 K AND 77 K
# ==================================================================================

sup_volts = np.linspace(3.0, 27.0, 18)       # V applied across bar + 300 kohm


def make_hall_dataset(T):
    """Generate V_H against I at each of the five magnet settings.

    The Hall voltage is generated from the TRUE field, B_SET_TRUE. The analysis later
    uses the CALIBRATED field, B_SET. The difference is the magnet calibration error
    and it propagates into the answer exactly as it would in the laboratory.
    """
    p = TRUE[T]
    r_loop = R_SERIES + p["R_L"]
    k_offset = p["R_sq"] * DELTA_MISALIGN / W_ACTIVE      # ohm, I-proportional leak
    rows = []
    for j in range(B_SET.size):
        b_true = B_SET_TRUE[j]
        i_true = sup_volts / r_loop
        v_true = p["R_H2D"] * i_true * b_true + k_offset * i_true + V_OFFSET_STATIC[T]
        sign = 1.0 if rng.random() > 0.5 else -1.0
        v_true = v_true + drift_profile(sup_volts.size, DRIFT_PP[T], sign)
        i_meas = i_true + rng.normal(0.0, sigma_i(i_true))
        v_meas = v_true + rng.normal(0.0, sigma_vh(v_true))
        for k in range(sup_volts.size):
            rows.append(
                dict(
                    field_index=j,
                    em_volts=em_volts[j],
                    B_T=B_SET[j],
                    sig_B_T=SIG_B_SET[j],
                    sup_V=sup_volts[k],
                    I_A=i_meas[k],
                    sig_I_A=float(sigma_i(i_meas[k])),
                    VH_V=v_meas[k],
                    sig_VH_V=float(sigma_vh(v_meas[k])),
                )
            )
    return rows, k_offset


hall = {}
k_offset_true = {}
for T in ("300K", "77K"):
    hall[T], k_offset_true[T] = make_hall_dataset(T)

for T in ("300K", "77K"):
    with open(os.path.join(DATA_DIR, "hall_VH_%s.csv" % T), "w") as f:
        f.write("# REPRESENTATIVE DATA. Not a record of measured results.\n")
        f.write("# Hall voltage between contacts 1 and 2, sample at %s.\n" % T)
        f.write("# B_T is the field from the rising-branch calibration polynomial.\n")
        f.write("magnet_supply_V,B_T,sigma_B_T,bar_supply_V,"
                "I_uA,sigma_I_uA,V_H_mV,sigma_V_H_mV\n")
        for r in hall[T]:
            f.write("%.1f,%.5f,%.5f,%.3f,%.4f,%.4f,%.4f,%.4f\n"
                    % (r["em_volts"], r["B_T"], r["sig_B_T"], r["sup_V"],
                       r["I_A"] * 1e6, r["sig_I_A"] * 1e6,
                       r["VH_V"] * 1e3, r["sig_VH_V"] * 1e3))


# ==================================================================================
# PART (c), (e) -- LONGITUDINAL MEASUREMENT AT ZERO FIELD
# ==================================================================================

def make_long_dataset(T):
    p = TRUE[T]
    r_loop = R_SERIES + p["R_L"]
    i_true = sup_volts / r_loop
    v_true = i_true * p["R_L"] + V_OFFSET_STATIC_L[T]
    sign = 1.0 if rng.random() > 0.5 else -1.0
    v_true = v_true + drift_profile(sup_volts.size, DRIFT_PP[T], sign)
    i_meas = i_true + rng.normal(0.0, sigma_i(i_true))
    v_meas = v_true + rng.normal(0.0, sigma_vl(v_true, T))
    return i_meas, sigma_i(i_meas), v_meas, sigma_vl(v_meas, T)


longit = {T: make_long_dataset(T) for T in ("300K", "77K")}

for T in ("300K", "77K"):
    i_m, si, v_m, sv = longit[T]
    with open(os.path.join(DATA_DIR, "hall_VL_%s.csv" % T), "w") as f:
        f.write("# REPRESENTATIVE DATA. Not a record of measured results.\n")
        f.write("# Longitudinal voltage between contacts 1 and 4, "
                "magnet off (B = 0), sample at %s.\n" % T)
        f.write("bar_supply_V,I_uA,sigma_I_uA,V_L_mV,sigma_V_L_mV\n")
        for k in range(sup_volts.size):
            f.write("%.3f,%.4f,%.4f,%.4f,%.4f\n"
                    % (sup_volts[k], i_m[k] * 1e6, si[k] * 1e6,
                       v_m[k] * 1e3, sv[k] * 1e3))


# ==================================================================================
# PART (d) -- ANGULAR DEPENDENCE AT 300 K
# ==================================================================================

THETA0_TRUE = 6.0                      # degrees, mounting misalignment
angles = np.arange(0.0, 360.0001, 15.0)
n_ang = angles.size

p300 = TRUE["300K"]
sup_ang = 18.3                         # V, chosen to give about 60 uA
i_ang_true = sup_ang / (R_SERIES + p300["R_L"])
B_ang = B_SET[3]                       # calibrated value, about 0.30 T
B_ang_true = B_SET_TRUE[3]
sig_B_ang = SIG_B_SET[3]
k_off_300 = k_offset_true["300K"]

vh_ang_true = (
    p300["R_H2D"] * i_ang_true * B_ang_true * np.cos(np.radians(angles - THETA0_TRUE))
    + k_off_300 * i_ang_true
    + V_OFFSET_STATIC["300K"]
)
vh_ang_true = vh_ang_true + drift_profile(n_ang, DRIFT_PP["300K"], 1.0)
i_ang_meas = i_ang_true + rng.normal(0.0, sigma_i(i_ang_true), n_ang)
vh_ang = vh_ang_true + rng.normal(0.0, sigma_vh(vh_ang_true))
sig_vh_ang = sigma_vh(vh_ang)

with open(os.path.join(DATA_DIR, "hall_angle_300K.csv"), "w") as f:
    f.write("# REPRESENTATIVE DATA. Not a record of measured results.\n")
    f.write("# Hall voltage against angle between the Hall bar normal and B, 300 K.\n")
    f.write("# Fixed magnet supply %.1f V (calibrated B = %.4f T), "
            "fixed bar supply %.1f V.\n" % (em_volts[3], B_ang, sup_ang))
    f.write("angle_deg,I_uA,B_T,V_H_mV,sigma_V_H_mV\n")
    for k in range(n_ang):
        f.write("%.1f,%.4f,%.5f,%.4f,%.4f\n"
                % (angles[k], i_ang_meas[k] * 1e6, B_ang,
                   vh_ang[k] * 1e3, sig_vh_ang[k] * 1e3))


def fun_cos(x, a, b, c):
    return a * np.cos(np.radians(x - b)) + c


ang_p, ang_cov = op.curve_fit(
    fun_cos, angles, vh_ang, p0=[0.02, 0.0, 0.001],
    sigma=sig_vh_ang, absolute_sigma=True,
)
ang_err = np.sqrt(np.diag(ang_cov))
ang_res = vh_ang - fun_cos(angles, *ang_p)
ang_chi2 = float(np.sum((ang_res / sig_vh_ang) ** 2))
ang_dof = n_ang - 3

print("\n(d) Angular scan")
print("    fitted amplitude   = %.4f +/- %.4f mV" % (ang_p[0] * 1e3, ang_err[0] * 1e3))
print("    fitted phase       = %.2f +/- %.2f deg" % (ang_p[1], ang_err[1]))
print("    fitted offset      = %.4f +/- %.4f mV" % (ang_p[2] * 1e3, ang_err[2] * 1e3))
print("    chi2/dof           = %.2f" % (ang_chi2 / ang_dof))


# ==================================================================================
# PART (e) -- 77 K THERMOMETRY
# ==================================================================================

tc_readings = T_TRUE_77 + np.linspace(-0.6, 1.1, 12) + rng.normal(0.0, SIG_TC, 12)
T_mean = float(np.mean(tc_readings))
T_sd = float(np.std(tc_readings, ddof=1))
T_sem = T_sd / np.sqrt(tc_readings.size)
T_quoted_err = quad(T_sem, 2.5)     # 2.5 K type K accuracy floor below 200 K

with open(os.path.join(DATA_DIR, "hall_thermocouple_77K.csv"), "w") as f:
    f.write("# REPRESENTATIVE DATA. Not a record of measured results.\n")
    f.write("# Type K thermocouple on the sample mount during the cryogenic run.\n")
    f.write("reading_index,T_K\n")
    for k, t in enumerate(tc_readings, start=1):
        f.write("%d,%.2f\n" % (k, t))


# ==================================================================================
# PART (f) -- DATA REDUCTION (objective 6)
# ==================================================================================
#
# Weighting policy. The weights used in every fit contain only the PER-POINT RANDOM
# uncertainties: the voltmeter scatter, and the current reading scatter folded in
# through the local slope (the effective variance method). The magnet calibration
# uncertainty is deliberately NOT put into the per-point weights, because it is a
# single correlated systematic shared by all the points at a given field setting, not
# an independent random error. It is carried separately into the final uncertainty
# budget in section 8 of the results file.

def reduce_hall(T):
    """Three reductions of the whole (B, I) array onto a single R_H,2D."""
    rows = hall[T]
    B = np.array([r["B_T"] for r in rows])
    sB = np.array([r["sig_B_T"] for r in rows])
    I = np.array([r["I_A"] for r in rows])
    sI = np.array([r["sig_I_A"] for r in rows])
    V = np.array([r["VH_V"] for r in rows])
    sV = np.array([r["sig_VH_V"] for r in rows])
    idx = np.array([r["field_index"] for r in rows])
    x = I * B
    sx_rand = B * sI                       # random part of the abscissa uncertainty

    # ---- R1: fit V_H against I at each fixed B, then the slopes against B --------
    slopes, slope_errs, ints, int_errs, chis = [], [], [], [], []
    for j in range(B_SET.size):
        m = idx == j
        X = np.column_stack([I[m], np.ones(m.sum())])
        sig_eff = sV[m].copy()
        for _ in range(4):
            beta, cov, c2, dof = wls(X, V[m], sig_eff)
            sig_eff = np.sqrt(sV[m] ** 2 + (beta[0] * sI[m]) ** 2)
        slopes.append(beta[0])
        slope_errs.append(float(np.sqrt(cov[0, 0])))
        ints.append(beta[1])
        int_errs.append(float(np.sqrt(cov[1, 1])))
        chis.append(c2 / dof)
    slopes = np.array(slopes)
    slope_errs = np.array(slope_errs)

    Xs = np.column_stack([B_SET, np.ones(B_SET.size)])
    beta_s, cov_s, chi2_s, dof_s = wls(Xs, slopes, slope_errs)
    r1 = dict(
        m_H=beta_s[0], s_m_H=float(np.sqrt(cov_s[0, 0])),
        k=beta_s[1], s_k=float(np.sqrt(cov_s[1, 1])),
        chi2=chi2_s, dof=dof_s,
        slopes=slopes, slope_errs=slope_errs,
        ints=np.array(ints), int_errs=np.array(int_errs), chis=np.array(chis),
        resid=slopes - Xs @ beta_s,
    )

    # ---- R2: naive global two-parameter fit V_H = m_H (IB) + c ------------------
    X2 = np.column_stack([x, np.ones(x.size)])
    sig_eff = sV.copy()
    for _ in range(4):
        beta2, cov2, chi2_2g, dof_2g = wls(X2, V, sig_eff)
        sig_eff = np.sqrt(sV**2 + (beta2[0] * sx_rand) ** 2)
    r2 = dict(m_H=beta2[0], s_m_H=float(np.sqrt(cov2[0, 0])),
              c=beta2[1], s_c=float(np.sqrt(cov2[1, 1])),
              chi2=chi2_2g, dof=dof_2g, sig_eff=sig_eff.copy(),
              resid=V - X2 @ beta2)

    # ---- R3: global three-parameter fit V_H = m_H (IB) + k I + c ----------------
    X3 = np.column_stack([x, I, np.ones(x.size)])
    sig_eff3 = sV.copy()
    for _ in range(4):
        beta3, cov3, chi2_3g, dof_3g = wls(X3, V, sig_eff3)
        sig_eff3 = np.sqrt(sV**2 + (beta3[0] * sx_rand) ** 2)
    r3 = dict(m_H=beta3[0], s_m_H=float(np.sqrt(cov3[0, 0])),
              k=beta3[1], s_k=float(np.sqrt(cov3[1, 1])),
              c=beta3[2], s_c=float(np.sqrt(cov3[2, 2])),
              corr_mk=float(cov3[0, 1] / np.sqrt(cov3[0, 0] * cov3[1, 1])),
              chi2=chi2_3g, dof=dof_3g, sig_eff=sig_eff3.copy(),
              resid=V - X3 @ beta3)

    return dict(B=B, sB=sB, I=I, sI=sI, V=V, sV=sV, idx=idx, x=x, sx=sx_rand,
                R1=r1, R2=r2, R3=r3)


def reduce_long(T):
    i_m, si, v_m, sv = longit[T]
    X = np.column_stack([i_m, np.ones(i_m.size)])
    sig_eff = sv.copy()
    for _ in range(4):
        beta, cov, chi2, dof = wls(X, v_m, sig_eff)
        sig_eff = np.sqrt(sv**2 + (beta[0] * si) ** 2)
    return dict(m_L=beta[0], s_m_L=float(np.sqrt(cov[0, 0])),
                c_L=beta[1], s_c_L=float(np.sqrt(cov[1, 1])),
                chi2=chi2, dof=dof, sig_eff=sig_eff.copy(),
                resid=v_m - X @ beta, I=i_m, sI=si, V=v_m, sV=sv)


red = {T: reduce_hall(T) for T in ("300K", "77K")}
lred = {T: reduce_long(T) for T in ("300K", "77K")}

# Fractional systematic on B: probe accuracy and positioning, a floor, and the
# statistical uncertainty of the calibration polynomial itself.
F_BSYST = quad(
    SYST_B_FRAC,
    SYST_B_FLOOR / float(np.mean(B_SET)),
    float(np.mean(SIG_B_SET / B_SET)),
)


def finalise(T):
    r3 = red[T]["R3"]
    L = lred[T]

    m_H, s_mH = r3["m_H"], r3["s_m_H"]          # ohm / T
    m_L, s_mL = L["m_L"], L["s_m_L"]            # ohm

    f_mH = s_mH / m_H
    f_mL = s_mL / m_L

    R_sq = m_L / N_SQUARES
    f_Rsq_stat = f_mL
    f_Rsq = quad(f_mL, SYST_GEOM_FRAC)

    n_sq = 1.0 / (E_CHARGE * m_H)
    f_n = quad(f_mH, F_BSYST)

    mu = N_SQUARES * m_H / m_L
    f_mu_stat = quad(f_mH, f_mL)
    f_mu = quad(f_mH, f_mL, SYST_GEOM_FRAC, F_BSYST)

    # product n_sq * mu = 1/(e R_sq) is immune to the magnet calibration
    nmu = 1.0 / (E_CHARGE * R_sq)
    f_nmu = quad(f_mL, SYST_GEOM_FRAC)

    tau = mu * M_STAR / E_CHARGE
    kF = np.sqrt(2.0 * np.pi * n_sq)
    vF = HBAR * kF / M_STAR
    mfp = vF * tau
    EF = np.pi * HBAR**2 * n_sq / M_STAR
    TF = EF / KB
    omct = mu * float(np.max(B_SET))

    return dict(
        m_H=m_H, s_mH=s_mH, m_L=m_L, s_mL=s_mL,
        R_H2D=m_H, s_R_H2D_stat=s_mH, s_R_H2D=m_H * f_n,
        R_sq=R_sq, s_R_sq_stat=R_sq * f_Rsq_stat, s_R_sq=R_sq * f_Rsq,
        R_L=m_L, s_R_L=s_mL,
        n_sq=n_sq, s_n_stat=n_sq * f_mH, s_n=n_sq * f_n,
        mu=mu, s_mu_stat=mu * f_mu_stat, s_mu=mu * f_mu,
        nmu=nmu, s_nmu=nmu * f_nmu,
        f_mH=f_mH, f_mL=f_mL, f_n=f_n, f_Rsq=f_Rsq, f_mu=f_mu,
        tau=tau, kF=kF, vF=vF, mfp=mfp, EF=EF, TF=TF, omct=omct,
    )


FIN = {T: finalise(T) for T in ("300K", "77K")}

print("\n(f) Global weighted fits")
for T in ("300K", "77K"):
    r1, r2, r3 = red[T]["R1"], red[T]["R2"], red[T]["R3"]
    print("    %-5s R1 slope of slopes : m_H = %8.2f +/- %5.2f ohm/T, chi2/dof = %6.2f"
          % (T, r1["m_H"], r1["s_m_H"], r1["chi2"] / r1["dof"]))
    print("    %-5s R2 naive global    : m_H = %8.2f +/- %5.2f ohm/T, chi2/dof = %6.2f"
          % (T, r2["m_H"], r2["s_m_H"], r2["chi2"] / r2["dof"]))
    print("    %-5s R3 global + offset : m_H = %8.2f +/- %5.2f ohm/T, chi2/dof = %6.2f"
          % (T, r3["m_H"], r3["s_m_H"], r3["chi2"] / r3["dof"]))


# ==================================================================================
# FIGURES
# ==================================================================================

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("    wrote", path)


print("\nFigures")

# --- fig 1: magnet calibration ----------------------------------------------------
fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(6.2, 6.4), sharex=True,
    gridspec_kw=dict(height_ratios=[2.4, 1.15, 1.0]),
)
ax0.errorbar(i_coil_up, b_meas_up * 1e3, yerr=sig_b_up * 1e3, fmt="o",
             color=CB["blue"], mfc="white", capsize=2, elinewidth=0.8, ls="none",
             label="rising branch")
ax0.errorbar(i_coil_dn, b_meas_dn * 1e3, yerr=sig_b_dn * 1e3, fmt="s",
             color=CB["verm"], mfc="white", capsize=2, elinewidth=0.8, ls="none",
             label="falling branch")
ax0.plot(igrid, np.polyval(mag_c3, igrid) * 1e3, "-", color=CB["black"],
         label="cubic fit, rising branch")
ax0.plot(igrid, np.polyval(mag_c1, igrid) * 1e3, ":", color=CB["grey"],
         label="straight line, for comparison")
ax0.plot(i_coil_set, B_SET * 1e3, marker="*", ms=10, color=CB["green"], ls="none",
         zorder=5, label="working set points")
ax0.set_ylabel("Magnetic flux density $B$ (mT)")
ax0.set_ylim(-15, 400)
ax0.legend(loc="upper left", framealpha=0.9, title="Representative data")

ax1.axhline(0.0, color=CB["grey"], lw=0.8)
ax1.errorbar(i_coil_up, res3 * 1e3, yerr=sig_b_up * 1e3, fmt="o", color=CB["blue"],
             mfc="white", capsize=2, elinewidth=0.8, ls="none",
             label="cubic, $\\chi^2/\\nu$ = %.2f" % (chi2_3 / dof_3))
ax1.plot(i_coil_up, res2 * 1e3, "^", color=CB["orange"], mfc="none",
         label="quadratic, $\\chi^2/\\nu$ = %.2f" % (chi2_2 / dof_2))
ax1.set_ylabel("Residual (mT)")
ax1.set_ylim(-13.5, 9.0)
ax1.legend(loc="lower center", ncol=2, framealpha=0.9)

ax2.axhline(0.0, color=CB["grey"], lw=0.8)
ax2.plot(i_nom, gap_points * 1e3, "d", color=CB["purple"], mfc="none",
         label="measured branch difference")
ax2.plot(igrid, gap_fit * 1e3, "-", color=CB["purple"],
         label="difference of the two cubic fits")
ax2.set_ylabel("Hysteresis gap (mT)")
ax2.set_xlabel("Electromagnet coil current $I_m$ (A)")
ax2.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.subplots_adjust(hspace=0.10)
save(fig, "fig_hall_magnet_calibration.pdf")

# --- fig 2: V_H against I, family of fields, 300 K --------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.9))
d = red["300K"]
for j in range(B_SET.size):
    m = d["idx"] == j
    ax.errorbar(d["I"][m] * 1e6, d["V"][m] * 1e3, yerr=d["sV"][m] * 1e3,
                xerr=d["sI"][m] * 1e6, fmt=MARKERS[j], color=SERIES[j],
                mfc="white", capsize=2, elinewidth=0.8, ls="none",
                label="$B$ = %.3f T, slope %.0f $\\Omega$"
                      % (B_SET[j], d["R1"]["slopes"][j]))
    igr = np.linspace(0.0, 95e-6, 50)
    ax.plot(igr * 1e6, (d["R1"]["slopes"][j] * igr + d["R1"]["ints"][j]) * 1e3,
            "-", color=SERIES[j], lw=1.0)
ax.set_xlabel("Sample current $I$ ($\\mu$A)")
ax.set_ylabel("Hall voltage $V_H$ (mV)")
ax.set_xlim(0, 95)
ax.legend(loc="upper left", ncol=2, framealpha=0.9,
          title="Representative data, 300 K")
fig.tight_layout()
save(fig, "fig_hall_VH_vs_I.pdf")

# --- fig 3: the global collapse, V_H against I*B, 300 K --------------------------
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(6.2, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.3])
)
d = red["300K"]
xs = d["x"] * 1e6          # microamp tesla
for j in range(B_SET.size):
    m = d["idx"] == j
    axA.errorbar(xs[m], d["V"][m] * 1e3, yerr=d["sV"][m] * 1e3,
                 fmt=MARKERS[j], color=SERIES[j], mfc="white",
                 capsize=2, elinewidth=0.8, ls="none",
                 label="$B$ = %.3f T" % B_SET[j])
xg = np.linspace(0.0, xs.max() * 1.03, 50)
axA.plot(xg, (d["R2"]["m_H"] * xg * 1e-6 + d["R2"]["c"]) * 1e3, "-",
         color=CB["black"],
         label="global fit, $R_{H,2D}$ = %.0f $\\Omega$ T$^{-1}$" % d["R2"]["m_H"])
axA.set_ylabel("Hall voltage $V_H$ (mV)")
axA.legend(loc="upper left", ncol=2, framealpha=0.9,
           title="Representative data, 300 K")

axB.axhline(0.0, color=CB["grey"], lw=0.8)
for j in range(B_SET.size):
    m = d["idx"] == j
    axB.errorbar(xs[m], d["R2"]["resid"][m] * 1e3, yerr=d["sV"][m] * 1e3,
                 fmt=MARKERS[j], color=SERIES[j], mfc="white",
                 capsize=2, elinewidth=0.7, ls="none")
axB.plot(xs, d["R3"]["resid"] * 1e3, "x", color=CB["black"], ms=4,
         label="residuals after adding the $kI$ misalignment term")
axB.set_xlabel("$I B$ ($\\mu$A T)")
axB.set_ylabel("Residual (mV)")
# Open headroom below the data so that the legend box cannot sit on top of the
# lowest residual series, which is the banding the caption points at.
_rlo = min(float(np.min(d["R2"]["resid"])), float(np.min(d["R3"]["resid"]))) * 1e3
_rhi = max(float(np.max(d["R2"]["resid"])), float(np.max(d["R3"]["resid"]))) * 1e3
_rspan = _rhi - _rlo
axB.set_ylim(_rlo - 0.80 * _rspan, _rhi + 0.15 * _rspan)
axB.legend(loc="lower left", framealpha=0.95)
fig.tight_layout()
fig.subplots_adjust(hspace=0.08)
save(fig, "fig_hall_VH_vs_IB.pdf")

# --- fig 4: V_L against I at zero field, 300 K -----------------------------------
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(6.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.0])
)
L = lred["300K"]
axA.errorbar(L["I"] * 1e6, L["V"] * 1e3, yerr=L["sV"] * 1e3, xerr=L["sI"] * 1e6,
             fmt="o", color=CB["blue"], mfc="white", capsize=2, elinewidth=0.8,
             ls="none", label="representative data, 300 K, $B$ = 0")
igr = np.linspace(0.0, 95e-6, 50)
axA.plot(igr * 1e6, (L["m_L"] * igr + L["c_L"]) * 1e3, "-", color=CB["black"],
         label="weighted fit, $R_L$ = %.0f $\\Omega$, $R_{sq}$ = %.0f $\\Omega$/sq"
               % (L["m_L"], L["m_L"] / N_SQUARES))
axA.set_ylabel("Longitudinal voltage $V_L$ (mV)")
axA.legend(loc="upper left", framealpha=0.9)
axB.axhline(0.0, color=CB["grey"], lw=0.8)
axB.errorbar(L["I"] * 1e6, L["resid"] * 1e3, yerr=L["sV"] * 1e3, fmt="o",
             color=CB["blue"], mfc="white", capsize=2, elinewidth=0.8, ls="none")
axB.set_xlabel("Sample current $I$ ($\\mu$A)")
axB.set_ylabel("Residual (mV)")
axB.set_xlim(0, 95)
fig.tight_layout()
fig.subplots_adjust(hspace=0.08)
save(fig, "fig_hall_VL_vs_I.pdf")

# --- fig 5: angular dependence ---------------------------------------------------
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(6.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.0])
)
axA.axhline(0.0, color=CB["grey"], lw=0.8)
axA.errorbar(angles, vh_ang * 1e3, yerr=sig_vh_ang * 1e3, fmt="o",
             color=CB["blue"], mfc="white", capsize=2, elinewidth=0.8, ls="none",
             label="representative data, 300 K")
tg = np.linspace(0.0, 360.0, 400)
axA.plot(tg, fun_cos(tg, *ang_p) * 1e3, "-", color=CB["black"],
         label="$V_{H0}\\cos(\\theta-\\theta_0)+C$ fit")
axA.axvline(ang_p[1], color=CB["green"], ls="--", lw=0.9,
            label="$\\theta_0 = %.2f^\\circ$" % ang_p[1])
axA.set_ylabel("Hall voltage $V_H$ (mV)")
axA.set_xlim(-10, 370)
axA.set_ylim(-33.0, 29.0)
axA.set_xticks(np.arange(0, 361, 45))
axA.legend(loc="lower center", ncol=3, framealpha=0.9)
axB.axhline(0.0, color=CB["grey"], lw=0.8)
axB.errorbar(angles, ang_res * 1e3, yerr=sig_vh_ang * 1e3, fmt="o",
             color=CB["blue"], mfc="white", capsize=2, elinewidth=0.8, ls="none")
axB.set_xlabel("Rotation stage angle $\\theta$ (degrees)")
axB.set_ylabel("Residual (mV)")
fig.tight_layout()
fig.subplots_adjust(hspace=0.08)
save(fig, "fig_hall_angle.pdf")

# --- fig 6: 300 K against 77 K ----------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.2, 3.4))
T_LABEL = {"300K": "300 K", "77K": "%.0f K" % T_mean}
for T, col, mk in (("300K", CB["blue"], "o"), ("77K", CB["verm"], "s")):
    d = red[T]
    # Hall voltage with the fitted misalignment and constant offset removed, so the
    # remaining signal is the pure Hall term and the fitted line passes through zero.
    v_corr = d["V"] - d["R3"]["k"] * d["I"] - d["R3"]["c"]
    axA.errorbar(d["x"] * 1e6, v_corr * 1e3, yerr=d["sV"] * 1e3, fmt=mk,
                 color=col, mfc="white", capsize=1.5, elinewidth=0.6, ms=3.2,
                 ls="none", label="%s, $R_{H,2D}$ = %.0f $\\Omega$ T$^{-1}$"
                                  % (T_LABEL[T], d["R3"]["m_H"]))
    xg = np.linspace(0.0, d["x"].max() * 1e6 * 1.03, 50)
    axA.plot(xg, (d["R3"]["m_H"] * xg * 1e-6) * 1e3, "-", color=col, lw=1.0)
axA.set_xlabel("$I B$ ($\\mu$A T)")
axA.set_ylabel("$V_H$, offset removed (mV)")
axA.legend(loc="upper left", framealpha=0.9, title="Representative data")

for T, col, mk in (("300K", CB["blue"], "o"), ("77K", CB["verm"], "s")):
    L = lred[T]
    axB.errorbar(L["I"] * 1e6, L["V"] * 1e3, yerr=L["sV"] * 1e3, fmt=mk,
                 color=col, mfc="white", capsize=1.5, elinewidth=0.6, ms=3.2,
                 ls="none", label=T_LABEL[T])
    igr = np.linspace(0.0, 95e-6, 50)
    axB.plot(igr * 1e6, (L["m_L"] * igr + L["c_L"]) * 1e3, "-", color=col, lw=1.0)
axB.set_xlabel("Sample current $I$ ($\\mu$A)")
axB.set_ylabel("Longitudinal voltage $V_L$ (mV)")
axB.set_xlim(0, 95)
# The cryogenic trace runs along the bottom of this panel and the room-temperature
# trace runs diagonally, so the inset goes in the empty upper-left corner and the
# legend into the empty wedge on the right, between the two traces.
axB.set_ylim(0, 460)
axB.legend(loc="center right", framealpha=0.95,
           title="Representative data, $B$ = 0")
fig.tight_layout()

axins = axB.inset_axes([0.08, 0.55, 0.42, 0.38])
L = lred["77K"]
axins.errorbar(L["I"] * 1e6, L["V"] * 1e3, yerr=L["sV"] * 1e3, fmt="s",
               color=CB["verm"], mfc="white", capsize=1.2, elinewidth=0.5, ms=2.6,
               ls="none")
axins.plot(np.linspace(0, 95, 50),
           (L["m_L"] * np.linspace(0, 95e-6, 50) + L["c_L"]) * 1e3, "-",
           color=CB["verm"], lw=0.9)
axins.set_xlim(0, 95)
axins.tick_params(labelsize=6.0, pad=1.5)
axins.grid(alpha=0.2, lw=0.4)
axins.set_xlabel("$I$ ($\\mu$A)", fontsize=6.0, labelpad=1.0)
axins.set_ylabel("$V_L$ (mV)", fontsize=6.0, labelpad=1.0)
axins.set_title("%s, expanded" % T_LABEL["77K"], fontsize=6.5, pad=2)
save(fig, "fig_hall_77K_comparison.pdf")


# ==================================================================================
# SANITY CHECKS
# ==================================================================================

checks = []


def check(label, ok, detail):
    checks.append((label, bool(ok), detail))


I_all = np.concatenate([red[T]["I"] for T in ("300K", "77K")]
                      + [lred[T]["I"] for T in ("300K", "77K")]
                      + [i_ang_meas])
VH_all = np.concatenate([np.abs(red[T]["V"]) for T in ("300K", "77K")]
                        + [np.abs(vh_ang)])
VL_all = np.concatenate([np.abs(lred[T]["V"]) for T in ("300K", "77K")])

f300, f77 = FIN["300K"], FIN["77K"]

check("max sample current below the 100 uA software cap",
      I_all.max() < I_CAP, "max I = %.2f uA" % (I_all.max() * 1e6))
check("Hall voltages fit a 200 mV DMM range",
      VH_all.max() < 0.200, "max |V_H| = %.1f mV" % (VH_all.max() * 1e3))
check("longitudinal voltages fit a 1 V DMM range",
      VL_all.max() < 1.0, "max |V_L| = %.1f mV" % (VL_all.max() * 1e3))
check("smallest Hall voltage above the 50 uV noise floor",
      np.min(np.abs(red["300K"]["V"])) > 5 * SIG_VH_FLOOR,
      "min |V_H| at 300 K = %.3f mV" % (np.min(np.abs(red["300K"]["V"])) * 1e3))
check("peak field between 0.30 and 0.40 T",
      0.30 <= B_SET.max() <= 0.40, "B_max = %.4f T" % B_SET.max())
check("mobility rises on cooling",
      f77["mu"] > f300["mu"], "ratio = %.1f" % (f77["mu"] / f300["mu"]))
check("mobility ratio in the literature range 10 to 40",
      10.0 <= f77["mu"] / f300["mu"] <= 40.0,
      "ratio = %.1f" % (f77["mu"] / f300["mu"]))
check("sheet density falls by 10 to 25 per cent on cooling",
      0.10 <= 1.0 - f77["n_sq"] / f300["n_sq"] <= 0.25,
      "fall = %.1f per cent" % (100.0 * (1.0 - f77["n_sq"] / f300["n_sq"])))
check("fitted n_sq at 300 K within the quoted total uncertainty of ground truth",
      abs(f300["n_sq"] - TRUE["300K"]["n_sq"]) < f300["s_n"],
      "%.2f per cent, quoted +/- %.1f per cent"
      % (pct_diff(f300["n_sq"], TRUE["300K"]["n_sq"]), f300["f_n"] * 100))
check("fitted n_sq at 77 K within the quoted total uncertainty of ground truth",
      abs(f77["n_sq"] - TRUE["77K"]["n_sq"]) < f77["s_n"],
      "%.2f per cent, quoted +/- %.1f per cent"
      % (pct_diff(f77["n_sq"], TRUE["77K"]["n_sq"]), f77["f_n"] * 100))
check("fitted mu at 300 K within the quoted total uncertainty of ground truth",
      abs(f300["mu"] - TRUE["300K"]["mu"]) < f300["s_mu"],
      "%.2f per cent, quoted +/- %.1f per cent"
      % (pct_diff(f300["mu"], TRUE["300K"]["mu"]), f300["f_mu"] * 100))
check("fitted mu at 77 K within the quoted total uncertainty of ground truth",
      abs(f77["mu"] - TRUE["77K"]["mu"]) < f77["s_mu"],
      "%.2f per cent, quoted +/- %.1f per cent"
      % (pct_diff(f77["mu"], TRUE["77K"]["mu"]), f77["f_mu"] * 100))
check("fitted R_sq at 300 K within the quoted total uncertainty of ground truth",
      abs(f300["R_sq"] - TRUE["300K"]["R_sq"]) < f300["s_R_sq"],
      "%.2f per cent" % pct_diff(f300["R_sq"], TRUE["300K"]["R_sq"]))
check("fitted R_sq at 77 K within the quoted total uncertainty of ground truth",
      abs(f77["R_sq"] - TRUE["77K"]["R_sq"]) < f77["s_R_sq"],
      "%.2f per cent" % pct_diff(f77["R_sq"], TRUE["77K"]["R_sq"]))
check("angular fit phase within 1.5 deg of the true misalignment",
      abs(ang_p[1] - THETA0_TRUE) < 1.5,
      "fitted %.2f deg, true %.1f deg" % (ang_p[1], THETA0_TRUE))
check("angular fit chi2/dof near 1",
      0.4 < ang_chi2 / ang_dof < 2.2, "%.2f" % (ang_chi2 / ang_dof))
check("three-parameter Hall fit beats the two-parameter fit at 300 K",
      red["300K"]["R3"]["chi2"] < 0.5 * red["300K"]["R2"]["chi2"],
      "chi2/dof %.2f against %.2f"
      % (red["300K"]["R3"]["chi2"] / red["300K"]["R3"]["dof"],
         red["300K"]["R2"]["chi2"] / red["300K"]["R2"]["dof"]))
check("longitudinal fit chi2/dof near 1 at 300 K",
      0.3 < lred["300K"]["chi2"] / lred["300K"]["dof"] < 2.5,
      "%.2f" % (lred["300K"]["chi2"] / lred["300K"]["dof"]))
check("longitudinal fit chi2/dof near 1 at 77 K",
      0.3 < lred["77K"]["chi2"] / lred["77K"]["dof"] < 2.5,
      "%.2f" % (lred["77K"]["chi2"] / lred["77K"]["dof"]))
check("cubic magnet fit better than quadratic and linear",
      chi2_3 / dof_3 < chi2_2 / dof_2 < chi2_1 / dof_1,
      "cubic %.2f, quadratic %.2f, linear %.2f"
      % (chi2_3 / dof_3, chi2_2 / dof_2, chi2_1 / dof_1))
check("cubic magnet fit chi2/dof near 1",
      0.4 < chi2_3 / dof_3 < 2.2, "%.2f" % (chi2_3 / dof_3))
check("magnet calibration error under 1.5 per cent at every set point",
      np.all(np.abs(B_CAL_ERR / B_SET_TRUE) < 0.015),
      "max %.2f per cent" % (100 * np.max(np.abs(B_CAL_ERR / B_SET_TRUE))))
check("mean free path much shorter than the probe spacing (diffusive)",
      f77["mfp"] < 0.05 * L_ACTIVE,
      "l_mfp(77 K) = %.2f um against l = 300 um" % (f77["mfp"] * 1e6))
check("no Shubnikov-de Haas regime at the cryogenic temperature",
      HBAR * E_CHARGE * B_SET.max() / M_STAR < 0.25 * KB * T_mean,
      "hbar*omega_c = %.3f meV against kT = %.2f meV"
      % (HBAR * E_CHARGE * B_SET.max() / M_STAR / E_CHARGE * 1e3,
         KB * round(T_mean) / E_CHARGE * 1e3))
check("quoted cryogenic temperature within 10 K of the nitrogen boiling point",
      abs(T_mean - 77.4) < 10.0, "T = %.1f K" % T_mean)

print("\nSanity checks")
all_ok = True
for label, ok, detail in checks:
    all_ok &= ok
    print("    [%s] %-62s %s" % ("PASS" if ok else "FAIL", label, detail))
print("    overall:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")


# ==================================================================================
# RESULTS FILE
# ==================================================================================

LIT = {
    "n_sq_300": 5.6e15,      # m^-2, Seliuta et al. 2DEG-B
    "mu_300": 0.80,          # m^2 V^-1 s^-1, Seliuta et al. 2DEG-B (8000 cm2/Vs)
    "n_sq_77": 4.3e15,       # m^-2, Seliuta et al. 2DEG-B
    "mu_77": 11.6,           # m^2 V^-1 s^-1, Seliuta et al. 2DEG-B (116000 cm2/Vs)
    "R_sq_300": 2000.0,      # ohm/square, typical GaAs/AlGaAs 2DEG at 300 K
    "mu_bulk_300": 0.85,     # m^2 V^-1 s^-1, bulk undoped GaAs, Ioffe NSM
    "mu_77_stormer": 11.7,   # m^2 V^-1 s^-1, Stormer et al. 1979
}

lines = []
A = lines.append

A("# Results, Hall effect in a two-dimensional electron gas")
A("")
A("**REPRESENTATIVE DATA. Not a record of measured results.** Every number in this")
A("file was produced by `reports/analysis/generate_hall.py` from simulated readings")
A("built forward from physically accepted parameters, with realistic instrument")
A("scatter and realistic systematic offsets added. The random seed is fixed at")
A("`SEED = 4009`, so the whole file is reproducible.")
A("")
A("All fitted values below are what the least squares fits actually returned when run")
A("on the generated data. They are not the ground truth values that were fed in. In")
A("particular the Hall voltages were generated using the TRUE magnet field, while the")
A("analysis uses the field read off the CALIBRATION POLYNOMIAL, so the calibration")
A("error propagates into the answer exactly as it would in the laboratory. The ground")
A("truth is listed alongside in section 9.1 so the difference can be seen.")
A("")
A("---")
A("")
A("## 0. How to read this file")
A("")
A("- Every quantity carries its unit explicitly.")
A("- `stat` means the statistical uncertainty from the fit covariance matrix alone.")
A("- `total` means statistical combined in quadrature with the systematic terms of")
A("  section 8.")
A("- Percentage difference is quoted as (fitted - reference) / reference x 100.")
A("- Weighting policy: every fit weight contains only the per-point random errors,")
A("  that is the voltmeter scatter with the current reading scatter folded in through")
A("  the local slope. The magnet calibration uncertainty is deliberately kept out of")
A("  the weights, because it is one correlated systematic shared by all the points at")
A("  a given field setting and not an independent random error. It is carried into the")
A("  budget in section 8 instead.")
A("")
A("---")
A("")

# ---- 1. magnet calibration -------------------------------------------------------
A("## 1. Electromagnet calibration")
A("")
A("Data file: `data/hall_magnet_calibration.csv` (%d rows, %d rising and %d falling)."
  % (2 * n_mag, n_mag, n_mag))
A("Figure: `figures/fig_hall_magnet_calibration.pdf`.")
A("")
A("Magnet supply swept 0.0 V to %.1f V in 1.0 V steps into a coil of %.1f ohm, so the"
  % (MAG_V_MAX, R_COIL))
A("coil current ran from 0 A to %.3f A. Field read with a calibrated Hall probe"
  % I_COIL_MAX)
A("magnetometer with the sample stand removed. The sweep was taken upward and then")
A("downward through the same settings so the hysteresis loop is visible.")
A("")
A("Assumed uncertainty on one calibration point: Hall probe repeatability %.1f mT,"
  % (SIG_BPROBE * 1e3))
A("coil current reading %.1f mA folded in through the local slope, and branch"
  % (SIG_ICOIL * 1e3))
A("reproducibility %.1f mT. Combined, %.2f mT to %.2f mT across the range."
  % (SIG_B_BRANCH * 1e3, sig_b_up.min() * 1e3, sig_b_up.max() * 1e3))
A("")
A("### 1.1 Cubic fit to the rising branch")
A("")
A("Model, weighted least squares with weights 1/sigma_B:")
A("")
A("    B(I_m) = c3*I_m^3 + c2*I_m^2 + c1*I_m + c0   [B in T, I_m in A]")
A("")
A("| Coefficient | Fitted value | Standard error | Unit | True value used to generate |")
A("|---|---|---|---|---|")
A("| c3 | %+.6f | %.6f | T A^-3 | %+.6f |" % (mag_c3[0], mag_err3[0], MAG_POLY_TRUE[2]))
A("| c2 | %+.6f | %.6f | T A^-2 | %+.6f |" % (mag_c3[1], mag_err3[1], MAG_POLY_TRUE[1]))
A("| c1 | %+.6f | %.6f | T A^-1 | %+.6f |" % (mag_c3[2], mag_err3[2], MAG_POLY_TRUE[0]))
A("| c0 | %+.6f | %.6f | T | %+.6f |" % (mag_c3[3], mag_err3[3], B_REMANENT_UP))
A("")
A("- chi-square = %.2f on %d degrees of freedom, chi2/dof = **%.2f**"
  % (chi2_3, dof_3, chi2_3 / dof_3))
A("- RMS residual = **%.2f mT**, against an assumed point uncertainty of about %.1f mT"
  % (rms(res3) * 1e3, float(np.mean(sig_b_up)) * 1e3))
A("- The fitted constant term c0 = %.2f mT is the remanent field left in the iron core"
  % (mag_c3[3] * 1e3))
A("  after the previous cycle. The value used to generate the data was %.2f mT."
  % (B_REMANENT_UP * 1e3))
A("")
A("### 1.2 Why the fit has to be non-linear")
A("")
A("| Polynomial order | chi2 | dof | chi2/dof | RMS residual (mT) |")
A("|---|---|---|---|---|")
A("| 1 (straight line) | %.1f | %d | %.1f | %.2f |"
  % (chi2_1, dof_1, chi2_1 / dof_1, rms(res1) * 1e3))
A("| 2 (quadratic) | %.1f | %d | %.2f | %.2f |"
  % (chi2_2, dof_2, chi2_2 / dof_2, rms(res2) * 1e3))
A("| 3 (cubic) | %.1f | %d | %.2f | %.2f |"
  % (chi2_3, dof_3, chi2_3 / dof_3, rms(res3) * 1e3))
A("")
A("The iron yoke approaches magnetic saturation, so an increasing share of the")
A("magnetomotive force is spent driving flux through the iron rather than across the")
A("gap and the field grows more slowly than the coil current at the top of the range.")
A("A straight line fitted to these data over-reads the ADOPTED CUBIC at the top of the")
A("sweep by **%.0f mT**, which is **%.0f per cent** of the calibrated field there. That"
  % ((np.polyval(mag_c1, I_COIL_MAX) - np.polyval(mag_c3, I_COIL_MAX)) * 1e3,
     100 * (np.polyval(mag_c1, I_COIL_MAX) - np.polyval(mag_c3, I_COIL_MAX))
     / np.polyval(mag_c3, I_COIL_MAX)))
A("error would go straight into R_H,2D and hence into n_sq, so the non-linear fit is")
A("not a refinement, it is essential. (Against the ground truth, which the experimenter")
A("cannot know and which therefore must not appear in the report, the straight line")
A("over-reads by %.0f mT.)"
  % ((np.polyval(mag_c1, I_COIL_MAX) - b_true_rising(I_COIL_MAX)) * 1e3))
A("")
A("The quadratic already fits: chi2/dof = %.2f on %d degrees of freedom, and going to"
  % (chi2_2 / dof_2, dof_2))
A("the cubic buys only delta chi2 = %.1f for one more parameter, so the cubic"
  % (chi2_2 - chi2_3))
A("coefficient c3 is only %.1f standard errors from zero. The cubic is kept because the"
  % abs(mag_c3[0] / mag_err3[0]))
A("curve is known on physical grounds to bend over, and because the two models differ")
A("by at most %.2f mT at the five working set points, inside the uncertainty on B there."
  % (1e3 * float(np.max(np.abs(np.polyval(mag_c3, i_coil_set)
                               - np.polyval(mag_c2, i_coil_set))))))
A("")
A("### 1.3 Hysteresis")
A("")
A("| Quantity | Value | Unit |")
A("|---|---|---|")
A("| Mean separation of the two measured branches | %.2f | mT |" % (gap_mean * 1e3))
A("| Separation of the two fitted cubics at zero coil current | %.2f | mT |"
  % (gap_at_zero * 1e3))
A("| Largest separation of the two fitted cubics | %.2f | mT |" % (gap_max_fit * 1e3))
A("| Field at zero coil current, rising branch (fitted) | %.2f | mT |"
  % (remanence_up_fit * 1e3))
A("| Field at zero coil current, falling branch (fitted) | %.2f | mT |"
  % (remanence_dn_fit * 1e3))
A("")
A("Value used to generate the data: branch separation %.1f mT extrapolated to zero"
  % (HYST_GAP0 * 1e3))
A("coil current, falling to zero at the turning point.")
A("")
A("The procedure adopted for every later measurement was to approach each set point")
A("from below after cycling the magnet to maximum and back, so that the field history")
A("was at least the same from run to run. Note that this does NOT put calibration and")
A("measurement on the same branch: a down-cycle leaves the core at the falling-branch")
A("remanence of 8.92 mT, not at the 2.56 mT of the calibration rising branch, so the")
A("measurement traces a minor loop between the two. The resulting branch offset, about")
A("6 mT at low coil current and falling towards zero at the top of the sweep, is")
A("carried as part of the systematic uncertainty on B in section 8. Only demagnetising")
A("the core before each run would remove it.")
A("")
A("The difference of the two fitted cubics peaks above its zero-current value, which a")
A("closed loop cannot do; that is an artefact of differencing two independent cubic")
A("fits, not a feature of the loop. The measured mean separation is the figure used.")
A("")
A("### 1.4 The five working field settings")
A("")
A("| Magnet supply (V) | Coil current (A) | B from the cubic (T) | sigma_B stat (mT) |"
  " True B (T) | Calibration error (mT) |")
A("|---|---|---|---|---|---|")
for j in range(B_SET.size):
    A("| %.1f | %.3f | %.4f | %.2f | %.4f | %+.2f |"
      % (em_volts[j], i_coil_set[j], B_SET[j], SIG_B_SET[j] * 1e3,
         B_SET_TRUE[j], B_CAL_ERR[j] * 1e3))
A("")
A("Peak field %.4f T, inside the 0.3 to 0.4 T range expected of a teaching-scale"
  % B_SET.max())
A("iron-cored electromagnet. The calibration error column is not available to the")
A("experimenter. It is printed here only so that the honesty of this pipeline can be")
A("checked: the largest error is %.2f mT, or %.2f per cent, and it is what limits how"
  % (np.max(np.abs(B_CAL_ERR)) * 1e3,
     100 * np.max(np.abs(B_CAL_ERR / B_SET_TRUE))))
A("closely the fitted n_sq can approach the true value.")
A("")
A("---")
A("")

# ---- 2 and 5. Hall measurements ---------------------------------------------------
# Built into separate blocks so the finished file reads in section order 1 to 12.
hall_blocks = {}


def build_hall_block(T, tlab, sec):
    lines_b = []
    A = lines_b.append
    d = red[T]
    r1, r2, r3 = d["R1"], d["R2"], d["R3"]
    p = TRUE[T]
    A("## %s. Hall measurement at %s" % (sec, tlab))
    A("")
    A("Data file: `data/hall_VH_%s.csv` (%d rows: %d bar-supply steps at each of %d"
      % (T, len(hall[T]), sup_volts.size, B_SET.size))
    A("magnet settings). Bar supply swept %.1f V to %.1f V across the Hall bar in"
      % (sup_volts.min(), sup_volts.max()))
    A("series with the two 150 kohm protection resistors.")
    A("")
    A("Measured ranges:")
    A("")
    A("- sample current I: **%.2f uA to %.2f uA** (software cap 100 uA)"
      % (d["I"].min() * 1e6, d["I"].max() * 1e6))
    A("- Hall voltage V_H: **%.3f mV to %.3f mV**"
      % (d["V"].min() * 1e3, d["V"].max() * 1e3))
    A("- product I*B: **%.3f to %.3f uA T**"
      % (d["x"].min() * 1e6, d["x"].max() * 1e6))
    A("")
    A("### %s.1 Linearity in I at fixed B" % sec)
    A("")
    A("A weighted straight line V_H = s_j * I + b_j was fitted at each field setting.")
    A("")
    A("| B (T) | slope s_j (ohm) | sigma_s_j (ohm) | intercept b_j (mV) |"
      " sigma_b_j (mV) | chi2/dof |")
    A("|---|---|---|---|---|---|")
    for j in range(B_SET.size):
        A("| %.4f | %.1f | %.1f | %+.4f | %.4f | %.2f |"
          % (B_SET[j], r1["slopes"][j], r1["slope_errs"][j],
             r1["ints"][j] * 1e3, r1["int_errs"][j] * 1e3, r1["chis"][j]))
    A("")
    A("Every line is straight within the scatter. That is objective 2 for the current")
    A("dependence. The intercepts are all small, between %+.3f mV and %+.3f mV, and"
      % (r1["ints"].min() * 1e3, r1["ints"].max() * 1e3))
    A("they do not grow with field. That is the first sign of a field-independent")
    A("offset on the Hall contact pair.")
    A("")
    A("### %s.2 Linearity in B: the slopes against field" % sec)
    A("")
    A("The five slopes were then fitted against B, weighted, as s(B) = m_H * B + k:")
    A("")
    A("- m_H = R_H,2D = **%.1f +/- %.1f ohm T^-1** (statistical)"
      % (r1["m_H"], r1["s_m_H"]))
    A("- k = **%.2f +/- %.2f ohm**, the field-independent longitudinal leak"
      % (r1["k"], r1["s_k"]))
    A("- chi2/dof = %.2f on %d degrees of freedom" % (r1["chi2"] / r1["dof"], r1["dof"]))
    A("- RMS residual on the slopes = %.2f ohm" % rms(r1["resid"]))
    A("")
    A("The chi2/dof of this second-stage fit is above 1 because the weights carry only")
    A("the statistical error on each slope. The field values themselves come from one")
    A("shared calibration polynomial, so their errors are correlated and are handled in")
    A("the budget of section 8 rather than in the weights. The same remark applies to")
    A("R3 below.")
    A("")
    A("The slopes are linear in B. That is objective 2 for the field dependence. Taken")
    A("with the previous table, V_H is shown to be a single linear function of the")
    A("product I*B rather than of I and B separately.")
    A("")
    A("### %s.3 The three reductions of the whole array (objective 6)" % sec)
    A("")
    A("| Reduction | Model | R_H,2D (ohm T^-1) | sigma stat (ohm T^-1) | chi2 | dof |"
      " chi2/dof |")
    A("|---|---|---|---|---|---|---|")
    A("| R1, slope of slopes | s_j = m_H B_j + k | %.1f | %.1f | %.1f | %d | %.2f |"
      % (r1["m_H"], r1["s_m_H"], r1["chi2"], r1["dof"], r1["chi2"] / r1["dof"]))
    A("| R2, naive global line | V_H = m_H (IB) + c | %.1f | %.1f | %.1f | %d | %.2f |"
      % (r2["m_H"], r2["s_m_H"], r2["chi2"], r2["dof"], r2["chi2"] / r2["dof"]))
    A("| R3, global with offset term | V_H = m_H (IB) + k I + c | **%.1f** | **%.1f** |"
      " %.1f | %d | **%.2f** |"
      % (r3["m_H"], r3["s_m_H"], r3["chi2"], r3["dof"], r3["chi2"] / r3["dof"]))
    A("")
    A("**R3 is the value adopted for the report.** The reasoning:")
    A("")
    A("- R2 forces the contact-misalignment voltage, which is proportional to I and")
    A("  independent of B, into a single constant. That voltage is correlated with the")
    A("  abscissa I*B, so it biases the slope. R2 returns %.1f ohm T^-1 against the R3"
      % r2["m_H"])
    A("  value of %.1f ohm T^-1, a difference of %+.2f per cent, and its chi2/dof of"
      % (r3["m_H"], pct_diff(r2["m_H"], r3["m_H"])))
    A("  %.1f says plainly that the two-parameter model does not describe the data."
      % (r2["chi2"] / r2["dof"]))
    A("- R1 and R3 agree to %+.2f per cent. R3 is preferred only because it uses every"
      % pct_diff(r1["m_H"], r3["m_H"]))
    A("  point in a single weighted fit rather than fitting slopes to slopes.")
    A("- Fitted misalignment term k = **%.2f +/- %.2f ohm** (R3). The value used to"
      % (r3["k"], r3["s_k"]))
    A("  generate the data was %.2f ohm, from a contact displacement of %.2f um on a"
      % (k_offset_true[T], DELTA_MISALIGN * 1e6))
    A("  %.0f um wide bar. The fitted k sits %.1f standard errors away from that"
      % (W_ACTIVE * 1e6, abs(r3["k"] - k_offset_true[T]) / r3["s_k"]))
    A("  value. The reason is that k is strongly correlated with m_H in this fit: the")
    A("  correlation coefficient from the covariance matrix is %+.2f. Any error in the"
      % r3["corr_mk"])
    A("  field calibration therefore leaks into k as well as into m_H, and the quoted")
    A("  standard error on k, which is statistical only, understates how well k is")
    A("  really known. k should be read as a nuisance parameter that soaks up a")
    A("  field-independent voltage, not as a precision measurement of the contact")
    A("  geometry.")
    A("- Fitted constant term c = **%+.4f +/- %.4f mV** (R3). The value used to"
      % (r3["c"] * 1e3, r3["s_c"] * 1e3))
    A("  generate the data was %+.3f mV." % (V_OFFSET_STATIC[T] * 1e3))
    A("- Weighted residual RMS: R2 %.4f mV, R3 %.4f mV."
      % (rms(r2["resid"]) * 1e3, rms(r3["resid"]) * 1e3))
    A("")
    A("A naive alternative, computing R_H,2D = V_H/(I*B) point by point and averaging,")
    A("is worse still. It weights the smallest I*B points, which have the worst signal")
    A("to noise, equally with the best, and the constant offset contributes c/(I*B) to")
    A("every ratio, a contribution that diverges as I*B goes to zero. For these data")
    A("that unweighted average gives %.1f ohm T^-1, which is %+.1f per cent away from"
      % (float(np.mean(d["V"] / d["x"])),
         pct_diff(float(np.mean(d["V"] / d["x"])), r3["m_H"])))
    A("the R3 value.")
    A("")
    A("Residuals were examined against I*B and against B. With R2 they show clear")
    A("structure grouped by field setting; with R3 that structure is largely gone. The")
    A("residual scatter that remains in R3 is set by the magnet calibration, which is a")
    A("correlated systematic and is not in the weights, so a chi2/dof of exactly one is")
    A("not expected. See the lower panel of `figures/fig_hall_VH_vs_IB.pdf` for the")
    A("300 K case.")
    A("")
    A("Value used to generate the data, R_H,2D at %s = **%.1f ohm T^-1**. The adopted"
      % (tlab.split()[0] + " K" if T == "300K" else "the cryogenic temperature",
         p["R_H2D"]))
    A("R3 value differs from it by **%+.2f per cent**, which is inside the total"
      % pct_diff(r3["m_H"], p["R_H2D"]))
    A("uncertainty of %.1f per cent quoted in section 8." % (FIN[T]["f_n"] * 100))
    A("")
    A("---")
    A("")
    return lines_b


hall_blocks["300K"] = build_hall_block("300K", "300 K (room temperature)", "2")
hall_blocks["77K"] = build_hall_block(
    "77K", "%.0f K (liquid nitrogen cooling)" % T_mean, "5"
)
lines.extend(hall_blocks["300K"])

# ---- 3. longitudinal -------------------------------------------------------------
A("## 3. Longitudinal resistance and resistance per square")
A("")
A("Data files: `data/hall_VL_300K.csv`, `data/hall_VL_77K.csv` (%d rows each)."
  % sup_volts.size)
A("Figures: `figures/fig_hall_VL_vs_I.pdf`, `figures/fig_hall_77K_comparison.pdf`.")
A("")
A("Measured between contacts 1 and 4 with the electromagnet off (emVolts = [0]).")
A("Measuring at zero field removes magnetoresistance, removes any Hall cross-talk into")
A("the longitudinal pair, and keeps the magnet calibration out of the R_sq error")
A("budget entirely. The four-terminal arrangement also means the contact resistances")
A("and the two 150 kohm protection resistors do not enter the result at all, because")
A("the current is measured directly rather than inferred from the supply voltage.")
A("")
A("The active region is %.0f um long between the voltage probes and %.0f um wide, so"
  % (L_ACTIVE * 1e6, W_ACTIVE * 1e6))
A("it is exactly **%.0f squares** and R_sq = R_L / %.0f." % (N_SQUARES, N_SQUARES))
A("")
A("| Quantity | 300 K | %s | Unit |" % T_LABEL["77K"])
A("|---|---|---|---|")
A("| Current range | %.2f to %.2f | %.2f to %.2f | uA |"
  % (lred["300K"]["I"].min() * 1e6, lred["300K"]["I"].max() * 1e6,
     lred["77K"]["I"].min() * 1e6, lred["77K"]["I"].max() * 1e6))
A("| V_L range | %.2f to %.2f | %.2f to %.2f | mV |"
  % (lred["300K"]["V"].min() * 1e3, lred["300K"]["V"].max() * 1e3,
     lred["77K"]["V"].min() * 1e3, lred["77K"]["V"].max() * 1e3))
A("| Fitted slope m_L = R_L | %.1f +/- %.1f | %.2f +/- %.2f | ohm |"
  % (lred["300K"]["m_L"], lred["300K"]["s_m_L"],
     lred["77K"]["m_L"], lred["77K"]["s_m_L"]))
A("| Fitted intercept c_L | %+.4f +/- %.4f | %+.4f +/- %.4f | mV |"
  % (lred["300K"]["c_L"] * 1e3, lred["300K"]["s_c_L"] * 1e3,
     lred["77K"]["c_L"] * 1e3, lred["77K"]["s_c_L"] * 1e3))
A("| chi2/dof | %.2f | %.2f | |"
  % (lred["300K"]["chi2"] / lred["300K"]["dof"],
     lred["77K"]["chi2"] / lred["77K"]["dof"]))
A("| **R_sq = m_L / 3** | **%.1f +/- %.1f (stat)** | **%.2f +/- %.2f (stat)** |"
  " ohm/square |"
  % (f300["R_sq"], f300["s_R_sq_stat"], f77["R_sq"], f77["s_R_sq_stat"]))
A("| R_sq, total uncertainty | +/- %.1f | +/- %.2f | ohm/square |"
  % (f300["s_R_sq"], f77["s_R_sq"]))
A("| Value used to generate R_L | %.1f | %.2f | ohm |"
  % (TRUE["300K"]["R_L"], TRUE["77K"]["R_L"]))
A("| Value used to generate R_sq | %.1f | %.2f | ohm/square |"
  % (TRUE["300K"]["R_sq"], TRUE["77K"]["R_sq"]))
A("| Difference from that value | %+.2f | %+.2f | per cent |"
  % (pct_diff(f300["R_sq"], TRUE["300K"]["R_sq"]),
     pct_diff(f77["R_sq"], TRUE["77K"]["R_sq"])))
A("")
A("R_sq falls by a factor of **%.1f** on cooling, while R_H,2D changes by only"
  % (f300["R_sq"] / f77["R_sq"]))
A("**%.1f per cent**. That asymmetry is the fingerprint of modulation doping: the"
  % abs(pct_diff(f77["R_H2D"], f300["R_H2D"])))
A("carrier density is set by charge transfer equilibrium across the heterojunction,")
A("which involves energies of tens to hundreds of meV and so barely moves between")
A("300 K and 77 K, while the mobility is set by scattering rates and the dominant")
A("channel at 300 K, polar optical phonon scattering, freezes out on cooling.")
A("")
A("A further consequence worth noting. At %s the Hall voltage at the top of the sweep,"
  % T_LABEL["77K"])
A("%.1f mV, is **larger** than the longitudinal voltage, %.1f mV, because mu*B = %.1f"
  % (red["77K"]["V"].max() * 1e3, lred["77K"]["V"].max() * 1e3, f77["omct"]))
A("exceeds l/w = 3. At 300 K the reverse holds: V_H reaches only %.1f mV against a V_L"
  % (red["300K"]["V"].max() * 1e3))
A("of %.1f mV. That inversion is a direct visual signature of the high mobility."
  % (lred["300K"]["V"].max() * 1e3))
A("")
A("---")
A("")

# ---- 4. angle --------------------------------------------------------------------
A("## 4. Angular dependence of the Hall voltage")
A("")
A("Data file: `data/hall_angle_300K.csv` (%d rows, 0 to 360 degrees in 15 degree"
  % n_ang)
A("steps). Figure: `figures/fig_hall_angle.pdf`.")
A("")
A("Taken at 300 K with the magnet supply fixed at %.1f V (calibrated B = %.4f T) and"
  % (em_volts[3], B_ang))
A("the bar supply fixed at %.1f V, giving a mean sample current of %.2f uA."
  % (sup_ang, float(np.mean(i_ang_meas)) * 1e6))
A("")
A("Only the component of B normal to the 2DEG plane produces a Hall voltage, because")
A("the carriers are confined and cannot acquire an out-of-plane velocity. The fit")
A("model is the script's own cosine, extended by a constant offset term:")
A("")
A("    V_H(theta) = V_H0 * cos(theta - theta_0) + C")
A("")
A("| Parameter | Fitted value | Standard error | Unit | Value used to generate |")
A("|---|---|---|---|---|")
A("| V_H0 | %.4f | %.4f | mV | %.4f |"
  % (ang_p[0] * 1e3, ang_err[0] * 1e3,
     TRUE["300K"]["R_H2D"] * i_ang_true * B_ang_true * 1e3))
A("| theta_0 | %.2f | %.2f | degrees | %.1f |" % (ang_p[1], ang_err[1], THETA0_TRUE))
A("| C | %.4f | %.4f | mV | %.4f |"
  % (ang_p[2] * 1e3, ang_err[2] * 1e3,
     (k_off_300 * i_ang_true + V_OFFSET_STATIC["300K"]) * 1e3))
A("")
A("- chi2 = %.2f on %d degrees of freedom, chi2/dof = **%.2f**"
  % (ang_chi2, ang_dof, ang_chi2 / ang_dof))
A("- RMS residual = **%.4f mV**, against an assumed meter scatter of %.3f mV"
  % (rms(ang_res) * 1e3, SIG_VH_FLOOR * 1e3))
A("- The amplitude predicted from the independently fitted R_H,2D at 300 K and the")
A("  measured I and B is **%.4f mV**, against a fitted **%.4f mV**, a difference of"
  % (f300["R_H2D"] * float(np.mean(i_ang_meas)) * B_ang * 1e3, ang_p[0] * 1e3))
A("  **%+.2f per cent**. The angular scan is therefore an independent confirmation of"
  % pct_diff(ang_p[0], f300["R_H2D"] * float(np.mean(i_ang_meas)) * B_ang))
A("  the Hall coefficient obtained from the current and field sweeps.")
A("- theta_0 = %.2f degrees is a mounting misalignment between the zero of the"
  % ang_p[1])
A("  rotation stage and the true normal orientation. It is not a property of the")
A("  sample. Because the cosine is flat near its maximum, a measurement taken at a")
A("  nominal zero without correcting for it under-reads the normal field component by")
A("  only %.2f per cent." % (100.0 * (1.0 - np.cos(np.radians(ang_p[1])))))
A("- C = %.4f mV is field-independent, and it is consistent in size with the offset"
  % (ang_p[2] * 1e3))
A("  found independently in the global Hall fit, which predicts k*I + c = %.4f mV at"
  % ((red["300K"]["R3"]["k"] * float(np.mean(i_ang_meas))
      + red["300K"]["R3"]["c"]) * 1e3))
A("  this current (k*I = %.4f mV plus c = %.4f mV). The two differ by %+.1f per cent."
  % (red["300K"]["R3"]["k"] * float(np.mean(i_ang_meas)) * 1e3,
     red["300K"]["R3"]["c"] * 1e3,
     pct_diff(ang_p[2], red["300K"]["R3"]["k"] * float(np.mean(i_ang_meas))
              + red["300K"]["R3"]["c"])))
A("  That is the level of agreement to expect given how strongly k correlates with")
A("  the slope in the global fit.")
A("- The cosine term reverses sign at theta_0 + 180 degrees, but the measured voltage")
A("  does not reverse with equal magnitude, because the offset C does not change sign:")
A("  the readings at 0 and 180 degrees are +%.4f mV and %.4f mV, and their sum, %.4f mV,"
  % (float(vh_ang[0]) * 1e3, float(vh_ang[12]) * 1e3,
     float(vh_ang[0] + vh_ang[12]) * 1e3))
A("  is 2C. Half the DIFFERENCE of two opposite angles removes the offset and returns")
A("  the pure Hall term; half their SUM returns the offset. That is the same parity")
A("  argument as reversing the field, and the reversal itself confirms the signal is a")
A("  genuine Hall voltage rather than an instrumental artefact.")
A("")
A("---")
A("")

lines.extend(hall_blocks["77K"])

# ---- 6. 77 K thermometry ---------------------------------------------------------
A("## 6. Temperature during the cryogenic run")
A("")
A("Data file: `data/hall_thermocouple_77K.csv` (%d type K readings taken on the sample"
  % tc_readings.size)
A("mount through the run).")
A("")
A("| Quantity | Value | Unit |")
A("|---|---|---|")
A("| Mean reading | %.2f | K |" % T_mean)
A("| Spread of readings (standard deviation) | %.2f | K |" % T_sd)
A("| Standard error on the mean | %.2f | K |" % T_sem)
A("| Type K accuracy floor below 200 K | 2.5 | K |")
A("| **Quoted temperature** | **%.0f +/- %.0f** | K |" % (T_mean, T_quoted_err))
A("| Value used to generate the readings | %.1f | K |" % T_TRUE_77)
A("| Liquid nitrogen boiling point at 1 atm | 77.4 | K |")
A("")
A("The thermocouple bead is on the mount rather than on the 2DEG itself, and the")
A("sample sits above the liquid in a pumped dewar, so a reading a few kelvin above the")
A("bath temperature is expected. The sensitivity of a type K junction falls from about")
A("41 uV/K at 300 K to roughly 16 uV/K at 77 K, which is why the low-temperature")
A("reading is the less precise one.")
A("")
A("Neither n_sq nor mu is calculated from T, so a temperature error does not propagate")
A("into the results. It matters only for the comparison with literature values quoted")
A("at a nominal 77 K.")
A("")
A("---")
A("")

# ---- 7. headline results ---------------------------------------------------------
A("## 7. Headline results")
A("")
A("R_H,2D comes from the R3 global weighted fit; R_sq comes from the zero-field")
A("longitudinal fit. Formulae used:")
A("")
A("    n_sq = 1 / (e * R_H,2D)")
A("    R_sq = m_L / 3")
A("    mu   = R_H,2D / R_sq = 3 * m_H / m_L")
A("")
A("with e = %.9e C." % E_CHARGE)
A("")
A("| Quantity | 300 K | %s | Unit |" % T_LABEL["77K"])
A("|---|---|---|---|")
A("| R_H,2D, statistical only | %.1f +/- %.1f | %.1f +/- %.1f | ohm T^-1 |"
  % (f300["R_H2D"], f300["s_R_H2D_stat"], f77["R_H2D"], f77["s_R_H2D_stat"]))
A("| R_H,2D, total | %.1f +/- %.1f | %.1f +/- %.1f | ohm T^-1 |"
  % (f300["R_H2D"], f300["s_R_H2D"], f77["R_H2D"], f77["s_R_H2D"]))
A("| R_L | %.1f +/- %.1f | %.2f +/- %.2f | ohm |"
  % (f300["R_L"], f300["s_mL"], f77["R_L"], f77["s_mL"]))
A("| R_sq, total | %.1f +/- %.1f | %.2f +/- %.2f | ohm/square |"
  % (f300["R_sq"], f300["s_R_sq"], f77["R_sq"], f77["s_R_sq"]))
A("| **n_sq** | **%.3f +/- %.3f** | **%.3f +/- %.3f** | 10^15 m^-2 |"
  % (f300["n_sq"] / 1e15, f300["s_n"] / 1e15, f77["n_sq"] / 1e15, f77["s_n"] / 1e15))
A("| n_sq, statistical only | %.3f +/- %.3f | %.3f +/- %.3f | 10^15 m^-2 |"
  % (f300["n_sq"] / 1e15, f300["s_n_stat"] / 1e15,
     f77["n_sq"] / 1e15, f77["s_n_stat"] / 1e15))
A("| n_sq in cm^-2 | %.3f +/- %.3f | %.3f +/- %.3f | 10^11 cm^-2 |"
  % (f300["n_sq"] / 1e15, f300["s_n"] / 1e15, f77["n_sq"] / 1e15, f77["s_n"] / 1e15))
A("| **mu** | **%.4f +/- %.4f** | **%.2f +/- %.2f** | m^2 V^-1 s^-1 |"
  % (f300["mu"], f300["s_mu"], f77["mu"], f77["s_mu"]))
A("| mu, statistical only | %.4f +/- %.4f | %.2f +/- %.2f | m^2 V^-1 s^-1 |"
  % (f300["mu"], f300["s_mu_stat"], f77["mu"], f77["s_mu_stat"]))
A("| mu in cm^2 V^-1 s^-1 | %.0f +/- %.0f | %.0f +/- %.0f | cm^2 V^-1 s^-1 |"
  % (f300["mu"] * 1e4, f300["s_mu"] * 1e4, f77["mu"] * 1e4, f77["s_mu"] * 1e4))
A("| n_sq * mu, free of magnet error | %.4f +/- %.4f | %.3f +/- %.3f |"
  " 10^15 V^-1 s^-1 (equivalently C^-1 ohm^-1) |"
  % (f300["nmu"] / 1e15, f300["s_nmu"] / 1e15,
     f77["nmu"] / 1e15, f77["s_nmu"] / 1e15))
A("")
A("Note: 10^15 m^-2 is exactly 10^11 cm^-2, so the two density rows carry the same")
A("digits. Both are given because the literature quotes cm^-2.")
A("")
A("### 7.1 Change on cooling")
A("")
A("| Quantity | 300 K | %s | Change |" % T_LABEL["77K"])
A("|---|---|---|---|")
A("| n_sq (10^15 m^-2) | %.3f | %.3f | %+.1f per cent |"
  % (f300["n_sq"] / 1e15, f77["n_sq"] / 1e15, pct_diff(f77["n_sq"], f300["n_sq"])))
A("| mu (m^2 V^-1 s^-1) | %.4f | %.2f | factor %.1f increase |"
  % (f300["mu"], f77["mu"], f77["mu"] / f300["mu"]))
A("| R_sq (ohm/square) | %.1f | %.2f | factor %.1f decrease |"
  % (f300["R_sq"], f77["R_sq"], f300["R_sq"] / f77["R_sq"]))
A("| R_H,2D (ohm T^-1) | %.1f | %.1f | %+.1f per cent |"
  % (f300["R_H2D"], f77["R_H2D"], pct_diff(f77["R_H2D"], f300["R_H2D"])))
A("")
A("Uncertainty on the mobility ratio, combining the two statistical uncertainties in")
A("quadrature and noting that the magnet and geometry systematics largely cancel in")
A("the ratio: **%.2f +/- %.2f**."
  % (f77["mu"] / f300["mu"],
     (f77["mu"] / f300["mu"]) * quad(f300["f_mH"], f300["f_mL"],
                                     f77["f_mH"], f77["f_mL"])))
A("")
A("---")
A("")

# ---- 8. uncertainty budget -------------------------------------------------------
A("## 8. Uncertainty budget")
A("")
A("| Contribution | Size | Affects |")
A("|---|---|---|")
A("| Fit standard error on m_H, 300 K | %.2f per cent | n_sq, mu |"
  % (f300["f_mH"] * 100))
A("| Fit standard error on m_H, %s | %.2f per cent | n_sq, mu |"
  % (T_LABEL["77K"], f77["f_mH"] * 100))
A("| Fit standard error on m_L, 300 K | %.2f per cent | R_sq, mu |"
  % (f300["f_mL"] * 100))
A("| Fit standard error on m_L, %s | %.2f per cent | R_sq, mu |"
  % (T_LABEL["77K"], f77["f_mL"] * 100))
A("| Hall probe absolute accuracy and positioning in the gap | %.1f per cent |"
  " n_sq, mu |" % (SYST_B_FRAC * 100))
A("| Field floor term, %.0f mT over a mean field of %.3f T | %.1f per cent | n_sq, mu |"
  % (SYST_B_FLOOR * 1e3, float(np.mean(B_SET)),
     100 * SYST_B_FLOOR / float(np.mean(B_SET))))
A("| Statistical uncertainty of the calibration polynomial | %.1f per cent | n_sq, mu |"
  % (100 * float(np.mean(SIG_B_SET / B_SET))))
A("| **Combined magnet systematic on B** | **%.1f per cent** | n_sq, mu, not R_sq |"
  % (F_BSYST * 100))
A("| Geometry, l/w from etch bias and mesa edge depletion | %.1f per cent |"
  " R_sq, mu, not n_sq |" % (SYST_GEOM_FRAC * 100))
A("| **Combined on n_sq, 300 K** | **%.1f per cent** | |" % (f300["f_n"] * 100))
A("| **Combined on n_sq, %s** | **%.1f per cent** | |"
  % (T_LABEL["77K"], f77["f_n"] * 100))
A("| **Combined on R_sq, 300 K** | **%.1f per cent** | |" % (f300["f_Rsq"] * 100))
A("| **Combined on R_sq, %s** | **%.1f per cent** | |"
  % (T_LABEL["77K"], f77["f_Rsq"] * 100))
A("| **Combined on mu, 300 K** | **%.1f per cent** | |" % (f300["f_mu"] * 100))
A("| **Combined on mu, %s** | **%.1f per cent** | |"
  % (T_LABEL["77K"], f77["f_mu"] * 100))
A("")
A("Three points worth making in the discussion:")
A("")
A("- The magnet calibration dominates the error on n_sq and on mu. Improving the")
A("  voltmeter would not help; only a better field measurement would.")
A("- R_sq carries no magnet uncertainty at all, because it was measured at zero field.")
A("- The product n_sq * mu = 1 / (e * R_sq) is fixed by the zero-field longitudinal")
A("  measurement alone and is immune to the magnet calibration. It is known to")
A("  %.1f per cent at 300 K and %.1f per cent at %s, far better than either factor"
  % (f300["s_nmu"] / f300["nmu"] * 100, f77["s_nmu"] / f77["nmu"] * 100,
     T_LABEL["77K"]))
A("  separately. n_sq and mu share the factor m_H, so their errors are correlated and")
A("  the magnet systematic pushes them in opposite directions.")
A("")
A("Not included in the budget above, and stated here as known limitations:")
A("")
A("- The Hall scattering factor is taken as exactly 1. At %s the gas is partly"
  % T_LABEL["77K"])
A("  degenerate (T/T_F = %.2f) and that is a reasonable assumption. At 300 K the gas"
  % (T_mean / f77["TF"]))
A("  is not degenerate (T/T_F = %.2f), so a Hall scattering factor a few per cent"
  % (300.0 / f300["TF"]))
A("  above unity is possible. Since R_H,2D = r_H / (n_sq e), that would make the true")
A("  300 K density a few per cent HIGHER than quoted, not lower, which is the opposite")
A("  direction to parallel conduction below.")
A("- Any parallel conduction in the doped AlGaAs layer at 300 K would raise the")
A("  apparent density and lower the apparent mobility. It freezes out on cooling.")
A("")
A("---")
A("")

# ---- 9. comparison tables --------------------------------------------------------
A("## 9. Comparison with the generating values and with the literature")
A("")
A("### 9.1 Against the values used to generate the data")
A("")
A("This table exists so that the honesty of the pipeline can be checked. The")
A("generating values were chosen first, from the literature; the fitted values are")
A("what the analysis returned without ever seeing them.")
A("")
A("| Quantity | Generating value | Fitted | Difference | Quoted uncertainty | Unit |")
A("|---|---|---|---|---|---|")
rows_gt = [
    ("n_sq, 300 K", TRUE["300K"]["n_sq"] / 1e15, f300["n_sq"] / 1e15,
     f300["f_n"] * 100, "%.3f", "10^15 m^-2"),
    ("n_sq, %s" % T_LABEL["77K"], TRUE["77K"]["n_sq"] / 1e15, f77["n_sq"] / 1e15,
     f77["f_n"] * 100, "%.3f", "10^15 m^-2"),
    ("mu, 300 K", TRUE["300K"]["mu"], f300["mu"], f300["f_mu"] * 100,
     "%.4f", "m^2 V^-1 s^-1"),
    ("mu, %s" % T_LABEL["77K"], TRUE["77K"]["mu"], f77["mu"], f77["f_mu"] * 100,
     "%.2f", "m^2 V^-1 s^-1"),
    ("R_sq, 300 K", TRUE["300K"]["R_sq"], f300["R_sq"], f300["f_Rsq"] * 100,
     "%.1f", "ohm/square"),
    ("R_sq, %s" % T_LABEL["77K"], TRUE["77K"]["R_sq"], f77["R_sq"],
     f77["f_Rsq"] * 100, "%.2f", "ohm/square"),
    ("R_H,2D, 300 K", TRUE["300K"]["R_H2D"], f300["R_H2D"], f300["f_n"] * 100,
     "%.1f", "ohm T^-1"),
    ("R_H,2D, %s" % T_LABEL["77K"], TRUE["77K"]["R_H2D"], f77["R_H2D"],
     f77["f_n"] * 100, "%.1f", "ohm T^-1"),
]
for name, gt, fit, unc, fmt, unit in rows_gt:
    A(("| %s | " + fmt + " | " + fmt + " | %+.2f per cent | +/- %.1f per cent | %s |")
      % (name, gt, fit, pct_diff(fit, gt), unc, unit))
A("")
A("Every fitted value sits inside its own quoted uncertainty, which is the behaviour a")
A("correctly propagated error budget should show.")
A("")
A("### 9.2 Against the literature")
A("")
A("| Quantity | This work (representative) | Literature | Source | Difference |")
A("|---|---|---|---|---|")
A("| n_sq, 300 K (10^11 cm^-2) | %.2f +/- %.2f | 5.6 | Seliuta et al.,"
  " arXiv:0711.0474, structure 2DEG-B | %+.1f per cent |"
  % (f300["n_sq"] / 1e15, f300["s_n"] / 1e15,
     pct_diff(f300["n_sq"], LIT["n_sq_300"])))
A("| mu, 300 K (cm^2 V^-1 s^-1) | %.0f +/- %.0f | 8000 | same | %+.1f per cent |"
  % (f300["mu"] * 1e4, f300["s_mu"] * 1e4, pct_diff(f300["mu"], LIT["mu_300"])))
A("| n_sq, %s (10^11 cm^-2) | %.2f +/- %.2f | 4.3 | same | %+.1f per cent |"
  % (T_LABEL["77K"], f77["n_sq"] / 1e15, f77["s_n"] / 1e15,
     pct_diff(f77["n_sq"], LIT["n_sq_77"])))
A("| mu, %s (cm^2 V^-1 s^-1) | %.0f +/- %.0f | 116000 | same | %+.1f per cent |"
  % (T_LABEL["77K"], f77["mu"] * 1e4, f77["s_mu"] * 1e4,
     pct_diff(f77["mu"], LIT["mu_77"])))
A("| mu at 77 K (cm^2 V^-1 s^-1) | %.0f +/- %.0f | 117000 | Stormer, Dingle, Gossard,"
  " Wiegmann and Sturge, Solid State Commun. 29, 705 (1979) | %+.1f per cent |"
  % (f77["mu"] * 1e4, f77["s_mu"] * 1e4, pct_diff(f77["mu"], LIT["mu_77_stormer"])))
A("| R_sq, 300 K (ohm/square) | %.0f +/- %.0f | about 2000 | typical GaAs/AlGaAs 2DEG"
  " at 300 K, HEMT characterisation literature | %+.1f per cent |"
  % (f300["R_sq"], f300["s_R_sq"], pct_diff(f300["R_sq"], LIT["R_sq_300"])))
A("| mu, 300 K, against bulk undoped GaAs (m^2 V^-1 s^-1) | %.3f +/- %.3f | 0.85 |"
  " Ioffe NSM database, electrical properties of GaAs | %+.1f per cent |"
  % (f300["mu"], f300["s_mu"], pct_diff(f300["mu"], LIT["mu_bulk_300"])))
A("| mu(cold) / mu(300 K) | %.1f | 14.5 | Seliuta et al., 2DEG-B | %+.1f per cent |"
  % (f77["mu"] / f300["mu"], pct_diff(f77["mu"] / f300["mu"], 14.5)))
A("| Electron effective mass used | 0.067 m_0 | 0.067 m_0 | Vurgaftman, Meyer and"
  " Ram-Mohan, J. Appl. Phys. 89, 5815 (2001) | assumed, not fitted |")
A("| Sample geometry l/w | 3.0 squares | 3.0 squares | the departmental script |"
  " given, not fitted |")
A("")
A("---")
A("")

# ---- 10. secondary quantities ----------------------------------------------------
A("## 10. Secondary quantities derived from the fitted results")
A("")
A("Computed with m* = 0.067 m_0 = %.4e kg. These need no extra measurement." % M_STAR)
A("")
A("| Quantity | Formula | 300 K | %s | Unit |" % T_LABEL["77K"])
A("|---|---|---|---|---|")
A("| Momentum relaxation time tau | mu m* / e | %.3f | %.2f | ps |"
  % (f300["tau"] * 1e12, f77["tau"] * 1e12))
A("| Fermi wavevector k_F | sqrt(2 pi n_sq) | %.3f | %.3f | 10^8 m^-1 |"
  % (f300["kF"] / 1e8, f77["kF"] / 1e8))
A("| Fermi velocity v_F | hbar k_F / m* | %.3f | %.3f | 10^5 m s^-1 |"
  % (f300["vF"] / 1e5, f77["vF"] / 1e5))
A("| Mean free path | v_F tau | %.3f | %.3f | um |"
  % (f300["mfp"] * 1e6, f77["mfp"] * 1e6))
A("| Fermi energy above the subband edge | pi hbar^2 n_sq / m* | %.2f | %.2f | meV |"
  % (f300["EF"] / E_CHARGE * 1e3, f77["EF"] / E_CHARGE * 1e3))
A("| Fermi temperature T_F | E_F / k_B | %.0f | %.0f | K |" % (f300["TF"], f77["TF"]))
A("| omega_c tau = mu B at B = %.3f T | mu B | %.3f | %.2f | dimensionless |"
  % (B_SET.max(), f300["omct"], f77["omct"]))
A("| 2D density of states including spin | m* / (pi hbar^2) | %.3f | %.3f |"
  " 10^13 eV^-1 cm^-2 |"
  % (M_STAR / (np.pi * HBAR**2) * E_CHARGE * 1e-4 / 1e13,
     M_STAR / (np.pi * HBAR**2) * E_CHARGE * 1e-4 / 1e13))
A("")
A("Both mean free paths are far shorter than the %.0f um probe separation, so"
  % (L_ACTIVE * 1e6))
A("transport is diffusive and the Drude picture applies. At %s, T / T_F = %.2f, so the"
  % (T_LABEL["77K"], T_mean / f77["TF"]))
A("gas is partly degenerate and taking the Hall scattering factor as 1 is reasonable.")
A("At 300 K, T / T_F = %.2f, so the gas is not degenerate at all. The Fermi quantities"
  % (300.0 / f300["TF"]))
A("in the table above are zero-temperature expressions, so their 300 K entries are")
A("indicative only: the thermal velocity sqrt(2 k_B T / m*) = %.2fe5 m/s gives a"
  % (np.sqrt(2 * KB * 300.0 / M_STAR) / 1e5))
A("room-temperature mean free path of %.3f um, which does not change the conclusion."
  % (np.sqrt(2 * KB * 300.0 / M_STAR) * f300["tau"] * 1e6))
A("")
A("The 2D density of states is independent of energy, which is the single most")
A("characteristic signature of two-dimensionality.")
A("")
A("Landau level spacing at the peak field: hbar*omega_c = %.3f meV, against"
  % (HBAR * E_CHARGE * B_SET.max() / M_STAR / E_CHARGE * 1e3))
A("k_B T = %.2f meV at %s and %.2f meV at 300 K. Thermal broadening exceeds the level"
  % (KB * round(T_mean) / E_CHARGE * 1e3, T_LABEL["77K"],
     KB * 300.0 / E_CHARGE * 1e3))
A("spacing by a factor of %.0f even at the cryogenic temperature, so no Shubnikov-de"
  % (KB * round(T_mean) / (HBAR * E_CHARGE * B_SET.max() / M_STAR)))
A("Haas oscillations and no quantum Hall plateaux are expected. No continuous field")
A("sweep of the longitudinal resistance was taken in any case, and R_sq was measured")
A("only at zero field, so these data could not have shown such oscillations.")
A("")
A("---")
A("")

# ---- 11. files and checks --------------------------------------------------------
A("## 11. Files produced")
A("")
A("| File | Contents |")
A("|---|---|")
A("| `data/hall_magnet_calibration.csv` | electromagnet calibration, rising and"
  " falling branches |")
A("| `data/hall_VH_300K.csv` | Hall voltage against current at five fields, 300 K |")
A("| `data/hall_VL_300K.csv` | longitudinal voltage against current at zero field,"
  " 300 K |")
A("| `data/hall_angle_300K.csv` | Hall voltage against angle over 360 degrees, 300 K |")
A("| `data/hall_VH_77K.csv` | Hall voltage against current at five fields, cryogenic |")
A("| `data/hall_VL_77K.csv` | longitudinal voltage against current at zero field,"
  " cryogenic |")
A("| `data/hall_thermocouple_77K.csv` | type K thermocouple readings during the cold"
  " run |")
A("| `figures/fig_hall_magnet_calibration.pdf` | both hysteresis branches, cubic fit,"
  " residuals, branch gap |")
A("| `figures/fig_hall_VH_vs_I.pdf` | family of V_H against I lines, one per field,"
  " 300 K |")
A("| `figures/fig_hall_VH_vs_IB.pdf` | global collapse onto one line with residual"
  " panel, 300 K |")
A("| `figures/fig_hall_VL_vs_I.pdf` | V_L against I at zero field with residual panel,"
  " 300 K |")
A("| `figures/fig_hall_angle.pdf` | cosine fit to the angular scan with residual"
  " panel |")
A("| `figures/fig_hall_77K_comparison.pdf` | 300 K and cryogenic Hall and longitudinal"
  " data side by side |")
A("")
A("Note on two of the figures, so the captions describe them correctly:")
A("")
A("- In `fig_hall_VH_vs_IB.pdf` the upper panel shows the raw Hall voltages against")
A("  I*B with the naive two-parameter global fit drawn through them. The lower panel")
A("  shows the residuals of that fit, coloured by field setting, together with the")
A("  residuals of the three-parameter fit as black crosses. The structure grouped by")
A("  field setting in the first set, and its absence in the second, is the visual")
A("  evidence for the contact-misalignment term.")
A("- In the left panel of `fig_hall_77K_comparison.pdf` the fitted misalignment term")
A("  k*I and the constant c have been subtracted from the Hall voltages, so the")
A("  remaining signal is the pure Hall term and the fitted lines pass through the")
A("  origin. The axis is labelled accordingly. The right panel shows raw longitudinal")
A("  voltages, with the cryogenic data repeated in an expanded inset.")
A("")
A("Every figure caption in the report must carry the words \"representative data\".")
A("")
A("## 12. Sanity checks run by the generator")
A("")
A("| Check | Result | Detail |")
A("|---|---|---|")
for label, ok, detail in checks:
    A("| %s | %s | %s |" % (label, "pass" if ok else "FAIL", detail))
A("")
A("Overall: **%s**." % ("all checks passed" if all_ok else "SOME CHECKS FAILED"))
A("")
A("---")
A("")
A("*Representative data. Not a record of measured results.*")

with open(os.path.join(DATA_DIR, "results_hall.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n    wrote", os.path.join(DATA_DIR, "results_hall.md"))


# ==================================================================================
# SUMMARY TABLE
# ==================================================================================

print("\n" + "=" * 78)
print("SUMMARY OF FITTED RESULTS (representative data, not measured results)")
print("=" * 78)
print("%-34s %19s %19s" % ("Quantity", "300 K", T_LABEL["77K"]))
print("-" * 78)
rows = [
    ("R_H,2D  (ohm / T)", "%.1f +/- %.1f" % (f300["R_H2D"], f300["s_R_H2D"]),
     "%.1f +/- %.1f" % (f77["R_H2D"], f77["s_R_H2D"])),
    ("R_L     (ohm)", "%.0f +/- %.0f" % (f300["R_L"], f300["s_mL"]),
     "%.1f +/- %.1f" % (f77["R_L"], f77["s_mL"])),
    ("R_sq    (ohm / square)", "%.0f +/- %.0f" % (f300["R_sq"], f300["s_R_sq"]),
     "%.1f +/- %.1f" % (f77["R_sq"], f77["s_R_sq"])),
    ("n_sq    (1e15 m^-2)", "%.3f +/- %.3f" % (f300["n_sq"] / 1e15, f300["s_n"] / 1e15),
     "%.3f +/- %.3f" % (f77["n_sq"] / 1e15, f77["s_n"] / 1e15)),
    ("mu      (m^2 / V s)", "%.4f +/- %.4f" % (f300["mu"], f300["s_mu"]),
     "%.2f +/- %.2f" % (f77["mu"], f77["s_mu"])),
    ("mu      (cm^2 / V s)", "%.0f +/- %.0f" % (f300["mu"] * 1e4, f300["s_mu"] * 1e4),
     "%.0f +/- %.0f" % (f77["mu"] * 1e4, f77["s_mu"] * 1e4)),
    ("tau     (ps)", "%.3f" % (f300["tau"] * 1e12), "%.2f" % (f77["tau"] * 1e12)),
    ("mean free path (um)", "%.3f" % (f300["mfp"] * 1e6), "%.3f" % (f77["mfp"] * 1e6)),
    ("mu*B at peak field", "%.3f" % f300["omct"], "%.2f" % f77["omct"]),
    ("V_H range (mV)", "%.2f to %.2f" % (red["300K"]["V"].min() * 1e3,
                                         red["300K"]["V"].max() * 1e3),
     "%.2f to %.2f" % (red["77K"]["V"].min() * 1e3, red["77K"]["V"].max() * 1e3)),
    ("V_L range (mV)", "%.1f to %.1f" % (lred["300K"]["V"].min() * 1e3,
                                         lred["300K"]["V"].max() * 1e3),
     "%.2f to %.2f" % (lred["77K"]["V"].min() * 1e3, lred["77K"]["V"].max() * 1e3)),
    ("I range (uA)", "%.2f to %.2f" % (red["300K"]["I"].min() * 1e6,
                                       red["300K"]["I"].max() * 1e6),
     "%.2f to %.2f" % (red["77K"]["I"].min() * 1e6, red["77K"]["I"].max() * 1e6)),
]
for a, b, c in rows:
    print("%-34s %19s %19s" % (a, b, c))
print("-" * 78)
print("%-34s %19s %19s" % ("value used to generate n_sq",
                           "%.3f" % (TRUE["300K"]["n_sq"] / 1e15),
                           "%.3f" % (TRUE["77K"]["n_sq"] / 1e15)))
print("%-34s %19s %19s" % ("value used to generate mu",
                           "%.4f" % TRUE["300K"]["mu"], "%.2f" % TRUE["77K"]["mu"]))
print("%-34s %18.2f%% %18.2f%%" % ("n_sq difference",
                                   pct_diff(f300["n_sq"], TRUE["300K"]["n_sq"]),
                                   pct_diff(f77["n_sq"], TRUE["77K"]["n_sq"])))
print("%-34s %18.2f%% %18.2f%%" % ("mu difference",
                                   pct_diff(f300["mu"], TRUE["300K"]["mu"]),
                                   pct_diff(f77["mu"], TRUE["77K"]["mu"])))
print("%-34s %18.1f%% %18.1f%%" % ("quoted total uncertainty on n_sq",
                                   f300["f_n"] * 100, f77["f_n"] * 100))
print("-" * 78)
print("mobility ratio mu(cold) / mu(300 K) = %.1f   (literature 10 to 40)"
      % (f77["mu"] / f300["mu"]))
print("sheet density fall on cooling       = %.1f per cent   (literature 10 to 25)"
      % (-pct_diff(f77["n_sq"], f300["n_sq"])))
print("R_sq fall on cooling                = factor %.1f" % (f300["R_sq"] / f77["R_sq"]))
print("quoted cryogenic temperature        = %.0f +/- %.0f K" % (T_mean, T_quoted_err))
print("=" * 78)
print("REPRESENTATIVE DATA. NOT A RECORD OF MEASURED RESULTS.")
print("=" * 78)
