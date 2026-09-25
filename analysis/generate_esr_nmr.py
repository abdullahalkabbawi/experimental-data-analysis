"""
generate_esr_nmr.py

Representative data generator and analysis for the Physics 3H laboratory report
"Electron Spin Resonance and Nuclear Magnetic Resonance" (PHYS 4009, University of
Glasgow, script by I. MacLaren, 6 October 2023).

IMPORTANT, READ THIS FIRST
--------------------------
Every number produced by this script is REPRESENTATIVE DATA. It is simulated from
physically accepted ground-truth parameters plus realistic instrument scatter. It is
NOT a record of measured results. The report that consumes these files must say so in
the box after the abstract, in every results table caption, in every data figure
caption and in the page footer.

REPRODUCIBILITY
---------------
The random number generator is seeded with the fixed integer SEED = 20231006 (the date
carried on the departmental script, used only as a memorable constant). Running this
file again reproduces every number, every CSV and every figure exactly. Nothing in this
file reads the system clock or the current date.

METHOD
------
Data are built FORWARD from ground truth:
  1. choose physically accepted underlying parameters (g factors, coil constants,
     magnet B(I) curve);
  2. compute the ideal observable that the apparatus would show;
  3. add realistic instrument scatter, display quantisation and the systematic
     offsets the experiment genuinely has.
The data are then ANALYSED exactly as a student would, by least squares with
uncertainties taken from the fit covariance matrices and propagated into the final
physical quantities. The values quoted in the report are the values these fits return,
which differ slightly from the ground truth. That difference is the point.

No systematic scale bias is injected into the field axis. The 1 per cent tolerance on
the magnetometer reference magnet is carried through as a stated systematic
UNCERTAINTY, which is how a student who does not know its value must treat it.

Outputs
-------
CSV     -> reports/data/
Figures -> reports/figures/   (vector PDF)
Summary -> reports/data/results_esr_nmr.md
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

SEED = 20231006
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(REPORTS, "data")
FIG_DIR = os.path.join(REPORTS, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BANNER = "# Representative data. Simulated from accepted physics, not a record of measured results."

# Okabe-Ito colourblind-safe palette
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_PINK = "#CC79A7"
C_AMBER = "#E69F00"
C_SKY = "#56B4E9"
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
        "errorbar.capsize": 2,
        "pdf.fonttype": 42,
        "text.usetex": False,
    }
)

# --------------------------------------------------------------------------------------
# Physical constants (CODATA 2022)
# --------------------------------------------------------------------------------------

H_PLANCK = 6.62607015e-34          # J s, exact by SI definition
HBAR = 1.054571817e-34             # J s
MU_B = 9.2740100657e-24            # J / T, Bohr magneton
MU_N = 5.0507837393e-27            # J / T, nuclear magneton
MU_0 = 4.0 * np.pi * 1e-7          # V s / (A m)
K_B = 1.380649e-23                 # J / K

G_ELECTRON_FREE = 2.00231930436092     # free electron g factor
G_DPPH_LIT = 2.0036                    # accepted g for solid DPPH
G_PROTON_LIT = 5.5856946893            # accepted 1H nuclear g factor
G_FLUORINE_LIT = 5.257736              # accepted 19F nuclear g factor

GAMMA2PI_H = G_PROTON_LIT * MU_N / H_PLANCK / 1e6      # MHz / T  -> 42.577478
GAMMA2PI_F = G_FLUORINE_LIT * MU_N / H_PLANCK / 1e6    # MHz / T  -> 40.077583

# Helmholtz nominal (script equation 6) with n = 320 turns, r = 6.8 cm
N_TURNS_NOMINAL = 320
R_COIL = 0.068                                          # m
S_THEORY_mT_per_A = (0.8 ** 1.5) * MU_0 * N_TURNS_NOMINAL / R_COIL * 1e3   # mT / A

# --------------------------------------------------------------------------------------
# Ground truth chosen for the simulation
# --------------------------------------------------------------------------------------

# ESR ground truth
S_TRUE_mT_per_A = 4.200        # real Helmholtz coil constant (coils do not have exactly 320 turns)
G_DPPH_TRUE = G_DPPH_LIT       # true DPPH g factor used to make the resonance currents
B_AMBIENT_TRUE_mT = 0.030      # residual ambient axial field at the sample after LOW FIELD zeroing
FWHM_TRUE_mT = 0.240           # true DPPH absorption line FWHM

# NMR ground truth: rising branch of the iron-cored electromagnet, valid 3.0 to 4.0 A only.
# Chosen so that B(3.00 A) = 0.3800 T, B(3.50 A) = 0.4195 T and B(4.00 A) = 0.4550 T, which
# brackets the "about 0.43 T" working point, and so that dB/dI falls by about 19 per cent
# across the range as the iron behind the pole pieces saturates.
A0_TRUE, A1_TRUE, A2_TRUE = 0.0590, 0.1310, -0.00800     # tesla, tesla/A, tesla/A^2
HYST_GAP_AT_3A_T = 6.0e-3                                 # branch separation at 3.00 A

# Instrument scatter
SIG_B_MAGNETOMETER_mT = 0.010     # ESR Hall magnetometer, LOW FIELD range
SIG_I_ESR = {"large": 0.005, "medium": 0.004, "small": 0.006}   # A, setting the resonance
N_REPEATS_ESR = 5
SIG_B_TESLAMETER_T = 0.5e-3       # NMR teslameter, reading scatter (display resolution 0.1 mT)
SIG_I_NMR_A = 0.005               # NMR coil current, ammeter reading reproducibility
SIG_NU = {"glycerine": 0.020, "ptfe": 0.050, "polystyrene": 0.120, "polythene": 0.050}  # MHz

# Correlated systematic carried as an uncertainty, never injected as a bias
PROBE_SCALE_SYSTEMATIC = 0.010    # 1 per cent, from the 112 mT +/- 1 per cent reference magnet


# --------------------------------------------------------------------------------------
# Small fitting helpers
# --------------------------------------------------------------------------------------

def weighted_linear_fit(x, y, sigma_y):
    """Weighted least squares y = m x + c. Returns m, c, cov, chi2, chi2_reduced."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)
    w = 1.0 / sigma_y ** 2
    Sw = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)
    delta = Sw * Sxx - Sx * Sx
    m = (Sw * Sxy - Sx * Sy) / delta
    c = (Sxx * Sy - Sx * Sxy) / delta
    var_m = Sw / delta
    var_c = Sxx / delta
    cov_mc = -Sx / delta
    cov = np.array([[var_m, cov_mc], [cov_mc, var_c]])
    resid = y - (m * x + c)
    chi2 = float(np.sum(w * resid ** 2))
    dof = len(x) - 2
    return m, c, cov, chi2, chi2 / dof


def poly_fit_weighted(x, y, sigma_y, order):
    """Weighted polynomial fit, coefficients returned in ASCENDING power order."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma_y = np.asarray(sigma_y, float)
    A = np.vstack([x ** k for k in range(order + 1)]).T
    W = np.diag(1.0 / sigma_y ** 2)
    ATWA = A.T @ W @ A
    cov = np.linalg.inv(ATWA)
    coef = cov @ (A.T @ W @ y)
    resid = y - A @ coef
    chi2 = float(resid @ W @ resid)
    dof = len(x) - (order + 1)
    return coef, cov, chi2, chi2 / dof


def poly_value(coef, x):
    scalar = np.isscalar(x) or (np.ndim(x) == 0)
    xa = np.asarray(x, float)
    out = sum(c * xa ** k for k, c in enumerate(coef))
    return float(out) if scalar else out


def poly_sigma(coef, cov, x):
    """1 sigma on the polynomial value at x, including all coefficient covariances."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        J = np.array([xi ** k for k in range(len(coef))])
        out[i] = np.sqrt(J @ cov @ J)
    return out


def effective_variance_fit(x, y, sigma_x, sigma_y, n_iter=8):
    """
    Straight-line fit with uncertainties on both axes, by the effective variance method:
        chi2 = sum (y - m x - c)^2 / (sigma_y^2 + m^2 sigma_x^2)
    The slope appears in the weights, so the fit is iterated to convergence.
    """
    m, c, cov, chi2, chi2r = weighted_linear_fit(x, y, sigma_y)
    for _ in range(n_iter):
        sig_eff = np.sqrt(np.asarray(sigma_y, float) ** 2 + (m ** 2) * np.asarray(sigma_x, float) ** 2)
        m_new, c, cov, chi2, chi2r = weighted_linear_fit(x, y, sig_eff)
        if abs(m_new - m) < 1e-12 * abs(m_new):
            m = m_new
            break
        m = m_new
    sig_eff = np.sqrt(np.asarray(sigma_y, float) ** 2 + (m ** 2) * np.asarray(sigma_x, float) ** 2)
    return m, c, cov, chi2, chi2r, sig_eff


def write_csv(path, banner, header, rows, fmt):
    lines = [banner, header]
    for row in rows:
        lines.append(fmt(row))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# ======================================================================================
# PART A (ESR).  Helmholtz coil calibration, script Appendix 4
# ======================================================================================

esr_I_cal = np.round(np.arange(0.1, 1.2001, 0.1), 3)                  # 0.1 to 1.2 A
b_off_mag_true = rng.uniform(-0.020, 0.020)                           # residual LOW FIELD zero offset, mT
B_cal_ideal = S_TRUE_mT_per_A * esr_I_cal + b_off_mag_true
B_cal_meas = B_cal_ideal + rng.normal(0.0, SIG_B_MAGNETOMETER_mT, esr_I_cal.size)
B_cal_meas = np.round(B_cal_meas, 2)                                  # 0.01 mT display resolution

sig_B_cal = np.full(esr_I_cal.size, SIG_B_MAGNETOMETER_mT)
s_fit, b_fit, cov_cal, chi2_cal, chi2r_cal = weighted_linear_fit(esr_I_cal, B_cal_meas, sig_B_cal)
sig_s_fit = np.sqrt(cov_cal[0, 0])
sig_b_fit = np.sqrt(cov_cal[1, 1])
resid_cal = B_cal_meas - (s_fit * esr_I_cal + b_fit)

# Total uncertainty on the coil constant: statistical plus the 1 per cent reference magnet
sig_s_total = s_fit * np.hypot(sig_s_fit / s_fit, PROBE_SCALE_SYSTEMATIC)

write_csv(
    os.path.join(DATA_DIR, "esr_helmholtz_calibration.csv"),
    BANNER,
    "coil_current_A,magnetic_field_mT,field_uncertainty_mT",
    zip(esr_I_cal, B_cal_meas, sig_B_cal),
    lambda r: f"{r[0]:.1f},{r[1]:.2f},{r[2]:.3f}",
)

# ======================================================================================
# PART B (ESR).  Resonance frequency against field, three plug-in coils
# ======================================================================================

esr_blocks = [
    ("large", np.array([15.0, 20.0, 25.0, 30.0])),
    ("medium", np.arange(35.0, 75.001, 5.0)),
    ("small", np.arange(80.0, 130.001, 5.0)),
]

esr_rows = []
for coil, nus in esr_blocks:
    sig_I = SIG_I_ESR[coil]
    for nu in nus:
        # forward from ground truth: field the DPPH needs, minus the ambient already present
        B_res_true_mT = H_PLANCK * nu * 1e6 / (G_DPPH_TRUE * MU_B) * 1e3
        I_true = (B_res_true_mT - B_AMBIENT_TRUE_mT) / S_TRUE_mT_per_A
        reps = np.round(I_true + rng.normal(0.0, sig_I, N_REPEATS_ESR), 3)   # 1 mA panel resolution
        I_mean = float(np.mean(reps))
        I_sd = float(np.std(reps, ddof=1))
        I_sem = I_sd / np.sqrt(N_REPEATS_ESR)
        esr_rows.append(
            {
                "coil": coil,
                "nu_MHz": float(nu),
                "reps": reps,
                "I_mean_A": I_mean,
                "I_sd_A": I_sd,
                "I_sem_A": I_sem,
            }
        )

esr_nu = np.array([r["nu_MHz"] for r in esr_rows])
esr_Imean = np.array([r["I_mean_A"] for r in esr_rows])
esr_Isem = np.array([r["I_sem_A"] for r in esr_rows])
esr_coilname = [r["coil"] for r in esr_rows]

# Convert current to field with the student's OWN calibration (script insists on this)
esr_B0 = s_fit * esr_Imean + b_fit
esr_sigB0 = s_fit * esr_Isem          # statistical part only; the calibration scale is correlated

for r, B, sB in zip(esr_rows, esr_B0, esr_sigB0):
    r["B0_mT"] = B
    r["sigB0_mT"] = sB

write_csv(
    os.path.join(DATA_DIR, "esr_resonance_frequency.csv"),
    BANNER,
    "plug_in_coil,frequency_MHz,I_run1_A,I_run2_A,I_run3_A,I_run4_A,I_run5_A,"
    "I_mean_A,I_sd_A,I_standard_error_A,B0_mT,B0_standard_error_mT",
    esr_rows,
    lambda r: (
        f"{r['coil']},{r['nu_MHz']:.1f},"
        + ",".join(f"{v:.3f}" for v in r["reps"])
        + f",{r['I_mean_A']:.4f},{r['I_sd_A']:.4f},{r['I_sem_A']:.4f},"
        f"{r['B0_mT']:.4f},{r['sigB0_mT']:.4f}"
    ),
)

# Single global straight-line fit across all three coils. Both axes carry uncertainty: the
# field through the repeat scatter of the resonance current, and the frequency through the
# 0.1 MHz readout resolution and short-term drift of the oscillator. The effective-variance
# method therefore applies, with the field error projected onto the frequency axis through
# the gradient.
SIG_NU_ESR_MHz = 0.1                       # MHz, readout resolution and short-term drift
NU_ESR_ERR = np.full(esr_nu.size, SIG_NU_ESR_MHz)
m_esr, c_esr, cov_esr, chi2_esr, chi2r_esr, sig_eff_esr = effective_variance_fit(
    esr_B0, esr_nu, esr_sigB0, NU_ESR_ERR
)
sig_m_esr_stat = np.sqrt(cov_esr[0, 0])
sig_c_esr = np.sqrt(cov_esr[1, 1])
resid_esr = esr_nu - (m_esr * esr_B0 + c_esr)

# Cross-check route: fit nu against current, then divide by the calibration slope
k_esr, nu0_esr, cov_k, chi2_k, chi2r_k, _ = effective_variance_fit(
    esr_Imean, esr_nu, esr_Isem, NU_ESR_ERR
)
sig_k_esr = np.sqrt(cov_k[0, 0])
m_esr_routeB = k_esr / s_fit
sig_m_esr_routeB = m_esr_routeB * np.hypot(sig_k_esr / k_esr, sig_s_fit / s_fit)

# g factor for DPPH.  m is in MHz/mT = 1e9 Hz/T
g_dpph = H_PLANCK * m_esr * 1e9 / MU_B
sig_g_stat = g_dpph * np.hypot(sig_m_esr_stat / m_esr, sig_s_fit / s_fit)
sig_g_syst = g_dpph * PROBE_SCALE_SYSTEMATIC
sig_g_total = np.hypot(sig_g_stat, sig_g_syst)

# Intercept diagnostic. The physics requires nu = (g mu_B / h) B_true with no constant term,
# so a fitted intercept c means the field at the sample exceeded the calibrated field by c/m.
B_extra_at_sample_mT = c_esr / m_esr
sig_B_extra_at_sample_mT = abs(B_extra_at_sample_mT) * np.hypot(
    sig_c_esr / abs(c_esr), sig_m_esr_stat / m_esr
)
# What the analysis chain should recover: the real ambient field at the sample, plus whatever
# zero offset the calibration fit has already absorbed into its own intercept.
B_extra_expected_mT = B_AMBIENT_TRUE_mT - b_fit

# ======================================================================================
# PART C (ESR).  Half-width at 50 MHz on the medium coil, script equation (7)
# ======================================================================================

# Resonance current at 50 MHz taken from the data set generated above
row50 = next(r for r in esr_rows if abs(r["nu_MHz"] - 50.0) < 1e-9)
I_res_50 = row50["I_mean_A"]
B_res_50_mT = row50["B0_mT"]

U_mod_V = 5.00                      # 0.5 V/cm over the full 10.0 cm X axis
sig_U_mod_V = 0.10                  # 2 per cent, scope gain plus judging "exactly 10 cm"
I_mod_true_A = 0.100                # RMS modulation current set to span the screen
I_mod_meas_A = round(float(I_mod_true_A + rng.normal(0.0, 0.003)), 3)
sig_I_mod_A = 0.005

# Forward: the true FWHM fixes the true half-width current, hence the true voltage width
delta_I_true_A = FWHM_TRUE_mT / S_TRUE_mT_per_A
delta_U_true_V = U_mod_V * delta_I_true_A / (2.0 * np.sqrt(2.0) * I_mod_true_A)
delta_U_meas_V = round(float(delta_U_true_V + rng.normal(0.0, 0.040)), 2)
sig_delta_U_V = 0.05                # two edges read to about 1 mm on the 0.2 V/cm scale

delta_I_A = 2.0 * np.sqrt(2.0) * I_mod_meas_A * delta_U_meas_V / U_mod_V
delta_B0_mT = s_fit * delta_I_A
rel_delta_B0 = np.sqrt(
    (sig_delta_U_V / delta_U_meas_V) ** 2
    + (sig_I_mod_A / I_mod_meas_A) ** 2
    + (sig_U_mod_V / U_mod_V) ** 2
    + (sig_s_total / s_fit) ** 2
)
sig_delta_B0_mT = delta_B0_mT * rel_delta_B0

# Relaxation time from the width, Lorentzian homogeneous line
gamma_dpph = g_dpph * MU_B / HBAR                   # rad / s / T
T2_s = 2.0 / (gamma_dpph * delta_B0_mT * 1e-3)
sig_T2_s = T2_s * rel_delta_B0

# Peak-to-peak derivative width, for comparison with the EPR literature
delta_Bpp_mT = delta_B0_mT / np.sqrt(3.0)
sig_delta_Bpp_mT = sig_delta_B0_mT / np.sqrt(3.0)

# Peak-to-peak modulation field, for the self-consistency check
B_mod_pp_mT = s_fit * 2.0 * np.sqrt(2.0) * I_mod_meas_A

write_csv(
    os.path.join(DATA_DIR, "esr_linewidth.csv"),
    BANNER,
    "quantity,value,uncertainty,unit,note",
    [
        ("RF frequency", 50.0, 0.1, "MHz", "set value on the medium plug-in coil"),
        ("resonance current I=", I_res_50, row50["I_sem_A"], "A", "mean of 5 repeats"),
        ("resonance field B0", B_res_50_mT, row50["sigB0_mT"], "mT", "from the Helmholtz calibration"),
        ("modulation current I_mod (RMS)", I_mod_meas_A, sig_I_mod_A, "A", "trace spans the 10.0 cm X axis"),
        ("full modulation voltage U_mod", U_mod_V, sig_U_mod_V, "V", "0.5 V/cm times 10.0 cm"),
        ("half-height width delta_U", delta_U_meas_V, sig_delta_U_V, "V", "read on the expanded 0.2 V/cm X scale"),
        ("half-width current delta_I", delta_I_A, delta_I_A * np.sqrt(
            (sig_delta_U_V / delta_U_meas_V) ** 2
            + (sig_I_mod_A / I_mod_meas_A) ** 2
            + (sig_U_mod_V / U_mod_V) ** 2), "A", "script equation (7)"),
        ("line width delta_B0 (FWHM)", delta_B0_mT, sig_delta_B0_mT, "mT", "converted with the measured coil constant"),
        ("peak-to-peak derivative width", delta_Bpp_mT, sig_delta_Bpp_mT, "mT", "FWHM divided by sqrt(3) for literature comparison"),
        ("transverse relaxation time T2", T2_s * 1e9, sig_T2_s * 1e9, "ns", "Lorentzian line so T2 = 2 / (gamma delta_B0)"),
    ],
    lambda r: f"{r[0]},{r[1]:.4g},{r[2]:.3g},{r[3]},{r[4]}",
)

# Representative absorption line shape for the construction figure
B_axis = np.linspace(B_res_50_mT - 0.60, B_res_50_mT + 0.60, 501)
Gamma_mT = delta_B0_mT / 2.0
line_clean = 1.0 / (1.0 + ((B_axis - B_res_50_mT) / Gamma_mT) ** 2)
baseline_tilt = 0.02 * (B_axis - B_res_50_mT) / 0.60
line_noisy = line_clean + baseline_tilt + rng.normal(0.0, 0.015, B_axis.size)

write_csv(
    os.path.join(DATA_DIR, "esr_lineshape.csv"),
    BANNER,
    "magnetic_field_mT,esr_signal_arbitrary_units",
    zip(B_axis, line_noisy),
    lambda r: f"{r[0]:.4f},{r[1]:.4f}",
)

# ======================================================================================
# PART D (NMR).  Electromagnet calibration with hysteresis
# ======================================================================================

def B_rise_true_T(I):
    return A0_TRUE + A1_TRUE * np.asarray(I, float) + A2_TRUE * np.asarray(I, float) ** 2


def hyst_gap_true_T(I):
    return HYST_GAP_AT_3A_T * (4.00 - np.asarray(I, float)) / 1.00


nmr_I_nom = np.round(np.arange(3.00, 4.0001, 0.10), 2)

# Rising sweep 3.0 to 4.0 A.  The dial reads I_nom; the current actually delivered scatters.
I_act_rise = nmr_I_nom + rng.normal(0.0, SIG_I_NMR_A, nmr_I_nom.size)
B_rise_meas = B_rise_true_T(I_act_rise) + rng.normal(0.0, SIG_B_TESLAMETER_T, nmr_I_nom.size)
B_rise_meas = np.round(B_rise_meas, 4)          # 0.1 mT teslameter display resolution

# Falling sweep 4.0 to 3.0 A, on the upper branch of the minor loop
I_act_fall = nmr_I_nom + rng.normal(0.0, SIG_I_NMR_A, nmr_I_nom.size)
B_fall_meas = B_rise_true_T(I_act_fall) + hyst_gap_true_T(I_act_fall) + rng.normal(0.0, SIG_B_TESLAMETER_T, nmr_I_nom.size)
B_fall_meas = np.round(B_fall_meas, 4)
# A minor loop must start the downward branch from the exact state the upward branch ended in,
# so the falling sweep is seeded at the rising endpoint rather than given its own reading noise
# there. Without this the two branches can cross at the turning point, which is unphysical.
B_fall_meas[-1] = B_rise_meas[-1]

write_csv(
    os.path.join(DATA_DIR, "nmr_magnet_calibration.csv"),
    BANNER,
    "sweep_direction,coil_current_A,magnetic_field_T,field_uncertainty_T,current_uncertainty_A",
    [("rising", i, b) for i, b in zip(nmr_I_nom, B_rise_meas)]
    + [("falling", i, b) for i, b in zip(nmr_I_nom[::-1], B_fall_meas[::-1])],
    lambda r: f"{r[0]},{r[1]:.2f},{r[2]:.4f},{SIG_B_TESLAMETER_T:.4f},{SIG_I_NMR_A:.3f}",
)

# Fit the RISING branch only, as the script demands.  Effective variance handles the
# fact that the recorded current carries its own uncertainty.
sig_B_cal_nmr = np.full(nmr_I_nom.size, SIG_B_TESLAMETER_T)
coef_lin, cov_lin, chi2_lin, chi2r_lin = poly_fit_weighted(nmr_I_nom, B_rise_meas, sig_B_cal_nmr, 1)
coef_q0, _, _, _ = poly_fit_weighted(nmr_I_nom, B_rise_meas, sig_B_cal_nmr, 2)
dBdI_guess = coef_q0[1] + 2.0 * coef_q0[2] * nmr_I_nom
sig_B_eff_nmr = np.sqrt(sig_B_cal_nmr ** 2 + (dBdI_guess * SIG_I_NMR_A) ** 2)

coef_quad, cov_quad, chi2_quad, chi2r_quad = poly_fit_weighted(nmr_I_nom, B_rise_meas, sig_B_eff_nmr, 2)
coef_lin_e, cov_lin_e, chi2_lin_e, chi2r_lin_e = poly_fit_weighted(nmr_I_nom, B_rise_meas, sig_B_eff_nmr, 1)
coef_cub, cov_cub, chi2_cub, chi2r_cub = poly_fit_weighted(nmr_I_nom, B_rise_meas, sig_B_eff_nmr, 3)

a0, a1, a2 = coef_quad
sig_a0, sig_a1, sig_a2 = np.sqrt(np.diag(cov_quad))
resid_quad = B_rise_meas - poly_value(coef_quad, nmr_I_nom)
resid_lin = B_rise_meas - poly_value(coef_lin_e, nmr_I_nom)
cubic_coeff = coef_cub[3]
sig_cubic_coeff = np.sqrt(cov_cub[3, 3])

# Is the quadratic term justified, and is a cubic term justified on top of it?
quad_significance = abs(a2) / sig_a2                  # in units of its own uncertainty
delta_chi2_lin_quad = chi2_lin_e - chi2_quad          # improvement from adding the I^2 term
F_lin_quad = delta_chi2_lin_quad / (chi2_quad / (nmr_I_nom.size - 3))
delta_chi2_quad_cub = chi2_quad - chi2_cub
F_quad_cub = delta_chi2_quad_cub / (chi2_cub / (nmr_I_nom.size - 4))
curvature_amplitude_mT = abs(a2) * (1.00 ** 2) / 8.0 * 1e3   # peak straight-line residual, mT

hyst_gap_meas = B_fall_meas - B_rise_meas
dBdI_at = lambda I: a1 + 2.0 * a2 * I
sig_dBdI_at = lambda I: np.sqrt(cov_quad[1, 1] + 4.0 * I * I * cov_quad[2, 2] + 4.0 * I * cov_quad[1, 2])


def field_from_current(I):
    """Field and its 1 sigma from the rising-branch calibration, including the current error."""
    I = np.atleast_1d(np.asarray(I, float))
    B = poly_value(coef_quad, I)
    sig_fit = poly_sigma(coef_quad, cov_quad, I)
    sig_set = np.abs(dBdI_at(I)) * SIG_I_NMR_A
    return B, np.sqrt(sig_fit ** 2 + sig_set ** 2)


# ======================================================================================
# PART E and F (NMR).  Proton in glycerine and fluorine in PTFE
# ======================================================================================

def make_nmr_run(gamma2pi_MHz_per_T, sigma_nu_MHz):
    I_act = nmr_I_nom + rng.normal(0.0, SIG_I_NMR_A, nmr_I_nom.size)
    nu_true = gamma2pi_MHz_per_T * B_rise_true_T(I_act)
    nu_meas = np.round(nu_true + rng.normal(0.0, sigma_nu_MHz, nmr_I_nom.size), 2)
    return nu_meas


nu_glycerine = make_nmr_run(GAMMA2PI_H, SIG_NU["glycerine"])
nu_ptfe = make_nmr_run(GAMMA2PI_F, SIG_NU["ptfe"])

B_from_cal, sigB_from_cal = field_from_current(nmr_I_nom)

write_csv(
    os.path.join(DATA_DIR, "nmr_resonance_frequency.csv"),
    BANNER,
    "sample,nucleus,coil_current_A,resonance_frequency_MHz,frequency_uncertainty_MHz,"
    "magnetic_field_T,field_uncertainty_T",
    [("glycerine", "1H", i, n, SIG_NU["glycerine"], b, sb)
     for i, n, b, sb in zip(nmr_I_nom, nu_glycerine, B_from_cal, sigB_from_cal)]
    + [("PTFE", "19F", i, n, SIG_NU["ptfe"], b, sb)
       for i, n, b, sb in zip(nmr_I_nom, nu_ptfe, B_from_cal, sigB_from_cal)],
    lambda r: f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.3f},{r[5]:.4f},{r[6]:.4f}",
)


def analyse_nmr(nu_meas, sigma_nu_MHz, g_lit, label):
    """Route A (effective variance), Route B (chain rule) and a Monte Carlo cross-check."""
    sig_nu = np.full(nu_meas.size, sigma_nu_MHz)

    # Route A
    m, c, cov, chi2, chi2r, sig_eff = effective_variance_fit(B_from_cal, nu_meas, sigB_from_cal, sig_nu)
    sig_m_stat = np.sqrt(cov[0, 0])
    sig_c = np.sqrt(cov[1, 1])
    resid = nu_meas - (m * B_from_cal + c)

    # Route B, gradients evaluated at the mid-range current
    I_mid = 3.50
    coef_nu, cov_nu, chi2_nu, chi2r_nu = poly_fit_weighted(nmr_I_nom, nu_meas, sig_nu, 2)
    dnudI = coef_nu[1] + 2.0 * coef_nu[2] * I_mid
    sig_dnudI = np.sqrt(cov_nu[1, 1] + 4.0 * I_mid ** 2 * cov_nu[2, 2] + 4.0 * I_mid * cov_nu[1, 2])
    dBdI = dBdI_at(I_mid)
    sig_dBdI = sig_dBdI_at(I_mid)
    m_B = dnudI / dBdI
    sig_m_B = abs(m_B) * np.hypot(sig_dnudI / dnudI, sig_dBdI / dBdI)

    # Monte Carlo: resample the calibration coefficients and the frequencies together
    n_mc = 5000
    coef_draws = rng.multivariate_normal(coef_quad, cov_quad, size=n_mc)
    slopes = np.empty(n_mc)
    for j in range(n_mc):
        Bj = poly_value(coef_draws[j], nmr_I_nom)
        nuj = nu_meas + rng.normal(0.0, sigma_nu_MHz, nu_meas.size)
        slopes[j] = np.polyfit(Bj, nuj, 1)[0]
    m_mc = float(np.mean(slopes))
    sig_m_mc = float(np.std(slopes, ddof=1))

    # g factor and its budget
    g = H_PLANCK * m * 1e6 / MU_N
    rel_stat = sig_m_stat / m
    rel_total = np.hypot(rel_stat, PROBE_SCALE_SYSTEMATIC)
    sig_g_st = g * rel_stat
    sig_g_sy = g * PROBE_SCALE_SYSTEMATIC
    sig_g_tot = g * rel_total
    pct_diff = 100.0 * (g - g_lit) / g_lit
    z = abs(g - g_lit) / sig_g_tot

    # Extra field present at the sample beyond the calibrated value, from the intercept
    B_off_T = c / m
    sig_B_off_T = abs(B_off_T) * np.hypot(sig_c / abs(c), sig_m_stat / m) if c != 0 else 0.0

    return dict(
        label=label, m=m, sig_m_stat=sig_m_stat, c=c, sig_c=sig_c, chi2=chi2, chi2r=chi2r,
        resid=resid, sig_eff=sig_eff, m_routeB=m_B, sig_m_routeB=sig_m_B,
        m_mc=m_mc, sig_m_mc=sig_m_mc, g=g, sig_g_stat=sig_g_st, sig_g_syst=sig_g_sy,
        sig_g_total=sig_g_tot, g_lit=g_lit, pct_diff=pct_diff, z=z,
        rel_stat=rel_stat, rel_total=rel_total,
        B_offset_T=B_off_T, sig_B_offset_T=sig_B_off_T, nu=nu_meas, sig_nu=sig_nu,
        dnudI=dnudI, sig_dnudI=sig_dnudI,
    )


res_H = analyse_nmr(nu_glycerine, SIG_NU["glycerine"], G_PROTON_LIT, "1H in glycerine")
res_F = analyse_nmr(nu_ptfe, SIG_NU["ptfe"], G_FLUORINE_LIT, "19F in PTFE")

ratio_meas = res_F["m"] / res_H["m"]
sig_ratio = ratio_meas * np.hypot(res_F["sig_m_stat"] / res_F["m"], res_H["sig_m_stat"] / res_H["m"])
ratio_lit = GAMMA2PI_F / GAMMA2PI_H

# ======================================================================================
# PART G (NMR).  Four-sample comparison at one fixed field
# ======================================================================================

I_FIXED = 3.50
B_fixed_true = float(B_rise_true_T(I_FIXED))
B_fixed_cal, sigB_fixed_cal = field_from_current(I_FIXED)
B_fixed_cal = float(B_fixed_cal[0])
sigB_fixed_cal = float(sigB_fixed_cal[0])

# Line widths are chosen to be consistent with the transient (wiggle) counts reported for the
# same samples: the number of countable wiggles is N = gamma (dB/dt) (T_2*)^2 / (4 pi), and
# T_2* = 1 / (pi x line width in frequency units). The narrow-component column is the width that
# actually controls the transient. For glycerine and polystyrene it is the whole line; for
# polythene the tabulated width is the composite of a narrow amorphous component and a broad
# crystalline one, and only the narrow component survives long enough to produce wiggles.
#   name, formula, nucleus, gamma/2pi, sigma_nu, width_mT, wiggles, rel_height, narrow_mT
samples = [
    ("glycerine", "C3H5(OH)3", "1H", GAMMA2PI_H, SIG_NU["glycerine"], 0.010, "4 to 6 and strong", 1.00, 0.010),
    ("polystyrene", "(C8H8)n", "1H", GAMMA2PI_H, SIG_NU["polystyrene"], 0.500, "none", 0.06, 0.500),
    ("polythene", "(C2H4)n", "1H (deduced)", GAMMA2PI_H, SIG_NU["polythene"], 0.150, "1 to 2 and weak", 0.20, 0.020),
    ("PTFE", "(C2F4)n", "19F", GAMMA2PI_F, SIG_NU["ptfe"], 0.150, "none", 0.28, 0.150),
]

# NMR field modulation, needed for the transient calculation
F_MOD_NMR_Hz = 28.0
B_MOD_NMR_PEAK_T = 2.4e-3
DBDT_NMR_T_per_s = 2.0 * np.pi * F_MOD_NMR_Hz * B_MOD_NMR_PEAK_T


def wiggle_count(width_mT, gamma2pi_MHz_per_T):
    """N = gamma (dB/dt) (T_2*)^2 / (4 pi), with T_2* = 1 / (pi x line width in Hz)."""
    width_Hz = gamma2pi_MHz_per_T * 1e6 * width_mT * 1e-3
    T2star = 1.0 / (np.pi * width_Hz)
    gamma = 2.0 * np.pi * gamma2pi_MHz_per_T * 1e6
    return gamma * DBDT_NMR_T_per_s * T2star ** 2 / (4.0 * np.pi), T2star


sample_rows = []
for name, formula, nucleus, gam, signu, width_mT, wiggles, rel_height, narrow_mT in samples:
    nu_true = gam * B_fixed_true
    nu_meas = round(float(nu_true + rng.normal(0.0, signu)), 2)
    n_wig, T2star = wiggle_count(narrow_mT, gam)
    sample_rows.append(
        dict(name=name, formula=formula, nucleus=nucleus, nu=nu_meas, sig=signu,
             fwhm_mT=width_mT, wiggles=wiggles, rel_height=rel_height, width_mT=width_mT,
             narrow_mT=narrow_mT, n_wiggles=n_wig, T2star_s=T2star,
             pred_H=GAMMA2PI_H * B_fixed_cal, pred_F=GAMMA2PI_F * B_fixed_cal)
    )

pred_H_fixed = GAMMA2PI_H * B_fixed_cal
pred_F_fixed = GAMMA2PI_F * B_fixed_cal
split_MHz = pred_H_fixed - pred_F_fixed

write_csv(
    os.path.join(DATA_DIR, "nmr_sample_comparison.csv"),
    BANNER,
    "sample,formula,resonant_nucleus,coil_current_A,magnetic_field_T,resonance_frequency_MHz,"
    "frequency_uncertainty_MHz,observed_line_width_mT,wiggles_observed,"
    "predicted_1H_frequency_MHz,predicted_19F_frequency_MHz",
    sample_rows,
    lambda r: (
        f"{r['name']},{r['formula']},{r['nucleus']},{I_FIXED:.2f},{B_fixed_cal:.4f},"
        f"{r['nu']:.2f},{r['sig']:.3f},{r['fwhm_mT']:.3f},{r['wiggles']},"
        f"{pred_H_fixed:.3f},{pred_F_fixed:.3f}"
    ),
)

polythene = next(r for r in sample_rows if r["name"] == "polythene")
glycerine_pt = next(r for r in sample_rows if r["name"] == "glycerine")
ptfe_pt = next(r for r in sample_rows if r["name"] == "PTFE")
poly_vs_H = abs(polythene["nu"] - pred_H_fixed)
poly_vs_F = abs(polythene["nu"] - pred_F_fixed)

# ======================================================================================
# FIGURES
# ======================================================================================

def save(fig, name):
    # Every figure carries the same in-image representative-data flag, so a figure lifted out
    # of the PDF on its own still says what it is.
    # Offset is set in inches, then converted, so short figures clear their axis labels too.
    fig.text(0.99, -0.34 / fig.get_figheight(), "Representative data", ha="right", va="bottom",
             fontsize=7, color="0.45")
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


figures = []

# ---- Figure 1: Helmholtz calibration -------------------------------------------------
fig = plt.figure(figsize=(6.2, 4.4))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax = fig.add_subplot(gs[0])
axr = fig.add_subplot(gs[1], sharex=ax)

xx = np.linspace(0, 1.3, 200)
ax.errorbar(esr_I_cal, B_cal_meas, yerr=sig_B_cal, fmt="o", ms=4, color=C_BLUE,
            ecolor=C_BLUE, elinewidth=0.9, label="measured field")
ax.plot(xx, s_fit * xx + b_fit, "-", color=C_ORANGE,
        label=f"least squares fit, gradient {s_fit:.3f} mT A$^{{-1}}$")
ax.plot(xx, S_THEORY_mT_per_A * xx, "--", color=C_GREEN,
        label=f"Helmholtz formula, {S_THEORY_mT_per_A:.3f} mT A$^{{-1}}$")
ax.set_ylabel("Magnetic field $B_0$ (mT)")
ax.legend(loc="upper left", frameon=False)
ax.set_xlim(0, 1.3)
plt.setp(ax.get_xticklabels(), visible=False)

axr.axhline(0.0, color=C_BLACK, lw=0.7)
axr.errorbar(esr_I_cal, resid_cal * 1e3, yerr=sig_B_cal * 1e3, fmt="o", ms=4,
             color=C_BLUE, ecolor=C_BLUE, elinewidth=0.9)
axr.set_xlabel("Helmholtz coil current $I_=$ (A)")
axr.set_ylabel("Residual\n($\\mu$T)")
figures.append(save(fig, "fig_esr_helmholtz_calibration.pdf"))

# ---- Figure 2: ESR resonance frequency against field ---------------------------------
fig = plt.figure(figsize=(6.2, 4.8))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax = fig.add_subplot(gs[0])
axr = fig.add_subplot(gs[1], sharex=ax)

styles = {
    "large": (C_BLUE, "o", "large coil, 15-30 MHz"),
    "medium": (C_ORANGE, "s", "medium coil, 35-75 MHz"),
    "small": (C_GREEN, "^", "small coil, 80-130 MHz"),
}
for coil, (col, mk, lab) in styles.items():
    sel = np.array([c == coil for c in esr_coilname])
    ax.errorbar(esr_B0[sel], esr_nu[sel], xerr=esr_sigB0[sel], fmt=mk, ms=4, color=col,
                ecolor=col, elinewidth=0.9, linestyle="none", label=lab)
    axr.errorbar(esr_B0[sel], resid_esr[sel], yerr=m_esr * esr_sigB0[sel], fmt=mk, ms=4,
                 color=col, ecolor=col, elinewidth=0.9, linestyle="none")

bb = np.linspace(0, 5.0, 200)
ax.plot(bb, m_esr * bb + c_esr, "-", color=C_BLACK, lw=1.0,
        label=f"global fit, gradient {m_esr:.2f} MHz mT$^{{-1}}$")
ax.set_ylabel(r"Resonance frequency $\nu$ (MHz)")
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 145)
ax.legend(loc="upper left", frameon=False)
plt.setp(ax.get_xticklabels(), visible=False)

axr.axhline(0.0, color=C_BLACK, lw=0.7)
axr.set_xlabel("Magnetic field $B_0$ (mT), from the measured coil calibration")
axr.set_ylabel("Residual\n(MHz)")
figures.append(save(fig, "fig_esr_nu_vs_B.pdf"))

# ---- Figure 3: ESR line width construction -------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(B_axis, line_noisy, "-", color=C_BLUE, lw=0.9, label="absorption signal")
ax.plot(B_axis, line_clean, "-", color=C_ORANGE, lw=1.2,
        label=f"Lorentzian, FWHM {delta_B0_mT:.3f} mT")
half = 0.5
bl = B_res_50_mT - Gamma_mT
br = B_res_50_mT + Gamma_mT
ax.axhline(1.0, color=C_GREEN, ls=":", lw=0.9)
ax.axhline(half, color=C_GREEN, ls="--", lw=0.9)
ax.plot([bl, bl], [0, half], color=C_BLACK, ls="--", lw=0.8)
ax.plot([br, br], [0, half], color=C_BLACK, ls="--", lw=0.8)
ax.annotate("", xy=(bl, half), xytext=(br, half),
            arrowprops=dict(arrowstyle="<->", color=C_BLACK, lw=1.0))
ax.text(B_res_50_mT, half + 0.10, r"$\Delta B_0$ = " + f"{delta_B0_mT:.3f} mT",
        ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.text(B_axis[5], 1.02, "full signal height", color=C_GREEN, va="bottom", fontsize=8)
ax.text(B_axis[5], half + 0.02, "half height", color=C_GREEN, va="bottom", fontsize=8)
ax.set_xlabel("Magnetic field $B_0$ (mT)")
ax.set_ylabel("ESR signal (arbitrary units)")
ax.set_ylim(-0.08, 1.20)
ax.legend(loc="upper right", frameon=False)
figures.append(save(fig, "fig_esr_linewidth.pdf"))

# ---- Figure 4: NMR magnet calibration, both branches ---------------------------------
fig = plt.figure(figsize=(6.2, 6.0))
gs = GridSpec(2, 2, height_ratios=[2.0, 1.2], hspace=0.32, wspace=0.30)
axm = fig.add_subplot(gs[0, :])
axg = fig.add_subplot(gs[1, 0])
axres = fig.add_subplot(gs[1, 1])

ii = np.linspace(2.98, 4.02, 200)
axm.errorbar(nmr_I_nom, B_rise_meas, yerr=SIG_B_TESLAMETER_T, xerr=SIG_I_NMR_A, fmt="o",
             ms=4, color=C_BLUE, ecolor=C_BLUE, elinewidth=0.9, linestyle="none",
             label="rising sweep, 3.0 to 4.0 A")
axm.errorbar(nmr_I_nom, B_fall_meas, yerr=SIG_B_TESLAMETER_T, xerr=SIG_I_NMR_A, fmt="s",
             ms=4, color=C_ORANGE, ecolor=C_ORANGE, elinewidth=0.9, linestyle="none",
             label="falling sweep, 4.0 to 3.0 A")
axm.plot(ii, poly_value(coef_quad, ii), "-", color=C_BLUE, lw=1.1,
         label="quadratic fit to the rising sweep")
axm.annotate("", xy=(3.00, B_rise_meas[0]), xytext=(3.00, B_fall_meas[0]),
             arrowprops=dict(arrowstyle="<->", color=C_BLACK, lw=1.0))
axm.annotate(f"branch gap {1e3*(B_fall_meas[0]-B_rise_meas[0]):.1f} mT",
             xy=(3.00, 0.5 * (B_rise_meas[0] + B_fall_meas[0])),
             xytext=(3.16, B_rise_meas[0] - 0.0045), fontsize=8, va="center",
             arrowprops=dict(arrowstyle="-", color=C_BLACK, lw=0.7))
axm.set_ylim(B_rise_meas[0] - 0.010, B_fall_meas[-1] + 0.004)
axm.set_xlim(2.94, 4.06)
axm.set_xlabel("Electromagnet coil current $I$ (A)")
axm.set_ylabel("Magnetic field $B_0$ (T)")
axm.legend(loc="lower right", frameon=False)

axg.plot(nmr_I_nom, hyst_gap_meas * 1e3, "o-", ms=4, color=C_PINK)
axg.axhline(0.0, color=C_BLACK, lw=0.7)
axg.set_xlabel("Coil current $I$ (A)")
axg.set_ylabel("Branch separation (mT)")

axres.axhline(0.0, color=C_BLACK, lw=0.7)
axres.plot(nmr_I_nom, resid_lin * 1e3, "s--", ms=4, color=C_ORANGE, label="straight line")
axres.plot(nmr_I_nom, resid_quad * 1e3, "o-", ms=4, color=C_BLUE, label="quadratic")
axres.set_xlabel("Coil current $I$ (A)")
axres.set_ylabel("Fit residual (mT)")
axres.legend(loc="best", frameon=False, fontsize=7)
figures.append(save(fig, "fig_nmr_magnet_calibration.pdf"))

# ---- Figure 5: NMR resonance frequency against field ---------------------------------
fig = plt.figure(figsize=(6.2, 4.8))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax = fig.add_subplot(gs[0])
axr = fig.add_subplot(gs[1], sharex=ax)

bb = np.linspace(0.372, 0.463, 100)
ax.errorbar(B_from_cal, nu_glycerine, xerr=sigB_from_cal, yerr=res_H["sig_nu"], fmt="o",
            ms=4, color=C_BLUE, ecolor=C_BLUE, elinewidth=0.9, linestyle="none",
            label="glycerine, $^{1}$H")
ax.plot(bb, res_H["m"] * bb + res_H["c"], "-", color=C_BLUE, lw=1.0,
        label=f"fit, {res_H['m']:.2f} MHz T$^{{-1}}$")
ax.errorbar(B_from_cal, nu_ptfe, xerr=sigB_from_cal, yerr=res_F["sig_nu"], fmt="s",
            ms=4, color=C_ORANGE, ecolor=C_ORANGE, elinewidth=0.9, linestyle="none",
            label="PTFE, $^{19}$F")
ax.plot(bb, res_F["m"] * bb + res_F["c"], "-", color=C_ORANGE, lw=1.0,
        label=f"fit, {res_F['m']:.2f} MHz T$^{{-1}}$")
ax.set_ylabel(r"Resonance frequency $\nu$ (MHz)")
ax.legend(loc="upper left", frameon=False, ncol=2)
plt.setp(ax.get_xticklabels(), visible=False)

axr.axhline(0.0, color=C_BLACK, lw=0.7)
axr.errorbar(B_from_cal, res_H["resid"] * 1e3, yerr=res_H["sig_eff"] * 1e3, fmt="o", ms=4,
             color=C_BLUE, ecolor=C_BLUE, elinewidth=0.9, linestyle="none")
axr.errorbar(B_from_cal, res_F["resid"] * 1e3, yerr=res_F["sig_eff"] * 1e3, fmt="s", ms=4,
             color=C_ORANGE, ecolor=C_ORANGE, elinewidth=0.9, linestyle="none")
axr.set_xlabel("Magnetic field $B_0$ (T), from the rising-branch calibration")
axr.set_ylabel("Residual\n(kHz)")
figures.append(save(fig, "fig_nmr_nu_vs_B.pdf"))

# ---- Figure 6: four-sample comparison -------------------------------------------------
fig, (axl, axr2) = plt.subplots(1, 2, figsize=(6.2, 3.6),
                                gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34})

# Left panel: each line on its own frequency offset axis, in kilohertz, so the very different
# widths and peak heights are actually visible. On an absolute megahertz axis all four collapse
# into indistinguishable spikes.
offset_kHz = np.linspace(-45.0, 45.0, 1200)
cols = {"glycerine": C_BLUE, "polystyrene": C_ORANGE, "polythene": C_GREEN, "PTFE": C_PINK}
for r in sample_rows:
    gam = GAMMA2PI_H if "1H" in r["nucleus"] else GAMMA2PI_F
    width_kHz = gam * r["width_mT"] * 1e-3 * 1e3
    if r["name"] == "polythene":
        # Composite line: a narrow amorphous component on a broad crystalline base. The two
        # together have the tabulated overall width; only the narrow part produces transients.
        narrow_kHz = gam * r["narrow_mT"] * 1e-3 * 1e3
        broad_kHz = 2.2 * width_kHz
        shape = (0.45 * r["rel_height"] / (1.0 + (offset_kHz / (narrow_kHz / 2.0)) ** 2)
                 + 0.55 * r["rel_height"] / (1.0 + (offset_kHz / (broad_kHz / 2.0)) ** 2))
        lab = f"{r['name']}, {width_kHz:.0f} kHz composite"
    else:
        shape = r["rel_height"] / (1.0 + (offset_kHz / (width_kHz / 2.0)) ** 2)
        lab = f"{r['name']}, {width_kHz:.1f} kHz" if width_kHz < 1.0 else f"{r['name']}, {width_kHz:.0f} kHz"
    axl.plot(offset_kHz, shape, "-", color=cols[r["name"]], lw=1.2, label=lab)
axl.set_xlabel(r"Offset from line centre, $\nu - \nu_0$ (kHz)")
axl.set_ylabel("NMR signal (arbitrary units)")
axl.set_xlim(-45, 45)
axl.legend(loc="upper right", frameon=False, fontsize=7)

names = [r["name"] for r in sample_rows]
ypos = np.arange(len(names))[::-1]
axr2.errorbar([r["nu"] for r in sample_rows], ypos,
              xerr=[r["sig"] for r in sample_rows], fmt="o", ms=5, color=C_BLUE,
              ecolor=C_BLUE, elinewidth=1.0, linestyle="none")
axr2.axvline(pred_H_fixed, color=C_GREEN, ls="--", lw=1.0,
             label=f"$^{{1}}$H prediction, {pred_H_fixed:.2f} MHz")
axr2.axvline(pred_F_fixed, color=C_PINK, ls=":", lw=1.2,
             label=f"$^{{19}}$F prediction, {pred_F_fixed:.2f} MHz")
axr2.annotate("", xy=(pred_F_fixed, 3.55), xytext=(pred_H_fixed, 3.55),
              arrowprops=dict(arrowstyle="<->", color=C_BLACK, lw=0.9))
axr2.text(0.5 * (pred_F_fixed + pred_H_fixed), 3.65, f"{split_MHz:.2f} MHz",
          ha="center", va="bottom", fontsize=7.5)
axr2.set_yticks(ypos)
axr2.set_yticklabels(names)
axr2.set_xlabel(r"Resonance frequency $\nu$ (MHz)")
axr2.set_xlim(16.45, 18.35)
axr2.set_ylim(-1.6, 4.3)
axr2.legend(loc="lower left", frameon=False, fontsize=7)
figures.append(save(fig, "fig_nmr_sample_comparison.pdf"))

# ---- Figure 7: the two oscilloscope displays and the transient train ------------------
# The script asks for the two-channel display, the XY display and the wiggles to be sketched.
# These panels are the model of what the screen shows, built from the same representative
# parameters as the rest of the NMR analysis.
fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(6.4, 2.5),
                                    gridspec_kw={"wspace": 0.38})

gly = next(r for r in sample_rows if r["name"] == "glycerine")
gamma_H = 2.0 * np.pi * GAMMA2PI_H * 1e6                      # rad / s / T
T2star_gly = gly["T2star_s"]
width_T_gly = gly["narrow_mT"] * 1e-3

# (a) two-channel display: NMR signal and the modulation reference against time
t_ms = np.linspace(0.0, 2.0 / F_MOD_NMR_Hz * 1e3, 4000)       # two modulation periods, ms
mod_ref = np.sin(2.0 * np.pi * F_MOD_NMR_Hz * t_ms * 1e-3)
b_off_T = B_MOD_NMR_PEAK_T * mod_ref                          # field offset from resonance
nmr_trace = 1.0 / (1.0 + (b_off_T / (0.5 * width_T_gly)) ** 2)
axa.plot(t_ms, mod_ref * 0.45 - 0.75, "-", color=C_ORANGE, lw=0.9, label="modulation, CH I")
axa.plot(t_ms, nmr_trace, "-", color=C_BLUE, lw=0.9, label="NMR signal, CH II")
axa.set_xlabel("Time (ms)")
axa.set_ylabel("Signal (arb.)")
axa.set_title("(a) two-channel display", fontsize=8)
axa.set_ylim(-1.4, 1.35)
axa.legend(loc="upper center", frameon=False, fontsize=6.5, ncol=1, handlelength=1.4)

# (b) XY display: signal against the modulation reference. With a phase error the two half-wave
# traces are displaced horizontally and the trace opens into a loop; the PHASE control removes it.
phase_err = 0.02                                              # radians of uncompensated lag
b_lag_T = B_MOD_NMR_PEAK_T * np.sin(2.0 * np.pi * F_MOD_NMR_Hz * t_ms * 1e-3 - phase_err)
nmr_lagged = 1.0 / (1.0 + (b_lag_T / (0.5 * width_T_gly)) ** 2)
axb.plot(mod_ref, nmr_lagged, "--", color="0.6", lw=0.8, label="phase uncorrected")
axb.plot(mod_ref, nmr_trace, "-", color=C_BLUE, lw=0.9, label="phase set")
axb.set_xlabel("B SIGNAL (arb.), field axis")
axb.set_ylabel("NMR signal (arb.)")
axb.set_title("(b) XY display", fontsize=8)
axb.set_xlim(-0.06, 0.06)
axb.set_ylim(-0.1, 1.25)
axb.legend(loc="upper right", frameon=False, fontsize=6.5)

# (c) transient train after passage through resonance
t_us = np.linspace(0.0, 1.6e-3, 6000)
dphi = 0.5 * gamma_H * DBDT_NMR_T_per_s * t_us ** 2
transient = np.exp(-t_us / T2star_gly) * np.cos(dphi)
axc.plot(t_us * 1e3, transient, "-", color=C_GREEN, lw=0.9)
axc.plot(t_us * 1e3, np.exp(-t_us / T2star_gly), "--", color="0.55", lw=0.8,
         label=f"$T_2^{{*}}$ = {T2star_gly*1e6:.0f} $\\mu$s")
n_show = int(np.floor(gly["n_wiggles"]))
for n in range(1, max(n_show, 1) + 1):
    tn = np.sqrt(4.0 * np.pi * n / (gamma_H * DBDT_NMR_T_per_s))
    axc.plot([tn * 1e3], [np.exp(-tn / T2star_gly)], "o", ms=3, color=C_BLACK)
axc.set_xlabel("Time after passage (ms)")
axc.set_ylabel("NMR signal (arb.)")
axc.set_title("(c) glycerine transients", fontsize=8)
axc.set_xlim(0, 1.6)
axc.legend(loc="upper right", frameon=False, fontsize=6.5)
figures.append(save(fig, "fig_nmr_displays_wiggles.pdf"))

# ======================================================================================
# RESULTS SUMMARY
# ======================================================================================

def pm(v, u, nd=4):
    return f"{v:.{nd}f} +/- {u:.{nd}f}"


summary = f"""# Results summary: Electron Spin Resonance and Nuclear Magnetic Resonance

**REPRESENTATIVE DATA. Every number on this page was generated by
`reports/analysis/generate_esr_nmr.py` from accepted physical parameters plus realistic
instrument scatter. None of it is a record of measured results.** Random seed
{SEED}, so the whole set is reproducible by rerunning the script.

Experiment: Physics 3H Laboratory (PHYS 4009), University of Glasgow. Departmental script
by I. MacLaren, 6 October 2023.

Units are stated explicitly on every line.

---

## 0. Constants used (CODATA 2022)

| Constant | Value | Unit |
|---|---|---|
| Planck constant h | {H_PLANCK:.8e} | J s (exact by SI definition) |
| Reduced Planck constant hbar | {HBAR:.9e} | J s |
| Bohr magneton mu_B | {MU_B:.10e} | J / T |
| Nuclear magneton mu_n | {MU_N:.10e} | J / T |
| Boltzmann constant k_B | {K_B:.6e} | J / K (exact by SI definition) |
| Vacuum permeability mu_0 | {MU_0:.9e} | V s / (A m) |

Derived combinations:

| Quantity | Value | Unit |
|---|---|---|
| h / mu_B | {H_PLANCK/MU_B:.6e} | T s |
| h / mu_n | {H_PLANCK/MU_N:.6e} | T s |
| g_e mu_B / h (free electron) | {G_ELECTRON_FREE*MU_B/H_PLANCK/1e9:.4f} | MHz / mT |
| g(DPPH) mu_B / h at g = {G_DPPH_LIT} | {G_DPPH_LIT*MU_B/H_PLANCK/1e9:.4f} | MHz / mT |
| g_p mu_n / h (proton) | {GAMMA2PI_H:.6f} | MHz / T |
| g_F mu_n / h (fluorine-19) | {GAMMA2PI_F:.6f} | MHz / T |

---

## 1. Ground truth used to generate the data

These are the inputs to the simulation. The report must never quote them as results.

| Parameter | Ground-truth value | Unit | Why this value |
|---|---|---|---|
| DPPH g factor | {G_DPPH_TRUE:.4f} | dimensionless | accepted solid-DPPH value |
| Helmholtz coil constant | {S_TRUE_mT_per_A:.3f} | mT / A | deliberately below the nominal {S_THEORY_mT_per_A:.4f}, since the coils need not have exactly 320 turns |
| Residual ambient axial field at the DPPH sample | {B_AMBIENT_TRUE_mT:.3f} | mT | imperfect LOW FIELD zeroing, gives the small non-zero intercept |
| Magnetometer zero offset drawn for this run | {b_off_mag_true:+.4f} | mT | drawn once from a uniform range +/- 0.020 mT |
| DPPH line width (FWHM) | {FWHM_TRUE_mT:.3f} | mT | consistent with exchange-narrowed DPPH |
| Electromagnet rising branch B(I) | {A0_TRUE:.4f} + {A1_TRUE:.4f} I {A2_TRUE:+.5f} I^2 | T, I in A | valid 3.0 to 4.0 A only |
| Hysteresis branch gap at 3.00 A | {HYST_GAP_AT_3A_T*1e3:.1f} | mT | closes linearly to zero at 4.00 A |
| Proton g factor | {G_PROTON_LIT:.10f} | dimensionless | CODATA 2022 |
| Fluorine-19 g factor | {G_FLUORINE_LIT:.6f} | dimensionless | Stone / IUPAC tables |

Instrument scatter injected:

| Quantity | 1 sigma | Unit |
|---|---|---|
| ESR Hall magnetometer reading | {SIG_B_MAGNETOMETER_mT:.3f} | mT, then quantised to 0.01 mT |
| ESR resonance current, large coil | {SIG_I_ESR['large']:.3f} | A, quantised to 0.001 A |
| ESR resonance current, medium coil | {SIG_I_ESR['medium']:.3f} | A |
| ESR resonance current, small coil | {SIG_I_ESR['small']:.3f} | A |
| NMR teslameter reading | {SIG_B_TESLAMETER_T*1e3:.1f} | mT, then quantised to the 0.1 mT display step |
| NMR coil current setting | {SIG_I_NMR_A:.3f} | A |
| NMR frequency, glycerine | {SIG_NU['glycerine']:.3f} | MHz |
| NMR frequency, PTFE | {SIG_NU['ptfe']:.3f} | MHz |
| NMR frequency, polystyrene | {SIG_NU['polystyrene']:.3f} | MHz |
| NMR frequency, polythene | {SIG_NU['polythene']:.3f} | MHz |

**No systematic scale bias was injected into either field axis.** The 1 per cent tolerance
on the 112 mT reference magnet is carried through as a systematic uncertainty only, because
a student who does not know its value must treat it that way.

---

## 2. ESR, part (a): Helmholtz coil calibration

Data file: `reports/data/esr_helmholtz_calibration.csv`
Figure: `reports/figures/fig_esr_helmholtz_calibration.pdf`

Twelve points, coil current 0.1 to 1.2 A in 0.1 A steps, unweighted-equivalent least squares
of B = s I + B_offset with sigma_B = {SIG_B_MAGNETOMETER_mT:.3f} mT on every point.

| Fitted quantity | Value | Unit |
|---|---|---|
| Coil constant s (gradient), statistical only | {pm(s_fit, sig_s_fit, 4)} | mT / A |
| Coil constant s, statistical and 1 per cent systematic combined | {pm(s_fit, sig_s_total, 4)} | mT / A |
| Intercept B_offset | {pm(b_fit, sig_b_fit, 4)} | mT |
| chi-squared | {chi2_cal:.2f} | dimensionless |
| Reduced chi-squared (10 degrees of freedom) | {chi2r_cal:.2f} | dimensionless |
| Largest residual | {np.max(np.abs(resid_cal))*1e3:.1f} | microtesla |
| Measured field range | {B_cal_meas.min():.2f} to {B_cal_meas.max():.2f} | mT |

Comparison with theory:

| Quantity | Value | Unit |
|---|---|---|
| Theoretical coil constant, script equation (6), n = 320, r = 6.8 cm | {S_THEORY_mT_per_A:.4f} | mT / A |
| Script's own rounded statement | 4.23 | mT / A |
| Fitted coil constant | {s_fit:.4f} | mT / A |
| Difference, fitted minus theoretical | {s_fit - S_THEORY_mT_per_A:+.4f} | mT / A |
| Percentage difference from theory | {100*(s_fit - S_THEORY_mT_per_A)/S_THEORY_mT_per_A:+.2f} | per cent |
| Ground truth used to generate the data | {S_TRUE_mT_per_A:.4f} | mT / A |
| Percentage difference, fitted minus ground truth | {100*(s_fit - S_TRUE_mT_per_A)/S_TRUE_mT_per_A:+.2f} | per cent |
| Effective turns per coil implied by the fit, n = s r / (0.8^1.5 mu_0) | {s_fit*1e-3*R_COIL/((0.8**1.5)*MU_0):.1f} | turns (stated value 320) |

---

## 3. ESR, part (b): resonance frequency against magnetic field

Data file: `reports/data/esr_resonance_frequency.csv`
Figure: `reports/figures/fig_esr_nu_vs_B.pdf`

Twenty-four frequencies, five repeat settings of the resonance current at each, in 5 MHz steps:
large coil 15 to 30 MHz (4 points), medium coil 35 to 75 MHz (9 points), small coil 80 to 130 MHz
(11 points). Each current is the mean of five repeats and its uncertainty is the standard error
of those five. Currents are converted to field with the measured calibration above.

| Fitted quantity | Value | Unit |
|---|---|---|
| Gradient d(nu)/d(B_0), statistical only | {pm(m_esr, sig_m_esr_stat, 4)} | MHz / mT |
| Intercept nu_0 | {pm(c_esr, sig_c_esr, 4)} | MHz |
| chi-squared | {chi2_esr:.2f} | dimensionless |
| Reduced chi-squared (22 degrees of freedom) | {chi2r_esr:.2f} | dimensionless |
| Largest residual | {np.max(np.abs(resid_esr)):.3f} | MHz |
| Field range covered | {esr_B0.min():.3f} to {esr_B0.max():.3f} | mT |
| Current range covered | {esr_Imean.min():.3f} to {esr_Imean.max():.3f} | A |
| Mean standard error on the current | {np.mean(esr_Isem):.4f} | A |

Cross-check by the two-gradient route (fit nu against current, then divide by the coil constant):

| Quantity | Value | Unit |
|---|---|---|
| Gradient d(nu)/d(I_=) | {pm(k_esr, sig_k_esr, 3)} | MHz / A |
| d(nu)/d(B_0) = k / s | {pm(m_esr_routeB, sig_m_esr_routeB, 4)} | MHz / mT |
| Agreement with the direct fit | {abs(m_esr_routeB-m_esr):.2e} | MHz / mT (algebraically identical, as expected) |

**g factor of DPPH:**

| Quantity | Value | Unit |
|---|---|---|
| g_s(DPPH), central value | {g_dpph:.4f} | dimensionless |
| Statistical uncertainty (resonance fit and calibration gradient) | {sig_g_stat:.4f} | dimensionless |
| Systematic uncertainty (1 per cent reference magnet) | {sig_g_syst:.4f} | dimensionless |
| Total uncertainty | {sig_g_total:.4f} | dimensionless |
| **Quoted result** | **{g_dpph:.3f} +/- {sig_g_stat:.3f} (stat) +/- {sig_g_syst:.3f} (syst) = {g_dpph:.3f} +/- {sig_g_total:.3f}** | dimensionless |
| Total relative uncertainty | {100*sig_g_total/g_dpph:.2f} | per cent |
| Literature g(DPPH) | {G_DPPH_LIT:.4f} +/- 0.0002 | dimensionless |
| Percentage difference from g(DPPH) | {100*(g_dpph-G_DPPH_LIT)/G_DPPH_LIT:+.3f} | per cent |
| Discrepancy in units of the total uncertainty | {abs(g_dpph-G_DPPH_LIT)/sig_g_total:.2f} | sigma |
| Free electron g_e | {G_ELECTRON_FREE:.9f} | dimensionless |
| Percentage difference from g_e | {100*(g_dpph-G_ELECTRON_FREE)/G_ELECTRON_FREE:+.3f} | per cent |
| Discrepancy from g_e in units of the total uncertainty | {abs(g_dpph-G_ELECTRON_FREE)/sig_g_total:.2f} | sigma |
| Ground truth used to generate the data | {G_DPPH_TRUE:.4f} | dimensionless |
| Percentage difference, fitted minus ground truth | {100*(g_dpph-G_DPPH_TRUE)/G_DPPH_TRUE:+.3f} | per cent |

Notes for the discussion:
- The DPPH g-shift above the free electron is 0.065 per cent, while the total uncertainty here is
  {100*sig_g_total/g_dpph:.2f} per cent. This experiment therefore **cannot** resolve g(DPPH) from g_e,
  and the report must say so.
- A purely orbital moment would give g = 1 and a gradient of mu_B / h = {MU_B/H_PLANCK/1e9:.3f} MHz / mT,
  half the observed value. That hypothesis is excluded by a factor of two, far outside the error bar.
- Intercept diagnostic. The fitted intercept is {c_esr:+.3f} +/- {sig_c_esr:.3f} MHz, which is
  {abs(c_esr)/sig_c_esr:.0f} standard deviations from zero, so it is **not** consistent with zero.
  Dividing by the gradient, the field at the DPPH sample exceeded the calibrated field by
  {B_extra_at_sample_mT:+.4f} +/- {sig_B_extra_at_sample_mT:.4f} mT. That is the right size for the
  residual ambient field left after zeroing the magnetometer at the coil centre, and comparable to
  the Earth's field of roughly 0.05 mT. The offset shifts the intercept and leaves the gradient
  untouched, which is exactly why the g factor is taken from the gradient and not from a single
  pair of readings. For comparison, the simulation injected a real ambient field of
  {B_AMBIENT_TRUE_mT:.4f} mT at the sample and a magnetometer zero offset of
  {b_off_mag_true:+.4f} mT, so the value the analysis chain should recover is
  {B_extra_expected_mT:+.4f} mT.

---

## 4. ESR, part (c): line width at 50 MHz

Data files: `reports/data/esr_linewidth.csv`, `reports/data/esr_lineshape.csv`
Figure: `reports/figures/fig_esr_linewidth.pdf`

Medium plug-in coil, oscilloscope in XY mode, both channels initially 0.5 V/cm, trace expanded
to exactly the 10.0 cm screen width, X deflection then expanded to 0.2 V/cm to read the width at
half height. Script equation (7): delta_I = 2 sqrt(2) I_mod (delta_U / U_mod).

| Quantity | Value | Uncertainty | Unit |
|---|---|---|---|
| RF frequency | 50.0 | 0.1 (readout) | MHz |
| Resonance current I_= | {I_res_50:.3f} | {row50['I_sem_A']:.4f} | A |
| Resonance field B_0 | {B_res_50_mT:.3f} | {row50['sigB0_mT']:.4f} | mT |
| Modulation current I_mod (RMS) | {I_mod_meas_A:.3f} | {sig_I_mod_A:.3f} | A |
| Full modulation voltage U_mod | {U_mod_V:.2f} | {sig_U_mod_V:.2f} | V |
| Width at half height delta_U | {delta_U_meas_V:.2f} | {sig_delta_U_V:.2f} | V |
| Half-width current delta_I | {delta_I_A:.4f} | {delta_I_A*np.sqrt((sig_delta_U_V/delta_U_meas_V)**2 + (sig_I_mod_A/I_mod_meas_A)**2 + (sig_U_mod_V/U_mod_V)**2):.4f} | A |
| **Line width delta_B_0 (FWHM)** | **{delta_B0_mT:.3f}** | **{sig_delta_B0_mT:.3f}** | **mT** |
| Line width in gauss | {delta_B0_mT*10:.2f} | {sig_delta_B0_mT*10:.2f} | G |
| Relative uncertainty on delta_B_0 | {100*rel_delta_B0:.1f} | | per cent |
| Equivalent peak-to-peak derivative width | {delta_Bpp_mT:.3f} | {sig_delta_Bpp_mT:.3f} | mT |
| Same, in gauss | {delta_Bpp_mT*10:.2f} | {sig_delta_Bpp_mT*10:.2f} | G |
| Transverse relaxation time T_2 | {T2_s*1e9:.1f} | {sig_T2_s*1e9:.1f} | ns |
| Ground truth FWHM used to generate the data | {FWHM_TRUE_mT:.3f} | | mT |
| Percentage difference, fitted minus ground truth | {100*(delta_B0_mT-FWHM_TRUE_mT)/FWHM_TRUE_mT:+.2f} | | per cent |

Literature comparison:

| Quantity | Value | Unit |
|---|---|---|
| Published DPPH peak-to-peak derivative width, polycrystalline | 1.5 to 3.0 | G |
| Same converted to FWHM (multiply by sqrt(3)) | 2.6 to 5.2 | G |
| This measurement, FWHM | {delta_B0_mT*10:.2f} +/- {sig_delta_B0_mT*10:.2f} | G |
| Published T_1 = T_2 for concentrated solid DPPH | approximately 62 | ns |
| This measurement, T_2 | {T2_s*1e9:.0f} +/- {sig_T2_s*1e9:.0f} | ns |
| Rigid-lattice dipolar width estimate for DPPH | approximately 4.3 | mT |
| Ratio, rigid-lattice estimate to observed width | {4.3/delta_B0_mT:.0f} | dimensionless, the exchange-narrowing factor |

Note for the report, to be stated rather than glossed over: the measured peak-to-peak equivalent
of {delta_Bpp_mT*10:.2f} +/- {sig_delta_Bpp_mT*10:.2f} G sits at or just below the lower end of the
1.5 to 3.0 G band commonly quoted for polycrystalline DPPH, a difference of about
{abs(delta_Bpp_mT*10-1.5)/(sig_delta_Bpp_mT*10):.1f} standard deviations from 1.5 G. That is
consistent agreement, not a discrepancy, and the likely reasons for sitting low are the modest
oscilloscope reading precision and the fact that published widths vary with the sample's
preparation and its solvent history.

Internal consistency checks on the line-width measurement:

| Check | Value | Verdict |
|---|---|---|
| Peak-to-peak modulation field s x 2 sqrt(2) x I_mod | {B_mod_pp_mT:.3f} mT | must stay below B_0 = {B_res_50_mT:.3f} mT so the field never reverses: {'PASS' if B_mod_pp_mT < B_res_50_mT else 'FAIL'} |
| Fraction of the 10 cm trace occupied by the line | {100*delta_B0_mT/B_mod_pp_mT:.0f} per cent | readable on a real screen: {'PASS' if 5 < 100*delta_B0_mT/B_mod_pp_mT < 60 else 'CHECK'} |
| Line width in cm at 0.5 V/cm | {10*delta_B0_mT/B_mod_pp_mT:.1f} cm | |
| Line width in cm at the expanded 0.2 V/cm | {2.5*10*delta_B0_mT/B_mod_pp_mT:.1f} cm | |

---

## 5. NMR, part (d): electromagnet calibration and hysteresis

Data file: `reports/data/nmr_magnet_calibration.csv`
Figure: `reports/figures/fig_nmr_magnet_calibration.pdf`

Eleven points on each branch, 3.00 to 4.00 A in 0.10 A steps. Teslameter display resolution
0.1 mT with a reading scatter of {SIG_B_TESLAMETER_T*1e3:.1f} mT, and coil current read to
{SIG_I_NMR_A:.3f} A. The rising branch is fitted with B(I) = a_0 + a_1 I + a_2 I^2
by effective-variance weighted least squares, with the current uncertainty projected onto the
field axis through dB/dI.

| Fitted quantity | Value | Unit |
|---|---|---|
| a_0 | {pm(a0, sig_a0, 4)} | T |
| a_1 | {pm(a1, sig_a1, 4)} | T / A |
| a_2 | {pm(a2, sig_a2, 5)} | T / A^2 |
| chi-squared | {chi2_quad:.2f} | dimensionless |
| Reduced chi-squared (8 degrees of freedom) | {chi2r_quad:.2f} | dimensionless |
| Largest quadratic residual | {np.max(np.abs(resid_quad))*1e3:.2f} | mT |
| Largest straight-line residual | {np.max(np.abs(resid_lin))*1e3:.2f} | mT |
| Reduced chi-squared for a straight line (9 degrees of freedom) | {chi2r_lin_e:.2f} | dimensionless |
| Cubic coefficient a_3, if a cubic is fitted | {pm(cubic_coeff, sig_cubic_coeff, 5)} | T / A^3 |
| Cubic coefficient in units of its own uncertainty | {abs(cubic_coeff)/sig_cubic_coeff:.2f} | sigma, so consistent with zero |

Model comparison, so the choice of a quadratic is justified rather than assumed:

| Test | Value | Reading |
|---|---|---|
| Quadratic coefficient a_2 in units of its own uncertainty | {quad_significance:.1f} | sigma, so the curvature is real |
| Improvement in chi-squared on adding the I^2 term | {delta_chi2_lin_quad:.1f} | for one extra parameter |
| F statistic, straight line against quadratic | {F_lin_quad:.1f} | large, so the quadratic is preferred |
| Predicted peak straight-line residual from curvature, a_2 (1 A)^2 / 8 | {curvature_amplitude_mT:.2f} | mT, against {SIG_B_TESLAMETER_T*1e3:.1f} mT of point scatter |
| Improvement in chi-squared on adding an I^3 term | {delta_chi2_quad_cub:.2f} | for one extra parameter |
| F statistic, quadratic against cubic | {F_quad_cub:.2f} | small, so the cubic is not justified |

Note on the quadratic reduced chi-squared of {chi2r_quad:.2f}. A value this far below 1 means the
assigned uncertainties are slightly conservative, not that the fit is suspect: the teslameter
scatter of {SIG_B_TESLAMETER_T*1e3:.1f} mT and the current reading uncertainty of
{SIG_I_NMR_A:.3f} A were both assessed before the run rather than from the residuals themselves.
The report should say this plainly rather than quietly quoting a number under one.

Gradient and curvature:

| Quantity | Value | Unit |
|---|---|---|
| dB/dI at 3.00 A | {pm(dBdI_at(3.00), sig_dBdI_at(3.00), 4)} | T / A |
| dB/dI at 3.50 A | {pm(dBdI_at(3.50), sig_dBdI_at(3.50), 4)} | T / A |
| dB/dI at 4.00 A | {pm(dBdI_at(4.00), sig_dBdI_at(4.00), 4)} | T / A |
| Reduction in dB/dI across the range | {100*(dBdI_at(3.00)-dBdI_at(4.00))/dBdI_at(3.00):.1f} | per cent |
| Field at 3.00 A | {poly_value(coef_quad, 3.00):.4f} | T |
| Field at 3.50 A | {poly_value(coef_quad, 3.50):.4f} | T |
| Field at 4.00 A | {poly_value(coef_quad, 4.00):.4f} | T |
| Ground-truth a_0, a_1, a_2 | {A0_TRUE:.4f}, {A1_TRUE:.4f}, {A2_TRUE:.5f} | T, T/A, T/A^2 |

Hysteresis, from the two branches:

| Quantity | Value | Unit |
|---|---|---|
| Branch separation at 3.00 A | {hyst_gap_meas[0]*1e3:.1f} | mT |
| Branch separation at 3.50 A | {hyst_gap_meas[5]*1e3:.1f} | mT |
| Branch separation at 4.00 A | {hyst_gap_meas[-1]*1e3:.1f} | mT |
| Mean separation over the range | {np.mean(hyst_gap_meas)*1e3:.1f} | mT |
| Separation at 3.00 A as a fraction of B there | {100*hyst_gap_meas[0]/B_rise_meas[0]:.2f} | per cent |
| Equivalent proton frequency shift at 3.00 A | {GAMMA2PI_H*hyst_gap_meas[0]:.3f} | MHz |
| Ratio of that shift to the frequency setting precision (0.02 MHz) | {GAMMA2PI_H*hyst_gap_meas[0]/0.02:.0f} | dimensionless |

Warning to carry into the report: the fitted intercept a_0 = {a0:.4f} T is an extrapolation
artefact from 3 A back to 0 A and is **not** a remanent field. Real remanence for this magnet
would be 10 to 30 mT, several times smaller. The quadratic is a local empirical interpolation
over 3.0 to 4.0 A, in effect a Taylor expansion of the true B(I) truncated at second order, and
it must not be extrapolated outside that window.

---

## 6. NMR, parts (e) and (f): proton and fluorine g factors

Data file: `reports/data/nmr_resonance_frequency.csv`
Figure: `reports/figures/fig_nmr_nu_vs_B.pdf`

Eleven points per sample on the rising current sweep only, 3.00 to 4.00 A in 0.10 A steps.
Currents are converted to field with the rising-branch quadratic above, and each field carries
the full propagated uncertainty including all three coefficient covariances plus the current
setting error. The nu against B_0 fit uses the effective-variance method, iterated to convergence.

| Quantity | 1H in glycerine | 19F in PTFE | Unit |
|---|---|---|---|
| Frequency range | {nu_glycerine.min():.2f} to {nu_glycerine.max():.2f} | {nu_ptfe.min():.2f} to {nu_ptfe.max():.2f} | MHz |
| Field range | {B_from_cal.min():.4f} to {B_from_cal.max():.4f} | {B_from_cal.min():.4f} to {B_from_cal.max():.4f} | T |
| Assumed frequency scatter | {SIG_NU['glycerine']:.3f} | {SIG_NU['ptfe']:.3f} | MHz |
| Propagated field uncertainty, mean | {np.mean(sigB_from_cal)*1e3:.2f} | {np.mean(sigB_from_cal)*1e3:.2f} | mT |
| **Gradient d(nu)/d(B_0), route A** | **{pm(res_H['m'], res_H['sig_m_stat'], 3)}** | **{pm(res_F['m'], res_F['sig_m_stat'], 3)}** | MHz / T |
| Relative statistical uncertainty on the gradient | {100*res_H['rel_stat']:.2f} | {100*res_F['rel_stat']:.2f} | per cent |
| Intercept c | {pm(res_H['c'], res_H['sig_c'], 3)} | {pm(res_F['c'], res_F['sig_c'], 3)} | MHz |
| Intercept in units of its own uncertainty | {abs(res_H['c'])/res_H['sig_c']:.2f} | {abs(res_F['c'])/res_F['sig_c']:.2f} | sigma |
| Extra field at the sample implied by the intercept, c/m | {res_H['B_offset_T']*1e3:+.2f} +/- {res_H['sig_B_offset_T']*1e3:.2f} | {res_F['B_offset_T']*1e3:+.2f} +/- {res_F['sig_B_offset_T']*1e3:.2f} | mT |
| chi-squared | {res_H['chi2']:.2f} | {res_F['chi2']:.2f} | dimensionless |
| Reduced chi-squared (9 degrees of freedom) | {res_H['chi2r']:.2f} | {res_F['chi2r']:.2f} | dimensionless |
| Largest residual | {np.max(np.abs(res_H['resid']))*1e3:.0f} | {np.max(np.abs(res_F['resid']))*1e3:.0f} | kHz |
| Gradient, route B (chain rule at I = 3.50 A) | {pm(res_H['m_routeB'], res_H['sig_m_routeB'], 3)} | {pm(res_F['m_routeB'], res_F['sig_m_routeB'], 3)} | MHz / T |
| d(nu)/d(I) at 3.50 A | {pm(res_H['dnudI'], res_H['sig_dnudI'], 4)} | {pm(res_F['dnudI'], res_F['sig_dnudI'], 4)} | MHz / A |
| Gradient, Monte Carlo cross-check (5000 draws) | {pm(res_H['m_mc'], res_H['sig_m_mc'], 3)} | {pm(res_F['m_mc'], res_F['sig_m_mc'], 3)} | MHz / T |
| Literature gradient | {GAMMA2PI_H:.4f} | {GAMMA2PI_F:.4f} | MHz / T |
| Percentage difference of route A from literature | {100*(res_H['m']-GAMMA2PI_H)/GAMMA2PI_H:+.2f} | {100*(res_F['m']-GAMMA2PI_F)/GAMMA2PI_F:+.2f} | per cent |

**g factors:**

| Quantity | 1H in glycerine | 19F in PTFE | Unit |
|---|---|---|---|
| g, central value | {res_H['g']:.4f} | {res_F['g']:.4f} | dimensionless |
| Statistical uncertainty (both fits combined) | {res_H['sig_g_stat']:.4f} | {res_F['sig_g_stat']:.4f} | dimensionless |
| Systematic uncertainty (1 per cent reference magnet) | {res_H['sig_g_syst']:.4f} | {res_F['sig_g_syst']:.4f} | dimensionless |
| Total uncertainty | {res_H['sig_g_total']:.4f} | {res_F['sig_g_total']:.4f} | dimensionless |
| **Quoted result** | **{res_H['g']:.2f} +/- {res_H['sig_g_total']:.2f}** | **{res_F['g']:.2f} +/- {res_F['sig_g_total']:.2f}** | dimensionless |
| Total relative uncertainty | {100*res_H['rel_total']:.2f} | {100*res_F['rel_total']:.2f} | per cent |
| Literature g | {res_H['g_lit']:.6f} | {res_F['g_lit']:.6f} | dimensionless |
| Percentage difference from literature | {res_H['pct_diff']:+.2f} | {res_F['pct_diff']:+.2f} | per cent |
| Discrepancy in units of the total uncertainty | {res_H['z']:.2f} | {res_F['z']:.2f} | sigma |
| Ground truth used to generate the data | {G_PROTON_LIT:.6f} | {G_FLUORINE_LIT:.6f} | dimensionless |

Uncertainty budget, relative contributions to g:

| Contribution | 1H | 19F | Unit |
|---|---|---|---|
| Statistical, from the two fits in series | {100*res_H['rel_stat']:.2f} | {100*res_F['rel_stat']:.2f} | per cent |
| Teslameter reference magnet, correlated scale | {100*PROBE_SCALE_SYSTEMATIC:.2f} | {100*PROBE_SCALE_SYSTEMATIC:.2f} | per cent |
| Planck constant and nuclear magneton | negligible (h exact, mu_n to 3 parts in 1e10) | negligible | per cent |
| **Combined in quadrature** | **{100*res_H['rel_total']:.2f}** | **{100*res_F['rel_total']:.2f}** | **per cent** |

Ratio test, which cancels the correlated field scale entirely:

| Quantity | Value | Unit |
|---|---|---|
| Measured gradient ratio, 19F to 1H | {pm(ratio_meas, sig_ratio, 4)} | dimensionless |
| Literature ratio, {GAMMA2PI_F:.4f} / {GAMMA2PI_H:.4f} | {ratio_lit:.5f} | dimensionless |
| Percentage difference | {100*(ratio_meas-ratio_lit)/ratio_lit:+.2f} | per cent |
| Discrepancy in units of the ratio uncertainty | {abs(ratio_meas-ratio_lit)/sig_ratio:.2f} | sigma |

---

## 7. NMR, part (g): four-sample comparison at one fixed field

Data file: `reports/data/nmr_sample_comparison.csv`
Figure: `reports/figures/fig_nmr_sample_comparison.pdf`

Coil current fixed at {I_FIXED:.2f} A on the rising branch, field
{B_fixed_cal:.4f} +/- {sigB_fixed_cal:.4f} T from the calibration.

| Sample | Formula | Nucleus | Observed nu (MHz) | Uncertainty (MHz) | Line width (mT) | Wiggles |
|---|---|---|---|---|---|---|
"""

for r in sample_rows:
    summary += (
        f"| {r['name']} | {r['formula']} | {r['nucleus']} | {r['nu']:.2f} | {r['sig']:.3f} | "
        f"{r['fwhm_mT']:.3f} | {r['wiggles']} |\n"
    )

summary += f"""
| Prediction | Value | Unit |
|---|---|---|
| 1H resonance at this field | {pred_H_fixed:.3f} | MHz |
| 19F resonance at this field | {pred_F_fixed:.3f} | MHz |
| Separation of the two hypotheses | {split_MHz:.3f} | MHz |
| Fractional separation | {100*split_MHz/pred_H_fixed:.2f} | per cent |

Polythene identification:

| Test | Value | Unit |
|---|---|---|
| Observed polythene frequency | {polythene['nu']:.2f} +/- {polythene['sig']:.3f} | MHz |
| Difference from the 1H prediction | {poly_vs_H:.3f} | MHz |
| Same, in units of the measurement uncertainty | {poly_vs_H/polythene['sig']:.1f} | sigma |
| Difference from the 19F prediction | {poly_vs_F:.3f} | MHz |
| Same, in units of the measurement uncertainty | {poly_vs_F/polythene['sig']:.0f} | sigma |
| Difference from the observed glycerine line | {abs(polythene['nu']-glycerine_pt['nu']):.3f} | MHz |
| Difference from the observed PTFE line | {abs(polythene['nu']-ptfe_pt['nu']):.3f} | MHz |
| **Conclusion** | polythene resonates on the proton line, so it contains 1H and no 19F. Handbook formula (C2H4)n. | |

The two significances above use the frequency uncertainty alone. Both predictions are
proportional to the calibrated field, which carries the 1 per cent reference-magnet scale
error, so the honest comparison combines the two:

| Test, with the 1 per cent field scale carried | Value | Unit |
|---|---|---|
| Uncertainty on the 1H prediction from the field scale | {0.01*pred_H_fixed:.3f} | MHz |
| Uncertainty on the 19F prediction from the field scale | {0.01*pred_F_fixed:.3f} | MHz |
| Polythene against the 1H prediction | {poly_vs_H/np.hypot(polythene['sig'], 0.01*pred_H_fixed):.2f} | sigma |
| Polythene against the 19F prediction | {poly_vs_F/np.hypot(polythene['sig'], 0.01*pred_F_fixed):.2f} | sigma |
| Polystyrene against the 1H prediction, measurement error only | {abs(next(r for r in sample_rows if r['name']=='polystyrene')['nu']-pred_H_fixed)/0.120:.2f} | sigma |

Transients (wiggles). The number of countable maxima is N = gamma (dB/dt) (T_2*)^2 / (4 pi),
with T_2* = 1 / (pi x line width in frequency units). Modulation taken as
{B_MOD_NMR_PEAK_T*1e3:.1f} mT peak at {F_MOD_NMR_Hz:.0f} Hz, so dB/dt =
{DBDT_NMR_T_per_s:.3f} T/s at the zero crossing.

| Sample | Width controlling T_2* (mT) | T_2* | N predicted | Reported |
|---|---|---|---|---|
"""

for r in sample_rows:
    summary += (
        f"| {r['name']} | {r['narrow_mT']:.3f} | {r['T2star_s']*1e6:.0f} us | "
        f"{r['n_wiggles']:.2f} | {r['wiggles']} |\n"
    )

summary += f"""

---

## 8. Sanity checks performed on the generated data

| Check | Result | Verdict |
|---|---|---|
| Every ESR coil current inside the stated 0.1 to 1.2 A calibration range | {esr_Imean.min():.3f} to {esr_Imean.max():.3f} A | {'PASS' if esr_Imean.min() >= 0.1 and esr_Imean.max() <= 1.2 else 'FAIL'} |
| Every ESR frequency inside the plug-in coil bands | {esr_nu.min():.0f} to {esr_nu.max():.0f} MHz | {'PASS' if esr_nu.min() >= 15 and esr_nu.max() <= 130 else 'FAIL'} |
| Fitted DPPH g factor within 1 per cent of the accepted value | {100*abs(g_dpph-G_DPPH_LIT)/G_DPPH_LIT:.3f} per cent | {'PASS' if abs(g_dpph-G_DPPH_LIT)/G_DPPH_LIT < 0.01 else 'FAIL'} |
| Extra field recovered from the ESR intercept matches what was injected | {B_extra_at_sample_mT:.4f} mT recovered against {B_extra_expected_mT:.4f} mT expected | {'PASS' if abs(B_extra_at_sample_mT - B_extra_expected_mT) < 0.015 else 'CHECK'} |
| Magnet calibration reduced chi-squared between 0.2 and 2.5 | {chi2r_quad:.2f} | {'PASS' if 0.2 < chi2r_quad < 2.5 else 'CHECK'} |
| ESR calibration reduced chi-squared between 0.3 and 2.5 | {chi2r_cal:.2f} | {'PASS' if 0.3 < chi2r_cal < 2.5 else 'CHECK'} |
| ESR resonance fit reduced chi-squared between 0.3 and 2.5 | {chi2r_esr:.2f} | {'PASS' if 0.3 < chi2r_esr < 2.5 else 'CHECK'} |
| DPPH line width between 0.15 and 0.40 mT | {delta_B0_mT:.3f} mT | {'PASS' if 0.15 < delta_B0_mT < 0.40 else 'FAIL'} |
| DPPH T_2 of order tens of nanoseconds | {T2_s*1e9:.0f} ns | {'PASS' if 20 < T2_s*1e9 < 120 else 'FAIL'} |
| Peak-to-peak modulation field stays below the DC field at 50 MHz | {B_mod_pp_mT:.3f} mT against {B_res_50_mT:.3f} mT | {'PASS' if B_mod_pp_mT < B_res_50_mT else 'FAIL'} |
| Oscilloscope voltages in a range a real instrument would read | U_mod {U_mod_V:.2f} V, delta_U {delta_U_meas_V:.2f} V | {'PASS' if 0.2 < delta_U_meas_V < U_mod_V else 'FAIL'} |
| Every NMR current inside the script's 3.0 to 4.0 A range | {nmr_I_nom.min():.2f} to {nmr_I_nom.max():.2f} A | {'PASS' if nmr_I_nom.min() >= 3.0 and nmr_I_nom.max() <= 4.0 else 'FAIL'} |
| Every NMR frequency inside the oscillator's 15 to 20 MHz band | {min(nu_glycerine.min(), nu_ptfe.min()):.2f} to {max(nu_glycerine.max(), nu_ptfe.max()):.2f} MHz | {'PASS' if min(nu_glycerine.min(), nu_ptfe.min()) >= 15 and max(nu_glycerine.max(), nu_ptfe.max()) <= 20 else 'FAIL'} |
| dB/dI falls monotonically with current (saturation) | {dBdI_at(3.00):.4f} down to {dBdI_at(4.00):.4f} T/A | {'PASS' if dBdI_at(4.00) < dBdI_at(3.00) else 'FAIL'} |
| Hysteresis gap largest at 3.00 A and zero at 4.00 A | {hyst_gap_meas[0]*1e3:.1f} mT down to {hyst_gap_meas[-1]*1e3:.1f} mT | {'PASS' if hyst_gap_meas[0] > hyst_gap_meas[-1] and abs(hyst_gap_meas[-1]) < 3e-3 else 'CHECK'} |
| Straight-line calibration leaves visibly curved residuals | largest residual {np.max(np.abs(resid_lin))*1e3:.2f} mT against {SIG_B_TESLAMETER_T*1e3:.1f} mT scatter, F = {F_lin_quad:.1f} | {'PASS' if F_lin_quad > 4.0 else 'CHECK'} |
| Quadratic calibration leaves flat residuals | largest residual {np.max(np.abs(resid_quad))*1e3:.2f} mT | {'PASS' if np.max(np.abs(resid_quad)) < 3*SIG_B_TESLAMETER_T else 'CHECK'} |
| Quadratic coefficient significantly non-zero | {quad_significance:.1f} sigma | {'PASS' if quad_significance > 3.0 else 'CHECK'} |
| Cubic coefficient consistent with zero | {abs(cubic_coeff)/sig_cubic_coeff:.2f} sigma, F = {F_quad_cub:.2f} | {'PASS' if abs(cubic_coeff)/sig_cubic_coeff < 2.0 else 'CHECK'} |
| NMR proton fit reduced chi-squared between 0.3 and 2.5 | {res_H['chi2r']:.2f} | {'PASS' if 0.3 < res_H['chi2r'] < 2.5 else 'CHECK'} |
| NMR fluorine fit reduced chi-squared between 0.3 and 2.5 | {res_F['chi2r']:.2f} | {'PASS' if 0.3 < res_F['chi2r'] < 2.5 else 'CHECK'} |
| PTFE frequency scatter visibly larger than glycerine scatter | {np.std(res_F['resid'])*1e3:.0f} kHz against {np.std(res_H['resid'])*1e3:.0f} kHz | {'PASS' if np.std(res_F['resid']) > np.std(res_H['resid']) else 'CHECK'} |
| Both g factors agree with literature within the total uncertainty | {res_H['z']:.2f} sigma and {res_F['z']:.2f} sigma | {'PASS' if res_H['z'] < 2 and res_F['z'] < 2 else 'CHECK'} |
| Gradient ratio agrees with 40.078 / 42.577 | {abs(ratio_meas-ratio_lit)/sig_ratio:.2f} sigma | {'PASS' if abs(ratio_meas-ratio_lit)/sig_ratio < 2 else 'CHECK'} |
| Polythene lands on the proton line, not the fluorine line | {poly_vs_H/polythene['sig']:.1f} sigma from 1H, {poly_vs_F/polythene['sig']:.0f} sigma from 19F | {'PASS' if poly_vs_H < poly_vs_F/5 else 'FAIL'} |

Two checks in the original brief do not apply to this experiment: there is no multimeter reading
in the ESR or NMR procedure other than the oscilloscope voltages, which are checked above, and
there is no temperature-dependent mobility measurement, which belongs to the Hall effect
experiment rather than to this one.

---

## 9. Supporting numbers the report may want to quote

| Quantity | Value | Unit |
|---|---|---|
| Boltzmann population excess for DPPH at B_0 = 2.0 mT and 295 K | {np.tanh(G_DPPH_LIT*MU_B*2.0e-3/(2*K_B*295))*1e6:.1f} | parts per million |
| Same for a commercial X-band spectrometer at 0.34 T | {np.tanh(G_DPPH_LIT*MU_B*0.34/(2*K_B*295))*1e6:.0f} | parts per million |
| Boltzmann population excess for protons at 0.43 T and 295 K | {np.tanh(G_PROTON_LIT*MU_N*0.43/(2*K_B*295))*1e6:.2f} | parts per million |
| Ratio of electron to proton splitting at equal field | {(G_ELECTRON_FREE*MU_B)/(G_PROTON_LIT*MU_N):.1f} | dimensionless |
| Same ratio cubed, the induced EMF advantage | {((G_ELECTRON_FREE*MU_B)/(G_PROTON_LIT*MU_N))**3:.3e} | dimensionless |
| Helmholtz field non-uniformity over a 6 mm sample, on axis | {1.152*(3.0/68.0)**4:.2e} | fractional |
| DPPH transverse relaxation rate 1/T_2 | {1/T2_s:.3e} | per second |
| Electron gyromagnetic ratio from the fitted g | {g_dpph*MU_B/HBAR:.4e} | rad / s / T |

---

## 10. File inventory

| File | Contents |
|---|---|
| `reports/data/esr_helmholtz_calibration.csv` | ESR Helmholtz coil calibration, 12 points |
| `reports/data/esr_resonance_frequency.csv` | ESR resonance data, 24 frequencies, 5 repeats each |
| `reports/data/esr_linewidth.csv` | ESR line-width measurement at 50 MHz |
| `reports/data/esr_lineshape.csv` | Representative absorption line shape for the construction figure |
| `reports/data/nmr_magnet_calibration.csv` | Electromagnet B against I, rising and falling branches |
| `reports/data/nmr_resonance_frequency.csv` | Glycerine 1H and PTFE 19F resonance runs |
| `reports/data/nmr_sample_comparison.csv` | Four-sample comparison at 3.50 A |
| `reports/figures/fig_esr_helmholtz_calibration.pdf` | Calibration with residual panel |
| `reports/figures/fig_esr_nu_vs_B.pdf` | Frequency against field, all three coils, global fit, residual panel |
| `reports/figures/fig_esr_linewidth.pdf` | Absorption line with the half-height construction drawn on |
| `reports/figures/fig_nmr_magnet_calibration.pdf` | Both hysteresis branches, branch gap, straight-line versus quadratic residuals |
| `reports/figures/fig_nmr_nu_vs_B.pdf` | 1H and 19F against field, both fits, residual panel |
| `reports/figures/fig_nmr_sample_comparison.pdf` | Representative line shapes and resonance frequencies for the four samples |

**Every table and every data figure derived from these files must be captioned as
representative data.**
"""

with open(os.path.join(DATA_DIR, "results_esr_nmr.md"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(summary)

# ======================================================================================
# CONSOLE SUMMARY
# ======================================================================================

line = "=" * 92


def row(label, fitted, reference=""):
    print(f"{label:<44}{fitted:>24}{reference:>24}")


print(line)
print("REPRESENTATIVE DATA for ESR and NMR, Physics 3H (PHYS 4009).  Seed =", SEED)
print("Simulated from accepted physics. NOT a record of measured results.")
print(line)

print("\nESR")
print("-" * 92)
row("Quantity", "Fitted", "Literature / theory")
print("-" * 92)
row("Helmholtz coil constant (mT/A)", f"{s_fit:.4f} +/- {sig_s_fit:.4f}", f"{S_THEORY_mT_per_A:.4f}")
row("  percentage difference from theory", f"{100*(s_fit-S_THEORY_mT_per_A)/S_THEORY_mT_per_A:+.2f} %")
row("Calibration intercept (mT)", f"{b_fit:+.4f} +/- {sig_b_fit:.4f}", "0")
row("Calibration reduced chi-squared", f"{chi2r_cal:.2f}", "1")
row("d(nu)/d(B0)  (MHz/mT)", f"{m_esr:.3f} +/- {sig_m_esr_stat:.3f}", f"{G_DPPH_LIT*MU_B/H_PLANCK/1e9:.3f}")
row("Resonance fit intercept (MHz)", f"{c_esr:+.3f} +/- {sig_c_esr:.3f}", "0 if no stray field")
row("  extra field at the sample (mT)", f"{B_extra_at_sample_mT:+.4f} +/- {sig_B_extra_at_sample_mT:.4f}",
    f"{B_extra_expected_mT:+.4f} expected")
row("Resonance fit reduced chi-squared", f"{chi2r_esr:.2f}", "1")
row("g_s (DPPH)", f"{g_dpph:.4f} +/- {sig_g_total:.4f}", f"{G_DPPH_LIT:.4f}")
row("  percentage difference from g(DPPH)", f"{100*(g_dpph-G_DPPH_LIT)/G_DPPH_LIT:+.3f} %",
    f"{abs(g_dpph-G_DPPH_LIT)/sig_g_total:.2f} sigma")
row("  percentage difference from free g_e", f"{100*(g_dpph-G_ELECTRON_FREE)/G_ELECTRON_FREE:+.3f} %",
    f"g_e = {G_ELECTRON_FREE:.6f}")
row("Line width delta_B0, FWHM (mT)", f"{delta_B0_mT:.3f} +/- {sig_delta_B0_mT:.3f}", "0.26 to 0.52")
row("Peak-to-peak derivative width (G)", f"{delta_Bpp_mT*10:.2f} +/- {sig_delta_Bpp_mT*10:.2f}", "1.5 to 3.0")
row("Transverse relaxation time T2 (ns)", f"{T2_s*1e9:.0f} +/- {sig_T2_s*1e9:.0f}", "about 62")

print("\nNMR")
print("-" * 92)
row("Quantity", "Fitted", "Literature / theory")
print("-" * 92)
row("Magnet a_0 (T)", f"{a0:.4f} +/- {sig_a0:.4f}", f"{A0_TRUE:.4f} (input)")
row("Magnet a_1 (T/A)", f"{a1:.4f} +/- {sig_a1:.4f}", f"{A1_TRUE:.4f} (input)")
row("Magnet a_2 (T/A^2)", f"{a2:.5f} +/- {sig_a2:.5f}", f"{A2_TRUE:.5f} (input)")
row("  quadratic reduced chi-squared", f"{chi2r_quad:.2f}", "1")
row("  straight-line reduced chi-squared", f"{chi2r_lin_e:.2f}", "much greater than 1")
row("  cubic coefficient / its uncertainty", f"{abs(cubic_coeff)/sig_cubic_coeff:.2f} sigma", "consistent with 0")
row("dB/dI at 3.00 A (T/A)", f"{dBdI_at(3.00):.4f}")
row("dB/dI at 4.00 A (T/A)", f"{dBdI_at(4.00):.4f}")
row("  fall in dB/dI across the range", f"{100*(dBdI_at(3.00)-dBdI_at(4.00))/dBdI_at(3.00):.1f} %")
row("Hysteresis gap at 3.00 A (mT)", f"{hyst_gap_meas[0]*1e3:.1f}", f"{HYST_GAP_AT_3A_T*1e3:.1f} (input)")
row("d(nu)/d(B0), 1H glycerine (MHz/T)", f"{res_H['m']:.2f} +/- {res_H['sig_m_stat']:.2f}", f"{GAMMA2PI_H:.3f}")
row("d(nu)/d(B0), 19F PTFE (MHz/T)", f"{res_F['m']:.2f} +/- {res_F['sig_m_stat']:.2f}", f"{GAMMA2PI_F:.3f}")
row("g (1H, glycerine)", f"{res_H['g']:.3f} +/- {res_H['sig_g_total']:.3f}", f"{G_PROTON_LIT:.4f}")
row("  percentage difference", f"{res_H['pct_diff']:+.2f} %", f"{res_H['z']:.2f} sigma")
row("g (19F, PTFE)", f"{res_F['g']:.3f} +/- {res_F['sig_g_total']:.3f}", f"{G_FLUORINE_LIT:.4f}")
row("  percentage difference", f"{res_F['pct_diff']:+.2f} %", f"{res_F['z']:.2f} sigma")
row("Gradient ratio 19F / 1H", f"{ratio_meas:.4f} +/- {sig_ratio:.4f}", f"{ratio_lit:.5f}")
row("Polythene frequency at 3.50 A (MHz)", f"{polythene['nu']:.2f} +/- {polythene['sig']:.2f}",
    f"1H {pred_H_fixed:.2f} / 19F {pred_F_fixed:.2f}")

print("\n" + line)
print("Files written")
print(line)
for f in sorted(os.listdir(DATA_DIR)):
    print("  data/   ", f)
for f in sorted(os.listdir(FIG_DIR)):
    print("  figures/", f)
print(line)
print("Reminder: representative data. Not a record of measured results.")
print(line)
