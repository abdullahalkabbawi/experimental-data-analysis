# Results, Hall effect in a two-dimensional electron gas

**REPRESENTATIVE DATA. Not a record of measured results.** Every number in this
file was produced by `reports/analysis/generate_hall.py` from simulated readings
built forward from physically accepted parameters, with realistic instrument
scatter and realistic systematic offsets added. The random seed is fixed at
`SEED = 4009`, so the whole file is reproducible.

All fitted values below are what the least squares fits actually returned when run
on the generated data. They are not the ground truth values that were fed in. In
particular the Hall voltages were generated using the TRUE magnet field, while the
analysis uses the field read off the CALIBRATION POLYNOMIAL, so the calibration
error propagates into the answer exactly as it would in the laboratory. The ground
truth is listed alongside in section 9.1 so the difference can be seen.

---

## 0. How to read this file

- Every quantity carries its unit explicitly.
- `stat` means the statistical uncertainty from the fit covariance matrix alone.
- `total` means statistical combined in quadrature with the systematic terms of
  section 8.
- Percentage difference is quoted as (fitted - reference) / reference x 100.
- Weighting policy: every fit weight contains only the per-point random errors,
  that is the voltmeter scatter with the current reading scatter folded in through
  the local slope. The magnet calibration uncertainty is deliberately kept out of
  the weights, because it is one correlated systematic shared by all the points at
  a given field setting and not an independent random error. It is carried into the
  budget in section 8 instead.

---

## 1. Electromagnet calibration

Data file: `data/hall_magnet_calibration.csv` (50 rows, 25 rising and 25 falling).
Figure: `figures/fig_hall_magnet_calibration.pdf`.

Magnet supply swept 0.0 V to 24.0 V in 1.0 V steps into a coil of 10.2 ohm, so the
coil current ran from 0 A to 2.353 A. Field read with a calibrated Hall probe
magnetometer with the sample stand removed. The sweep was taken upward and then
downward through the same settings so the hysteresis loop is visible.

Assumed uncertainty on one calibration point: Hall probe repeatability 2.5 mT,
coil current reading 5.0 mA folded in through the local slope, and branch
reproducibility 2.0 mT. Combined, 3.25 mT to 3.36 mT across the range.

### 1.1 Cubic fit to the rising branch

Model, weighted least squares with weights 1/sigma_B:

    B(I_m) = c3*I_m^3 + c2*I_m^2 + c1*I_m + c0   [B in T, I_m in A]

| Coefficient | Fitted value | Standard error | Unit | True value used to generate |
|---|---|---|---|---|
| c3 | +0.004361 | 0.002414 | T A^-3 | +0.004000 |
| c2 | -0.034490 | 0.008665 | T A^-2 | -0.035000 |
| c1 | +0.202574 | 0.008707 | T A^-1 | +0.205000 |
| c0 | +0.002563 | 0.002335 | T | +0.001500 |

- chi-square = 17.23 on 21 degrees of freedom, chi2/dof = **0.82**
- RMS residual = **2.74 mT**, against an assumed point uncertainty of about 3.3 mT
- The fitted constant term c0 = 2.56 mT is the remanent field left in the iron core
  after the previous cycle. The value used to generate the data was 1.50 mT.

### 1.2 Why the fit has to be non-linear

| Polynomial order | chi2 | dof | chi2/dof | RMS residual (mT) |
|---|---|---|---|---|
| 1 (straight line) | 185.1 | 23 | 8.0 | 8.98 |
| 2 (quadratic) | 20.5 | 22 | 0.93 | 2.99 |
| 3 (cubic) | 17.2 | 21 | 0.82 | 2.74 |

The iron yoke approaches magnetic saturation, so an increasing share of the
magnetomotive force is spent driving flux through the iron rather than across the
gap and the field grows more slowly than the coil current at the top of the range.
A straight line fitted to these data over-reads the ADOPTED CUBIC at the top of the
sweep by **14 mT**, which is **4 per cent** of the calibrated field there. That
error would go straight into R_H,2D and hence into n_sq, so the non-linear fit is
not a refinement, it is essential. (Against the ground truth, which the experimenter
cannot know and which therefore must not appear in the report, the straight line
over-reads by 17 mT.)

The quadratic already fits: chi2/dof = 0.93 on 22 degrees of freedom, and going to
the cubic buys only delta chi2 = 3.3 for one more parameter, so the cubic
coefficient c3 is only 1.8 standard errors from zero. The cubic is kept because the
curve is known on physical grounds to bend over, and because the two models differ
by at most 2.52 mT at the five working set points, inside the uncertainty on B there.

### 1.3 Hysteresis

| Quantity | Value | Unit |
|---|---|---|
| Mean separation of the two measured branches | 5.52 | mT |
| Separation of the two fitted cubics at zero coil current | 6.35 | mT |
| Largest separation of the two fitted cubics | 8.72 | mT |
| Field at zero coil current, rising branch (fitted) | 2.56 | mT |
| Field at zero coil current, falling branch (fitted) | 8.92 | mT |

Value used to generate the data: branch separation 8.0 mT extrapolated to zero
coil current, falling to zero at the turning point.

The procedure adopted for every later measurement was to approach each set point
from below after cycling the magnet to maximum and back, so that the field history
was at least the same from run to run. Note that this does NOT put calibration and
measurement on the same branch: a down-cycle leaves the core at the falling-branch
remanence of 8.92 mT, not at the 2.56 mT of the calibration rising branch, so the
measurement traces a minor loop between the two. The resulting branch offset, about
6 mT at low coil current and falling towards zero at the top of the sweep, is
carried as part of the systematic uncertainty on B in section 8. Only demagnetising
the core before each run would remove it.

The difference of the two fitted cubics peaks above its zero-current value, which a
closed loop cannot do; that is an artefact of differencing two independent cubic
fits, not a feature of the loop. The measured mean separation is the figure used.

### 1.4 The five working field settings

| Magnet supply (V) | Coil current (A) | B from the cubic (T) | sigma_B stat (mT) | True B (T) | Calibration error (mT) |
|---|---|---|---|---|---|
| 6.0 | 0.588 | 0.1107 | 2.47 | 0.1108 | -0.11 |
| 11.0 | 1.078 | 0.1864 | 2.35 | 0.1869 | -0.51 |
| 16.0 | 1.569 | 0.2523 | 2.39 | 0.2524 | -0.09 |
| 20.5 | 2.010 | 0.3058 | 2.36 | 0.3046 | +1.18 |
| 24.0 | 2.353 | 0.3451 | 3.09 | 0.3422 | +2.88 |

Peak field 0.3451 T, inside the 0.3 to 0.4 T range expected of a teaching-scale
iron-cored electromagnet. The calibration error column is not available to the
experimenter. It is printed here only so that the honesty of this pipeline can be
checked: the largest error is 2.88 mT, or 0.84 per cent, and it is what limits how
closely the fitted n_sq can approach the true value.

---

## 2. Hall measurement at 300 K (room temperature)

Data file: `data/hall_VH_300K.csv` (90 rows: 18 bar-supply steps at each of 5
magnet settings). Bar supply swept 3.0 V to 27.0 V across the Hall bar in
series with the two 150 kohm protection resistors.

Measured ranges:

- sample current I: **9.69 uA to 88.75 uA** (software cap 100 uA)
- Hall voltage V_H: **1.804 mV to 39.110 mV**
- product I*B: **1.073 to 30.543 uA T**

### 2.1 Linearity in I at fixed B

A weighted straight line V_H = s_j * I + b_j was fitted at each field setting.

| B (T) | slope s_j (ohm) | sigma_s_j (ohm) | intercept b_j (mV) | sigma_b_j (mV) | chi2/dof |
|---|---|---|---|---|---|
| 0.1107 | 149.8 | 0.5 | +0.3220 | 0.0285 | 0.95 |
| 0.1864 | 244.8 | 0.6 | +0.2934 | 0.0312 | 0.71 |
| 0.2523 | 327.3 | 0.7 | +0.2645 | 0.0341 | 2.17 |
| 0.3058 | 393.9 | 0.7 | +0.1862 | 0.0368 | 0.51 |
| 0.3451 | 438.9 | 0.8 | +0.2876 | 0.0388 | 0.92 |

Every line is straight within the scatter. That is objective 2 for the current
dependence. The intercepts are all small, between +0.186 mV and +0.322 mV, and
they do not grow with field. That is the first sign of a field-independent
offset on the Hall contact pair.

### 2.2 Linearity in B: the slopes against field

The five slopes were then fitted against B, weighted, as s(B) = m_H * B + k:

- m_H = R_H,2D = **1241.0 +/- 3.4 ohm T^-1** (statistical)
- k = **13.08 +/- 0.79 ohm**, the field-independent longitudinal leak
- chi2/dof = 5.64 on 3 degrees of freedom
- RMS residual on the slopes = 1.35 ohm

The chi2/dof of this second-stage fit is above 1 because the weights carry only
the statistical error on each slope. The field values themselves come from one
shared calibration polynomial, so their errors are correlated and are handled in
the budget of section 8 rather than in the weights. The same remark applies to
R3 below.

The slopes are linear in B. That is objective 2 for the field dependence. Taken
with the previous table, V_H is shown to be a single linear function of the
product I*B rather than of I and B separately.

### 2.3 The three reductions of the whole array (objective 6)

| Reduction | Model | R_H,2D (ohm T^-1) | sigma stat (ohm T^-1) | chi2 | dof | chi2/dof |
|---|---|---|---|---|---|---|
| R1, slope of slopes | s_j = m_H B_j + k | 1241.0 | 3.4 | 16.9 | 3 | 5.64 |
| R2, naive global line | V_H = m_H (IB) + c | 1273.8 | 1.0 | 1249.6 | 88 | 14.20 |
| R3, global with offset term | V_H = m_H (IB) + k I + c | **1234.9** | **1.5** | 131.3 | 87 | **1.51** |

**R3 is the value adopted for the report.** The reasoning:

- R2 forces the contact-misalignment voltage, which is proportional to I and
  independent of B, into a single constant. That voltage is correlated with the
  abscissa I*B, so it biases the slope. R2 returns 1273.8 ohm T^-1 against the R3
  value of 1234.9 ohm T^-1, a difference of +3.15 per cent, and its chi2/dof of
  14.2 says plainly that the two-parameter model does not describe the data.
- R1 and R3 agree to +0.49 per cent. R3 is preferred only because it uses every
  point in a single weighted fit rather than fitting slopes to slopes.
- Fitted misalignment term k = **14.41 +/- 0.43 ohm** (R3). The value used to
  generate the data was 12.48 ohm, from a contact displacement of 0.80 um on a
  100 um wide bar. The fitted k sits 4.5 standard errors away from that
  value. The reason is that k is strongly correlated with m_H in this fit: the
  correlation coefficient from the covariance matrix is -0.75. Any error in the
  field calibration therefore leaks into k as well as into m_H, and the quoted
  standard error on k, which is statistical only, understates how well k is
  really known. k should be read as a nuisance parameter that soaks up a
  field-independent voltage, not as a precision measurement of the contact
  geometry.
- Fitted constant term c = **+0.2774 +/- 0.0147 mV** (R3). The value used to
  generate the data was +0.250 mV.
- Weighted residual RMS: R2 0.2421 mV, R3 0.0853 mV.

A naive alternative, computing R_H,2D = V_H/(I*B) point by point and averaging,
is worse still. It weights the smallest I*B points, which have the worst signal
to noise, equally with the best, and the constant offset contributes c/(I*B) to
every ratio, a contribution that diverges as I*B goes to zero. For these data
that unweighted average gives 1346.2 ohm T^-1, which is +9.0 per cent away from
the R3 value.

Residuals were examined against I*B and against B. With R2 they show clear
structure grouped by field setting; with R3 that structure is largely gone. The
residual scatter that remains in R3 is set by the magnet calibration, which is a
correlated systematic and is not in the weights, so a chi2/dof of exactly one is
not expected. See the lower panel of `figures/fig_hall_VH_vs_IB.pdf` for the
300 K case.

Value used to generate the data, R_H,2D at 300 K = **1248.3 ohm T^-1**. The adopted
R3 value differs from it by **-1.07 per cent**, which is inside the total
uncertainty of 3.9 per cent quoted in section 8.

---

## 3. Longitudinal resistance and resistance per square

Data files: `data/hall_VL_300K.csv`, `data/hall_VL_77K.csv` (18 rows each).
Figures: `figures/fig_hall_VL_vs_I.pdf`, `figures/fig_hall_77K_comparison.pdf`.

Measured between contacts 1 and 4 with the electromagnet off (emVolts = [0]).
Measuring at zero field removes magnetoresistance, removes any Hall cross-talk into
the longitudinal pair, and keeps the magnet calibration out of the R_sq error
budget entirely. The four-terminal arrangement also means the contact resistances
and the two 150 kohm protection resistors do not enter the result at all, because
the current is measured directly rather than inferred from the supply voltage.

The active region is 300 um long between the voltage probes and 100 um wide, so
it is exactly **3 squares** and R_sq = R_L / 3.

| Quantity | 300 K | 79 K | Unit |
|---|---|---|---|
| Current range | 9.73 to 88.56 | 10.07 to 89.81 | uA |
| V_L range | 46.18 to 415.85 | 3.58 to 31.86 | mV |
| Fitted slope m_L = R_L | 4682.8 +/- 6.1 | 353.63 +/- 0.69 | ohm |
| Fitted intercept c_L | +0.3559 +/- 0.2999 | +0.1112 +/- 0.0352 | mV |
| chi2/dof | 0.95 | 0.78 | |
| **R_sq = m_L / 3** | **1560.9 +/- 2.0 (stat)** | **117.88 +/- 0.23 (stat)** | ohm/square |
| R_sq, total uncertainty | +/- 15.7 | +/- 1.20 | ohm/square |
| Value used to generate R_L | 4681.1 | 354.63 | ohm |
| Value used to generate R_sq | 1560.4 | 118.21 | ohm/square |
| Difference from that value | +0.03 | -0.28 | per cent |

R_sq falls by a factor of **13.2** on cooling, while R_H,2D changes by only
**13.9 per cent**. That asymmetry is the fingerprint of modulation doping: the
carrier density is set by charge transfer equilibrium across the heterojunction,
which involves energies of tens to hundreds of meV and so barely moves between
300 K and 77 K, while the mobility is set by scattering rates and the dominant
channel at 300 K, polar optical phonon scattering, freezes out on cooling.

A further consequence worth noting. At 79 K the Hall voltage at the top of the sweep,
43.9 mV, is **larger** than the longitudinal voltage, 31.9 mV, because mu*B = 4.1
exceeds l/w = 3. At 300 K the reverse holds: V_H reaches only 39.1 mV against a V_L
of 415.8 mV. That inversion is a direct visual signature of the high mobility.

---

## 4. Angular dependence of the Hall voltage

Data file: `data/hall_angle_300K.csv` (25 rows, 0 to 360 degrees in 15 degree
steps). Figure: `figures/fig_hall_angle.pdf`.

Taken at 300 K with the magnet supply fixed at 20.5 V (calibrated B = 0.3058 T) and
the bar supply fixed at 18.3 V, giving a mean sample current of 60.07 uA.

Only the component of B normal to the 2DEG plane produces a Hall voltage, because
the carriers are confined and cannot acquire an out-of-plane velocity. The fit
model is the script's own cosine, extended by a constant offset term:

    V_H(theta) = V_H0 * cos(theta - theta_0) + C

| Parameter | Fitted value | Standard error | Unit | Value used to generate |
|---|---|---|---|---|
| V_H0 | 22.8551 | 0.0161 | mV | 22.8383 |
| theta_0 | 6.07 | 0.04 | degrees | 6.0 |
| C | 1.0025 | 0.0111 | mV | 0.9998 |

- chi2 = 28.34 on 22 degrees of freedom, chi2/dof = **1.29**
- RMS residual = **0.0582 mV**, against an assumed meter scatter of 0.050 mV
- The amplitude predicted from the independently fitted R_H,2D at 300 K and the
  measured I and B is **22.6859 mV**, against a fitted **22.8551 mV**, a difference of
  **+0.75 per cent**. The angular scan is therefore an independent confirmation of
  the Hall coefficient obtained from the current and field sweeps.
- theta_0 = 6.07 degrees is a mounting misalignment between the zero of the
  rotation stage and the true normal orientation. It is not a property of the
  sample. Because the cosine is flat near its maximum, a measurement taken at a
  nominal zero without correcting for it under-reads the normal field component by
  only 0.56 per cent.
- C = 1.0025 mV is field-independent, and it is consistent in size with the offset
  found independently in the global Hall fit, which predicts k*I + c = 1.1430 mV at
  this current (k*I = 0.8656 mV plus c = 0.2774 mV). The two differ by -12.3 per cent.
  That is the level of agreement to expect given how strongly k correlates with
  the slope in the global fit.
- The cosine term reverses sign at theta_0 + 180 degrees, but the measured voltage
  does not reverse with equal magnitude, because the offset C does not change sign:
  the readings at 0 and 180 degrees are +23.6741 mV and -21.7516 mV, and their sum, 1.9224 mV,
  is 2C. Half the DIFFERENCE of two opposite angles removes the offset and returns
  the pure Hall term; half their SUM returns the offset. That is the same parity
  argument as reversing the field, and the reversal itself confirms the signal is a
  genuine Hall voltage rather than an instrumental artefact.

---

## 5. Hall measurement at 79 K (liquid nitrogen cooling)

Data file: `data/hall_VH_77K.csv` (90 rows: 18 bar-supply steps at each of 5
magnet settings). Bar supply swept 3.0 V to 27.0 V across the Hall bar in
series with the two 150 kohm protection resistors.

Measured ranges:

- sample current I: **9.71 uA to 90.11 uA** (software cap 100 uA)
- Hall voltage V_H: **1.710 mV to 43.875 mV**
- product I*B: **1.075 to 31.033 uA T**

### 5.1 Linearity in I at fixed B

A weighted straight line V_H = s_j * I + b_j was fitted at each field setting.

| B (T) | slope s_j (ohm) | sigma_s_j (ohm) | intercept b_j (mV) | sigma_b_j (mV) | chi2/dof |
|---|---|---|---|---|---|
| 0.1107 | 157.9 | 0.5 | +0.1080 | 0.0287 | 0.74 |
| 0.1864 | 265.4 | 0.6 | +0.1519 | 0.0319 | 1.19 |
| 0.2523 | 359.9 | 0.7 | +0.0639 | 0.0355 | 1.14 |
| 0.3058 | 434.1 | 0.8 | +0.0481 | 0.0386 | 0.95 |
| 0.3451 | 488.7 | 0.8 | -0.0108 | 0.0411 | 0.96 |

Every line is straight within the scatter. That is objective 2 for the current
dependence. The intercepts are all small, between -0.011 mV and +0.152 mV, and
they do not grow with field. That is the first sign of a field-independent
offset on the Hall contact pair.

### 5.2 Linearity in B: the slopes against field

The five slopes were then fitted against B, weighted, as s(B) = m_H * B + k:

- m_H = R_H,2D = **1414.3 +/- 3.5 ohm T^-1** (statistical)
- k = **1.75 +/- 0.80 ohm**, the field-independent longitudinal leak
- chi2/dof = 1.92 on 3 degrees of freedom
- RMS residual on the slopes = 0.78 ohm

The chi2/dof of this second-stage fit is above 1 because the weights carry only
the statistical error on each slope. The field values themselves come from one
shared calibration polynomial, so their errors are correlated and are handled in
the budget of section 8 rather than in the weights. The same remark applies to
R3 below.

The slopes are linear in B. That is objective 2 for the field dependence. Taken
with the previous table, V_H is shown to be a single linear function of the
product I*B rather than of I and B separately.

### 5.3 The three reductions of the whole array (objective 6)

| Reduction | Model | R_H,2D (ohm T^-1) | sigma stat (ohm T^-1) | chi2 | dof | chi2/dof |
|---|---|---|---|---|---|---|
| R1, slope of slopes | s_j = m_H B_j + k | 1414.3 | 3.5 | 5.8 | 3 | 1.92 |
| R2, naive global line | V_H = m_H (IB) + c | 1415.8 | 1.1 | 212.1 | 88 | 2.41 |
| R3, global with offset term | V_H = m_H (IB) + k I + c | **1406.4** | **1.6** | 150.9 | 87 | **1.73** |

**R3 is the value adopted for the report.** The reasoning:

- R2 forces the contact-misalignment voltage, which is proportional to I and
  independent of B, into a single constant. That voltage is correlated with the
  abscissa I*B, so it biases the slope. R2 returns 1415.8 ohm T^-1 against the R3
  value of 1406.4 ohm T^-1, a difference of +0.67 per cent, and its chi2/dof of
  2.4 says plainly that the two-parameter model does not describe the data.
- R1 and R3 agree to +0.56 per cent. R3 is preferred only because it uses every
  point in a single weighted fit rather than fitting slopes to slopes.
- Fitted misalignment term k = **3.44 +/- 0.44 ohm** (R3). The value used to
  generate the data was 0.95 ohm, from a contact displacement of 0.80 um on a
  100 um wide bar. The fitted k sits 5.7 standard errors away from that
  value. The reason is that k is strongly correlated with m_H in this fit: the
  correlation coefficient from the covariance matrix is -0.74. Any error in the
  field calibration therefore leaks into k as well as into m_H, and the quoted
  standard error on k, which is statistical only, understates how well k is
  really known. k should be read as a nuisance parameter that soaks up a
  field-independent voltage, not as a precision measurement of the contact
  geometry.
- Fitted constant term c = **+0.0844 +/- 0.0153 mV** (R3). The value used to
  generate the data was +0.120 mV.
- Weighted residual RMS: R2 0.1148 mV, R3 0.0942 mV.

A naive alternative, computing R_H,2D = V_H/(I*B) point by point and averaging,
is worse still. It weights the smallest I*B points, which have the worst signal
to noise, equally with the best, and the constant offset contributes c/(I*B) to
every ratio, a contribution that diverges as I*B goes to zero. For these data
that unweighted average gives 1436.3 ohm T^-1, which is +2.1 per cent away from
the R3 value.

Residuals were examined against I*B and against B. With R2 they show clear
structure grouped by field setting; with R3 that structure is largely gone. The
residual scatter that remains in R3 is set by the magnet calibration, which is a
correlated systematic and is not in the weights, so a chi2/dof of exactly one is
not expected. See the lower panel of `figures/fig_hall_VH_vs_IB.pdf` for the
300 K case.

Value used to generate the data, R_H,2D at the cryogenic temperature = **1418.5 ohm T^-1**. The adopted
R3 value differs from it by **-0.86 per cent**, which is inside the total
uncertainty of 3.9 per cent quoted in section 8.

---

## 6. Temperature during the cryogenic run

Data file: `data/hall_thermocouple_77K.csv` (12 type K readings taken on the sample
mount through the run).

| Quantity | Value | Unit |
|---|---|---|
| Mean reading | 78.69 | K |
| Spread of readings (standard deviation) | 1.11 | K |
| Standard error on the mean | 0.32 | K |
| Type K accuracy floor below 200 K | 2.5 | K |
| **Quoted temperature** | **79 +/- 3** | K |
| Value used to generate the readings | 78.6 | K |
| Liquid nitrogen boiling point at 1 atm | 77.4 | K |

The thermocouple bead is on the mount rather than on the 2DEG itself, and the
sample sits above the liquid in a pumped dewar, so a reading a few kelvin above the
bath temperature is expected. The sensitivity of a type K junction falls from about
41 uV/K at 300 K to roughly 16 uV/K at 77 K, which is why the low-temperature
reading is the less precise one.

Neither n_sq nor mu is calculated from T, so a temperature error does not propagate
into the results. It matters only for the comparison with literature values quoted
at a nominal 77 K.

---

## 7. Headline results

R_H,2D comes from the R3 global weighted fit; R_sq comes from the zero-field
longitudinal fit. Formulae used:

    n_sq = 1 / (e * R_H,2D)
    R_sq = m_L / 3
    mu   = R_H,2D / R_sq = 3 * m_H / m_L

with e = 1.602176634e-19 C.

| Quantity | 300 K | 79 K | Unit |
|---|---|---|---|
| R_H,2D, statistical only | 1234.9 +/- 1.5 | 1406.4 +/- 1.6 | ohm T^-1 |
| R_H,2D, total | 1234.9 +/- 47.6 | 1406.4 +/- 54.2 | ohm T^-1 |
| R_L | 4682.8 +/- 6.1 | 353.63 +/- 0.69 | ohm |
| R_sq, total | 1560.9 +/- 15.7 | 117.88 +/- 1.20 | ohm/square |
| **n_sq** | **5.054 +/- 0.195** | **4.438 +/- 0.171** | 10^15 m^-2 |
| n_sq, statistical only | 5.054 +/- 0.006 | 4.438 +/- 0.005 | 10^15 m^-2 |
| n_sq in cm^-2 | 5.054 +/- 0.195 | 4.438 +/- 0.171 | 10^11 cm^-2 |
| **mu** | **0.7912 +/- 0.0315** | **11.93 +/- 0.48** | m^2 V^-1 s^-1 |
| mu, statistical only | 0.7912 +/- 0.0014 | 11.93 +/- 0.03 | m^2 V^-1 s^-1 |
| mu in cm^2 V^-1 s^-1 | 7912 +/- 315 | 119311 +/- 4755 | cm^2 V^-1 s^-1 |
| n_sq * mu, free of magnet error | 3.9986 +/- 0.0403 | 52.950 +/- 0.539 | 10^15 V^-1 s^-1 (equivalently C^-1 ohm^-1) |

Note: 10^15 m^-2 is exactly 10^11 cm^-2, so the two density rows carry the same
digits. Both are given because the literature quotes cm^-2.

### 7.1 Change on cooling

| Quantity | 300 K | 79 K | Change |
|---|---|---|---|
| n_sq (10^15 m^-2) | 5.054 | 4.438 | -12.2 per cent |
| mu (m^2 V^-1 s^-1) | 0.7912 | 11.93 | factor 15.1 increase |
| R_sq (ohm/square) | 1560.9 | 117.88 | factor 13.2 decrease |
| R_H,2D (ohm T^-1) | 1234.9 | 1406.4 | +13.9 per cent |

Uncertainty on the mobility ratio, combining the two statistical uncertainties in
quadrature and noting that the magnet and geometry systematics largely cancel in
the ratio: **15.08 +/- 0.04**.

---

## 8. Uncertainty budget

| Contribution | Size | Affects |
|---|---|---|
| Fit standard error on m_H, 300 K | 0.13 per cent | n_sq, mu |
| Fit standard error on m_H, 79 K | 0.11 per cent | n_sq, mu |
| Fit standard error on m_L, 300 K | 0.13 per cent | R_sq, mu |
| Fit standard error on m_L, 79 K | 0.19 per cent | R_sq, mu |
| Hall probe absolute accuracy and positioning in the gap | 3.0 per cent | n_sq, mu |
| Field floor term, 5 mT over a mean field of 0.240 T | 2.1 per cent | n_sq, mu |
| Statistical uncertainty of the calibration polynomial | 1.2 per cent | n_sq, mu |
| **Combined magnet systematic on B** | **3.9 per cent** | n_sq, mu, not R_sq |
| Geometry, l/w from etch bias and mesa edge depletion | 1.0 per cent | R_sq, mu, not n_sq |
| **Combined on n_sq, 300 K** | **3.9 per cent** | |
| **Combined on n_sq, 79 K** | **3.9 per cent** | |
| **Combined on R_sq, 300 K** | **1.0 per cent** | |
| **Combined on R_sq, 79 K** | **1.0 per cent** | |
| **Combined on mu, 300 K** | **4.0 per cent** | |
| **Combined on mu, 79 K** | **4.0 per cent** | |

Three points worth making in the discussion:

- The magnet calibration dominates the error on n_sq and on mu. Improving the
  voltmeter would not help; only a better field measurement would.
- R_sq carries no magnet uncertainty at all, because it was measured at zero field.
- The product n_sq * mu = 1 / (e * R_sq) is fixed by the zero-field longitudinal
  measurement alone and is immune to the magnet calibration. It is known to
  1.0 per cent at 300 K and 1.0 per cent at 79 K, far better than either factor
  separately. n_sq and mu share the factor m_H, so their errors are correlated and
  the magnet systematic pushes them in opposite directions.

Not included in the budget above, and stated here as known limitations:

- The Hall scattering factor is taken as exactly 1. At 79 K the gas is partly
  degenerate (T/T_F = 0.43) and that is a reasonable assumption. At 300 K the gas
  is not degenerate (T/T_F = 1.43), so a Hall scattering factor a few per cent
  above unity is possible. Since R_H,2D = r_H / (n_sq e), that would make the true
  300 K density a few per cent HIGHER than quoted, not lower, which is the opposite
  direction to parallel conduction below.
- Any parallel conduction in the doped AlGaAs layer at 300 K would raise the
  apparent density and lower the apparent mobility. It freezes out on cooling.

---

## 9. Comparison with the generating values and with the literature

### 9.1 Against the values used to generate the data

This table exists so that the honesty of the pipeline can be checked. The
generating values were chosen first, from the literature; the fitted values are
what the analysis returned without ever seeing them.

| Quantity | Generating value | Fitted | Difference | Quoted uncertainty | Unit |
|---|---|---|---|---|---|
| n_sq, 300 K | 5.000 | 5.054 | +1.08 per cent | +/- 3.9 per cent | 10^15 m^-2 |
| n_sq, 79 K | 4.400 | 4.438 | +0.86 per cent | +/- 3.9 per cent | 10^15 m^-2 |
| mu, 300 K | 0.8000 | 0.7912 | -1.10 per cent | +/- 4.0 per cent | m^2 V^-1 s^-1 |
| mu, 79 K | 12.00 | 11.93 | -0.57 per cent | +/- 4.0 per cent | m^2 V^-1 s^-1 |
| R_sq, 300 K | 1560.4 | 1560.9 | +0.03 per cent | +/- 1.0 per cent | ohm/square |
| R_sq, 79 K | 118.21 | 117.88 | -0.28 per cent | +/- 1.0 per cent | ohm/square |
| R_H,2D, 300 K | 1248.3 | 1234.9 | -1.07 per cent | +/- 3.9 per cent | ohm T^-1 |
| R_H,2D, 79 K | 1418.5 | 1406.4 | -0.86 per cent | +/- 3.9 per cent | ohm T^-1 |

Every fitted value sits inside its own quoted uncertainty, which is the behaviour a
correctly propagated error budget should show.

### 9.2 Against the literature

| Quantity | This work (representative) | Literature | Source | Difference |
|---|---|---|---|---|
| n_sq, 300 K (10^11 cm^-2) | 5.05 +/- 0.19 | 5.6 | Seliuta et al., arXiv:0711.0474, structure 2DEG-B | -9.7 per cent |
| mu, 300 K (cm^2 V^-1 s^-1) | 7912 +/- 315 | 8000 | same | -1.1 per cent |
| n_sq, 79 K (10^11 cm^-2) | 4.44 +/- 0.17 | 4.3 | same | +3.2 per cent |
| mu, 79 K (cm^2 V^-1 s^-1) | 119311 +/- 4755 | 116000 | same | +2.9 per cent |
| mu at 77 K (cm^2 V^-1 s^-1) | 119311 +/- 4755 | 117000 | Stormer, Dingle, Gossard, Wiegmann and Sturge, Solid State Commun. 29, 705 (1979) | +2.0 per cent |
| R_sq, 300 K (ohm/square) | 1561 +/- 16 | about 2000 | typical GaAs/AlGaAs 2DEG at 300 K, HEMT characterisation literature | -22.0 per cent |
| mu, 300 K, against bulk undoped GaAs (m^2 V^-1 s^-1) | 0.791 +/- 0.032 | 0.85 | Ioffe NSM database, electrical properties of GaAs | -6.9 per cent |
| mu(cold) / mu(300 K) | 15.1 | 14.5 | Seliuta et al., 2DEG-B | +4.0 per cent |
| Electron effective mass used | 0.067 m_0 | 0.067 m_0 | Vurgaftman, Meyer and Ram-Mohan, J. Appl. Phys. 89, 5815 (2001) | assumed, not fitted |
| Sample geometry l/w | 3.0 squares | 3.0 squares | the departmental script | given, not fitted |

---

## 10. Secondary quantities derived from the fitted results

Computed with m* = 0.067 m_0 = 6.1033e-32 kg. These need no extra measurement.

| Quantity | Formula | 300 K | 79 K | Unit |
|---|---|---|---|---|
| Momentum relaxation time tau | mu m* / e | 0.301 | 4.55 | ps |
| Fermi wavevector k_F | sqrt(2 pi n_sq) | 1.782 | 1.670 | 10^8 m^-1 |
| Fermi velocity v_F | hbar k_F / m* | 3.079 | 2.885 | 10^5 m s^-1 |
| Mean free path | v_F tau | 0.093 | 1.311 | um |
| Fermi energy above the subband edge | pi hbar^2 n_sq / m* | 18.06 | 15.86 | meV |
| Fermi temperature T_F | E_F / k_B | 210 | 184 | K |
| omega_c tau = mu B at B = 0.345 T | mu B | 0.273 | 4.12 | dimensionless |
| 2D density of states including spin | m* / (pi hbar^2) | 2.799 | 2.799 | 10^13 eV^-1 cm^-2 |

Both mean free paths are far shorter than the 300 um probe separation, so
transport is diffusive and the Drude picture applies. At 79 K, T / T_F = 0.43, so the
gas is partly degenerate and taking the Hall scattering factor as 1 is reasonable.
At 300 K, T / T_F = 1.43, so the gas is not degenerate at all. The Fermi quantities
in the table above are zero-temperature expressions, so their 300 K entries are
indicative only: the thermal velocity sqrt(2 k_B T / m*) = 3.68e5 m/s gives a
room-temperature mean free path of 0.111 um, which does not change the conclusion.

The 2D density of states is independent of energy, which is the single most
characteristic signature of two-dimensionality.

Landau level spacing at the peak field: hbar*omega_c = 0.596 meV, against
k_B T = 6.81 meV at 79 K and 25.85 meV at 300 K. Thermal broadening exceeds the level
spacing by a factor of 11 even at the cryogenic temperature, so no Shubnikov-de
Haas oscillations and no quantum Hall plateaux are expected. No continuous field
sweep of the longitudinal resistance was taken in any case, and R_sq was measured
only at zero field, so these data could not have shown such oscillations.

---

## 11. Files produced

| File | Contents |
|---|---|
| `data/hall_magnet_calibration.csv` | electromagnet calibration, rising and falling branches |
| `data/hall_VH_300K.csv` | Hall voltage against current at five fields, 300 K |
| `data/hall_VL_300K.csv` | longitudinal voltage against current at zero field, 300 K |
| `data/hall_angle_300K.csv` | Hall voltage against angle over 360 degrees, 300 K |
| `data/hall_VH_77K.csv` | Hall voltage against current at five fields, cryogenic |
| `data/hall_VL_77K.csv` | longitudinal voltage against current at zero field, cryogenic |
| `data/hall_thermocouple_77K.csv` | type K thermocouple readings during the cold run |
| `figures/fig_hall_magnet_calibration.pdf` | both hysteresis branches, cubic fit, residuals, branch gap |
| `figures/fig_hall_VH_vs_I.pdf` | family of V_H against I lines, one per field, 300 K |
| `figures/fig_hall_VH_vs_IB.pdf` | global collapse onto one line with residual panel, 300 K |
| `figures/fig_hall_VL_vs_I.pdf` | V_L against I at zero field with residual panel, 300 K |
| `figures/fig_hall_angle.pdf` | cosine fit to the angular scan with residual panel |
| `figures/fig_hall_77K_comparison.pdf` | 300 K and cryogenic Hall and longitudinal data side by side |

Note on two of the figures, so the captions describe them correctly:

- In `fig_hall_VH_vs_IB.pdf` the upper panel shows the raw Hall voltages against
  I*B with the naive two-parameter global fit drawn through them. The lower panel
  shows the residuals of that fit, coloured by field setting, together with the
  residuals of the three-parameter fit as black crosses. The structure grouped by
  field setting in the first set, and its absence in the second, is the visual
  evidence for the contact-misalignment term.
- In the left panel of `fig_hall_77K_comparison.pdf` the fitted misalignment term
  k*I and the constant c have been subtracted from the Hall voltages, so the
  remaining signal is the pure Hall term and the fitted lines pass through the
  origin. The axis is labelled accordingly. The right panel shows raw longitudinal
  voltages, with the cryogenic data repeated in an expanded inset.

Every figure caption in the report must carry the words "representative data".

## 12. Sanity checks run by the generator

| Check | Result | Detail |
|---|---|---|
| max sample current below the 100 uA software cap | pass | max I = 90.11 uA |
| Hall voltages fit a 200 mV DMM range | pass | max |V_H| = 43.9 mV |
| longitudinal voltages fit a 1 V DMM range | pass | max |V_L| = 415.8 mV |
| smallest Hall voltage above the 50 uV noise floor | pass | min |V_H| at 300 K = 1.804 mV |
| peak field between 0.30 and 0.40 T | pass | B_max = 0.3451 T |
| mobility rises on cooling | pass | ratio = 15.1 |
| mobility ratio in the literature range 10 to 40 | pass | ratio = 15.1 |
| sheet density falls by 10 to 25 per cent on cooling | pass | fall = 12.2 per cent |
| fitted n_sq at 300 K within the quoted total uncertainty of ground truth | pass | 1.08 per cent, quoted +/- 3.9 per cent |
| fitted n_sq at 77 K within the quoted total uncertainty of ground truth | pass | 0.86 per cent, quoted +/- 3.9 per cent |
| fitted mu at 300 K within the quoted total uncertainty of ground truth | pass | -1.10 per cent, quoted +/- 4.0 per cent |
| fitted mu at 77 K within the quoted total uncertainty of ground truth | pass | -0.57 per cent, quoted +/- 4.0 per cent |
| fitted R_sq at 300 K within the quoted total uncertainty of ground truth | pass | 0.03 per cent |
| fitted R_sq at 77 K within the quoted total uncertainty of ground truth | pass | -0.28 per cent |
| angular fit phase within 1.5 deg of the true misalignment | pass | fitted 6.07 deg, true 6.0 deg |
| angular fit chi2/dof near 1 | pass | 1.29 |
| three-parameter Hall fit beats the two-parameter fit at 300 K | pass | chi2/dof 1.51 against 14.20 |
| longitudinal fit chi2/dof near 1 at 300 K | pass | 0.95 |
| longitudinal fit chi2/dof near 1 at 77 K | pass | 0.78 |
| cubic magnet fit better than quadratic and linear | pass | cubic 0.82, quadratic 0.93, linear 8.05 |
| cubic magnet fit chi2/dof near 1 | pass | 0.82 |
| magnet calibration error under 1.5 per cent at every set point | pass | max 0.84 per cent |
| mean free path much shorter than the probe spacing (diffusive) | pass | l_mfp(77 K) = 1.31 um against l = 300 um |
| no Shubnikov-de Haas regime at the cryogenic temperature | pass | hbar*omega_c = 0.596 meV against kT = 6.81 meV |
| quoted cryogenic temperature within 10 K of the nitrogen boiling point | pass | T = 78.7 K |

Overall: **all checks passed**.

---

*Representative data. Not a record of measured results.*
