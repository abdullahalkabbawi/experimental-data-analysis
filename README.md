# Experimental Data Analysis: Fitting, Error Propagation and Uncertainty

Curve fitting, uncertainty propagation and the separation of random from systematic error, worked
end to end on two physics experiments and written up to publication standard.

The datasets here are **generated rather than measured**, which is deliberate: because the true
values used to build them are known, every fit can be checked against the answer it was supposed to
recover. Full detail in [About the data](#about-the-data) below.

Author: Abdullah Alkabbawi. The experiments are third-year laboratory experiments from a BSc (Hons)
Physics degree at the University of Glasgow.

## What this demonstrates

| Technique | Where |
| :--- | :--- |
| Weighted least squares, uncertainties from the covariance matrix | `analysis/generate_esr_nmr.py`, `analysis/generate_hall.py` |
| Straight-line fitting with uncertainties on **both** axes, by the iterated effective variance method | `effective_variance_fit()`, `generate_esr_nmr.py:215` |
| Monte Carlo cross-check, 5000 draws, resampling correlated calibration coefficients with the data | `generate_esr_nmr.py:572` |
| The same quantity derived three independent ways and compared: effective variance, chain rule, Monte Carlo | `generate_esr_nmr.py:553` |
| Reduced chi-squared reported with degrees of freedom, and used to judge the fit | both generators |
| Statistical and systematic uncertainty separated, with correlated systematics carried through to the final figure | both generators |
| Polynomial calibration of a non-linear instrument, with hysteresis measured rather than ignored | iron-cored electromagnet, both reports |
| Global weighted fit that separates the wanted term from a contaminating one | Hall: `V_H = m_H(IB) + kI + c` |
| Every result compared against published values, with the difference expressed in standard deviations | `data/results_*.md` |

Each fitted value is recorded in `data/results_esr_nmr.md` and `data/results_hall.md` alongside the
true value used to generate it and the published value it is compared against, so the accuracy of
the analysis itself can be audited line by line.

## Case study 1: magnetic resonance

[`esr_nmr_lab_report.pdf`](esr_nmr_lab_report.pdf), 26 pages. Electron spin resonance in DPPH and
nuclear magnetic resonance of protons and of fluorine.

The measurement chain has two calibrations in series, so the uncertainty on the final g-factor comes
from two fitted curves at once. That is what makes it a good error-propagation exercise, and it is
why the gradient is derived three separate ways and the three compared.

| Quantity | Recovered | Published |
| :--- | :--- | :--- |
| g-factor of DPPH | 1.998 +/- 0.021 | 2.0036 |
| Absorption line width (FWHM) | 0.238 +/- 0.018 mT | 1.5 to 3.0 G peak-to-peak |
| g-factor of the proton | 5.57 +/- 0.08 | 5.5857 |
| g-factor of fluorine-19 | 5.19 +/- 0.10 | 5.2577 |

Also covered: Helmholtz coil calibration against a magnetometer, the hysteresis between rising and
falling branches of an iron-cored electromagnet, and identifying polythene from its resonance
frequency alone.

## Case study 2: the Hall effect in a two-dimensional electron gas

[`hall_effect_2deg_lab_report.pdf`](hall_effect_2deg_lab_report.pdf), 26 pages. A GaAs/AlGaAs
modulation-doped heterostructure on a six-contact Hall bar, at room temperature and at about 79 K.

The interesting analysis problem is contact misalignment: a small offset voltage contaminates the
Hall signal. Rather than subtract it by eye, a three-parameter global weighted fit separates the
Hall term from the current-proportional contamination, using all 90 points at once.

| Quantity | 300 K | 79 K |
| :--- | :--- | :--- |
| Sheet carrier density | 5.05 +/- 0.19 x 10^15 m^-2 | 4.44 +/- 0.17 x 10^15 m^-2 |
| Electron mobility | 0.791 +/- 0.032 m^2 V^-1 s^-1 | 11.93 +/- 0.48 m^2 V^-1 s^-1 |
| Resistance per square | 1561 +/- 16 ohm/sq | 117.9 +/- 1.2 ohm/sq |

Mobility rises by a factor of 15.1 on cooling while the carrier density falls only 12 per cent. That
contrast is the signature of modulation doping, and both mobilities agree with published values for
comparable structures to within 3 per cent.

## Reproducing the analysis

```bash
python analysis/generate_esr_nmr.py
python analysis/generate_hall.py
```

The generators are seeded, so the same numbers come back every run. Each one builds the dataset
forward from accepted physical parameters, adds realistic instrument scatter and systematic offsets,
then fits the result and propagates the uncertainties. **The values quoted in the reports are the
values those fits return, not the parameters fed in**, so the two differ slightly, and that
difference is the measure of how well the analysis works.

Requires numpy, scipy, matplotlib and pandas. Rebuilding the PDFs needs LaTeX:

```bash
pdflatex -interaction=nonstopmode esr_nmr_lab_report.tex
```

## About the data

The laboratory notebooks for these experiments are no longer available. The reports were therefore
written up from the experiment briefs, with representative datasets generated from accepted physical
constants plus realistic instrument scatter, and then analysed exactly as real readings would be.

**No number in either report is a record of a measurement taken by the author, and nothing here
should be cited or reused as experimental evidence.** The apparatus, the procedure, the theory and
the comparisons with published values are genuine; the readings are not.

This is stated in a note after each abstract, in the caption of every results table and data figure,
in a footer on all 52 pages of the two reports, in the header of all 16 data files, and in the
figure images themselves so that a figure taken out of context stays labelled.

## Layout

```text
.
├── esr_nmr_lab_report.tex / .pdf
├── hall_effect_2deg_lab_report.tex / .pdf
├── analysis/     data generators, figure scripts and table builders
├── data/         generated datasets as CSV, plus a results file per report
├── figures/      every figure, as vector PDF
└── tables/       generated LaTeX fragments for the appendix tables
```

## Source material

The experiment briefs these reports follow are departmental documents belonging to the University of
Glasgow. They are **not** included here and are not redistributed. Only the reports, the analysis
code and the generated data are in this repository.
