# Physics Laboratory Reports: ESR, NMR and the Hall Effect in a 2DEG

> ## Read this first: the data in these reports is not measured data
>
> **Every number, table and figure in both reports is representative data.** It was generated from
> accepted physical constants plus realistic instrument scatter, then analysed exactly as real
> laboratory readings would be, in order to demonstrate the method, the analysis and the treatment of
> uncertainty.
>
> **None of it is a record of measurements taken by the author.** The original laboratory notebooks
> for these experiments are no longer available, so the reports were written up from the experiment
> briefs instead. The apparatus, the procedure, the theory and the comparisons with published values
> are genuine. The measured values are not.
>
> **Nothing here should be read, cited or reused as experimental evidence.**

Two complete undergraduate physics laboratory reports, written to demonstrate experimental analysis
and uncertainty treatment end to end: from raw readings, through fitting and error propagation, to a
physical result compared against the literature.

Author: Abdullah Alkabbawi. The experiments are third-year laboratory experiments from a BSc (Hons)
Physics degree at the University of Glasgow.

## The two reports

| Report | Pages | PDF |
| :--- | :--- | :--- |
| Electron Spin Resonance and Nuclear Magnetic Resonance | 26 | [`esr_nmr_lab_report.pdf`](esr_nmr_lab_report.pdf) |
| The Hall Effect in a Two-Dimensional Electron Gas | 26 | [`hall_effect_2deg_lab_report.pdf`](hall_effect_2deg_lab_report.pdf) |

### ESR and NMR

Magnetic resonance of an unpaired electron and of two spin one-half nuclei.

- Helmholtz coil calibration against a Hall magnetometer.
- ESR in DPPH across 15 to 130 MHz on three plug-in coils, giving the electron spin g-factor.
- Absorption line width, and the transverse relaxation time implied by it.
- Iron-cored electromagnet calibration, including the hysteresis between rising and falling branches.
- NMR of protons in glycerine and of fluorine in PTFE, giving both nuclear g-factors.
- Identification of polythene from its resonance frequency.

Representative results: g(DPPH) = 1.998 +/- 0.021 against an accepted 2.0036; line width
0.238 +/- 0.018 mT; g(1H) = 5.57 +/- 0.08 against 5.5857; g(19F) = 5.19 +/- 0.10 against 5.2577.

### Hall effect in a 2DEG

Characterisation of a GaAs/AlGaAs modulation-doped heterostructure on a six-contact Hall bar, at
room temperature and under liquid nitrogen cooling.

- Electromagnet calibration with a polynomial fit and visible hysteresis.
- Hall voltage against sample current and flux density, reduced by a global weighted fit that
  separates the Hall term from a contact-misalignment term.
- Longitudinal resistance and resistance per square.
- Angular dependence of the Hall voltage, fitted to a cosine.
- The same measurements repeated at about 79 K.

Representative results: sheet density falls from 5.05 to 4.44 x 10^15 m^-2 on cooling, a change of
12 per cent, while mobility rises from 0.79 to 11.93 m^2 V^-1 s^-1, a factor of 15.1. That contrast
is the expected signature of modulation doping.

## Reproducing the analysis

The generators are seeded, so the same data comes back every time.

```bash
python analysis/generate_esr_nmr.py
python analysis/generate_hall.py
```

Each script builds the representative dataset forward from accepted physical parameters, adds
realistic scatter and systematic offsets, then fits the result and propagates the uncertainties. The
values quoted in the reports are the values those fits return, not the ground truth fed in, so the
two differ slightly. That difference is the point.

Requires numpy, scipy, matplotlib and pandas. Building the PDFs needs a LaTeX installation:

```bash
pdflatex -interaction=nonstopmode esr_nmr_lab_report.tex
```

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

`data/results_esr_nmr.md` and `data/results_hall.md` record every fitted value with its uncertainty,
the ground truth used to generate it, the published value it is compared against, and the
percentage difference.

## Source material

The experiment briefs these reports follow are departmental documents belonging to the University of
Glasgow. They are **not** included here and should not be redistributed. Only the reports, the
analysis code and the generated data are in this repository.
