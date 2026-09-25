"""Regenerate (and check) the appendix data tables of the ESR/NMR report from the CSVs.

The Hall report builds its appendix tables with make_hall_tables.py. The ESR/NMR report
had its appendix tables typed in by hand, which is how a single digit in the falling
branch of Table A3 came to be wrong. This script removes that failure mode: it rebuilds
every appendix table row straight from the data files, and it can compare what it builds
against what is currently in the .tex, cell by cell.

Tables covered:
    A1  Helmholtz coil calibration          data/esr_helmholtz_calibration.csv
    A2  ESR resonance data for DPPH         data/esr_resonance_frequency.csv
    A3  Electromagnet calibration           data/nmr_magnet_calibration.csv
    A4  NMR resonance data                  data/nmr_resonance_frequency.csv

Usage:
    python analysis/make_esr_tables.py              print the LaTeX rows
    python analysis/make_esr_tables.py --check      compare against esr_nmr_lab_report.tex

All of these files hold representative data. They are not a record of measured results.
"""

import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(REPORTS, "data")
TEX = os.path.join(REPORTS, "esr_nmr_lab_report.tex")


def read_csv(name):
    """Read one data file, skipping the representative-data banner comment."""
    path = os.path.join(DATA_DIR, name)
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    return rows


def f(value, nd):
    """Format a CSV field to a fixed number of decimal places."""
    return "%.*f" % (nd, float(value))


# ----------------------------------------------------------------------------------
# Row builders. Each returns a list of rows, each row a list of cell strings.
# ----------------------------------------------------------------------------------

def rows_a1():
    data = read_csv("esr_helmholtz_calibration.csv")
    return [
        [r"$I_=$ (\si{\ampere})"] + [f(r["coil_current_A"], 1) for r in data],
        [r"$B_0$ (\si{\milli\tesla})"] + [f(r["magnetic_field_mT"], 2) for r in data],
    ]


def rows_a2():
    data = read_csv("esr_resonance_frequency.csv")
    out = []
    for r in data:
        out.append(
            [r["plug_in_coil"], f(r["frequency_MHz"], 1)]
            + [f(r["I_run%d_A" % i], 3) for i in range(1, 6)]
            + [f(r["I_mean_A"], 4), f(r["I_sd_A"], 4), f(r["I_standard_error_A"], 4),
               f(r["B0_mT"], 4), f(r["B0_standard_error_mT"], 4)]
        )
    return out


def rows_a3():
    data = read_csv("nmr_magnet_calibration.csv")
    rising = [r for r in data if r["sweep_direction"] == "rising"]
    falling = [r for r in data if r["sweep_direction"] == "falling"]
    if len(rising) != len(falling):
        raise SystemExit("branches have different lengths")
    out = []
    for up, down in zip(rising, falling):
        out.append([
            f(up["coil_current_A"], 2), f(up["magnetic_field_T"], 4),
            f(up["field_uncertainty_T"], 4),
            f(down["coil_current_A"], 2), f(down["magnetic_field_T"], 4),
            f(down["field_uncertainty_T"], 4),
        ])
    return out


def rows_a4():
    data = read_csv("nmr_resonance_frequency.csv")
    gly = [r for r in data if r["sample"] == "glycerine"]
    ptfe = [r for r in data if r["sample"] == "PTFE"]
    if len(gly) != len(ptfe):
        raise SystemExit("samples have different lengths")
    out = []
    for g, p in zip(gly, ptfe):
        if g["coil_current_A"] != p["coil_current_A"]:
            raise SystemExit("currents do not line up between the two samples")
        out.append([
            f(g["coil_current_A"], 2),
            f(g["resonance_frequency_MHz"], 2), f(g["magnetic_field_T"], 4),
            f(g["field_uncertainty_T"], 4),
            f(p["resonance_frequency_MHz"], 2), f(p["magnetic_field_T"], 4),
            f(p["field_uncertainty_T"], 4),
        ])
    return out


BUILDERS = [("A1", rows_a1), ("A2", rows_a2), ("A3", rows_a3), ("A4", rows_a4)]
LABELS = {"A1": "tab:a1", "A2": "tab:a2", "A3": "tab:a3", "A4": "tab:a4"}


# ----------------------------------------------------------------------------------
# Reading the rows back out of the .tex
# ----------------------------------------------------------------------------------

RULES = re.compile(r"\\(?:top|mid|bottom)rule|\\cmidrule(?:\([^)]*\))?(?:\{[^}]*\})?")


def tex_rows(name, text):
    """Pull the data rows of one appendix table out of the LaTeX source."""
    start = text.index("\\label{%s}" % LABELS[name])
    body = text[text.index("\\begin{tabular}", start):text.index("\\end{tabular}", start)]
    body = body[body.index("\n"):]
    out = []
    for raw in body.split(r"\\"):
        line = " ".join(RULES.sub(" ", raw).split())
        if not line or r"\multicolumn" in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        first = cells[0]
        if name == "A1":
            keep = first.startswith("$I_=$") or first.startswith("$B_0$")
        elif name == "A2":
            keep = first in ("large", "medium", "small")
        else:
            keep = bool(re.match(r"^\d+\.\d+$", first))
        if keep:
            out.append(cells)
    return out


def check():
    with open(TEX, encoding="utf-8") as handle:
        text = handle.read()

    checked = 0
    mismatches = []
    for name, builder in BUILDERS:
        want = builder()
        got = tex_rows(name, text)
        if len(want) != len(got):
            mismatches.append("%s: %d rows in the CSV, %d rows in the .tex"
                              % (name, len(want), len(got)))
            continue
        for i, (wrow, grow) in enumerate(zip(want, got), start=1):
            if len(wrow) != len(grow):
                mismatches.append("%s row %d: %d cells expected, %d found"
                                  % (name, i, len(wrow), len(grow)))
                continue
            for j, (w, g) in enumerate(zip(wrow, grow), start=1):
                checked += 1
                if w != g:
                    mismatches.append("%s row %d column %d: .tex has %r, CSV gives %r"
                                      % (name, i, j, g, w))
        print("%s: %d rows x %d columns" % (name, len(want), len(want[0])))

    print("\ncells checked : %d" % checked)
    print("mismatches    : %d" % len(mismatches))
    for line in mismatches:
        print("  " + line)
    return 1 if mismatches else 0


def emit():
    for name, builder in BUILDERS:
        print("%% ---- Table %s ----" % name)
        for row in builder():
            print(" & ".join(row) + r" \\")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv[1:] else emit())
