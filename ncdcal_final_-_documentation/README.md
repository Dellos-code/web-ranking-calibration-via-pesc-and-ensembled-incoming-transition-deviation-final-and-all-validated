# Two-Stage Calibration of NCDawareRank / Βαθμονόμηση δύο σταδίων του NCDawareRank

Ioannis Konstantellos — CEID, University of Patras — October 2026

## English

This archive contains the complete research package for the manuscript *Two-Stage Calibration of
Decomposable Web Ranking: Block Escape Probability, Incoming-Transition Deviation, and the Anchoring of
Oracle Damping*.

| What | Where |
|---|---|
| English paper | `paper/paper_en.pdf` (source `paper/paper_en.tex`) |
| Greek paper | `paper/paper_gr.pdf` (source `paper/paper_gr.tex`) |
| Technical documentation and thesis guide | `DOCUMENTATION_EN.md` / `.pdf`, `DOCUMENTATION_GR.md` / `.pdf` |
| Method disclosure (English and Greek) | `METHOD_DISCLOSURE_EN_GR.md` / `.txt` |
| Code | `*.py`, `*.sh`, `tests/` |
| Practical autotune tool | `ncd_autotune.py` (validated anchors 0.85 and 0.95) |
| Results, figures, logs | `results/`, `figures/`, `figures_gr/`, `logs/` |
| Prepared real graphs | `data/real/` |

Quick start (from this directory):

```
python3 tests/test_lib.py          # unit tests (seconds)
python3 tests/test_stage2.py
python3 tests/test_nonmono.py
python3 tests/test_autotune.py
python3 check_build.py
python3 verify_stage2.py           # recompute headline numbers
python3 verify_v2.py               # independently verify v2 headline inputs
bash build_papers.sh               # regenerate tables/macros, compile papers
bash build_docs.sh                 # build the documentation PDFs
```

The full reproduction (raw data download, both experiment stages, analyses and figures) is described
step by step in Section 11 of `DOCUMENTATION_EN.pdf`.

The practical single-snapshot tool is `ncd_autotune.py`; it accepts a prepared graph NPZ and supports
the validated reference anchors 0.85 and 0.95.

Practical recipe: call `tune(src, dst, block, alpha_ref=0.85)` for the validated 0.85 reference
and `alpha=0.81`, or use `alpha_ref=0.95` for `alpha=0.93`. At the 0.95 anchor the calibration
optimum for the Stage 1 rule is 0.920, only 0.0002 $\tau$ above 0.93. The tool
returns the Stage 1 rule's `r` by design, not the reference's own `r`; `baseline` is currently
ignored with a warning because no baseline rule passed the preregistered WP3/WP4 checks.

## Ελληνικά

Το αρχείο περιέχει το πλήρες ερευνητικό πακέτο της εργασίας *Βαθμονόμηση δύο σταδίων της διασπάσιμης
κατάταξης ιστοσελίδων: πιθανότητα διαφυγής από το μπλοκ, απόκλιση εισερχόμενων μεταβάσεων και η
αγκύρωση της βέλτιστης απόσβεσης*: τα δύο κείμενα (`paper/paper_gr.pdf`, `paper/paper_en.pdf`), την
τεχνική τεκμηρίωση (`DOCUMENTATION_GR.pdf`, `DOCUMENTATION_EN.pdf`), τη δήλωση μεθόδου
(`METHOD_DISCLOSURE_EN_GR.md`), τον κώδικα, τα αποτελέσματα και τα σχήματα. Οι εντολές γρήγορης εκκίνησης είναι οι ίδιες με τις παραπάνω· η πλήρης αναπαραγωγή περιγράφεται
βήμα προς βήμα στην Ενότητα 11 του `DOCUMENTATION_GR.pdf`.
