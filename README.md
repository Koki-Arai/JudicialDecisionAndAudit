# Judicial Decisions as Legitimizing Mechanisms: Evidence from Japan's Audit Market

Replication code and data for the paper:

> **Judicial Decisions as Legitimizing Mechanisms: Evidence from Japan's Audit Market**  
> *Journal of Law and Economics / European Journal of Law and Economics* (under review)

---

## Overview

This repository contains all code and processed data needed to replicate the empirical analyses in the paper. The study examines the role of the November 2020 Supreme Court of Japan decision within a three-stage institutional co-evolution of Japan's audit market (2015–2023), using two independent datasets:

| Layer | Data | Method | Key finding |
|---|---|---|---|
| **Layer 2** (large firms) | JICPA Survey of Audit Practices, 2013–2023 | Difference-in-Differences | Audit fee DID = +3,630 thousand JPY (p=0.033); hourly rate DID = −395 JPY (p=0.009) |
| **Layer 1** (small firms) | Ministry of Justice registry statistics, 2009–2025 | Interrupted Time Series | KAM level change = −40.5% (p=0.001); SC level change = +52.5% (p<0.001) |
| **Bridge** | Layer 1 × Layer 2 merged | VAR/Granger + mediation | Registry turnover → hourly rate suppression (coef = +144, p=0.050) |

---

## Repository Structure

```
.
├── code/
│   ├── analysis1_did_jicpa.py       # Layer 2: DID analysis (JICPA data)
│   ├── analysis2_its_registry.py    # Layer 1: ITS + macro controls (registry data)
│   └── analysis3_var_bridge.py      # VAR/Granger causality + bridge analysis
│
├── data/
│   ├── raw/                         # Raw data files (see Data section below)
│   └── processed/
│       ├── jicpa_panel_2013_2023.csv      # JICPA panel (3 categories × 11 years)
│       └── registry_panel_2009_2025.csv   # Monthly registry panel (N=204)
│
├── figures/
│   ├── fig1_coevolution_three_stage.png   # Fig 1: Three-stage co-evolution (main)
│   ├── fig2_its_counterfactual.png        # Fig 2: ITS counterfactual
│   ├── fig3_event_study.png               # Fig 3: Monthly event study
│   ├── fig4_var_irf.png                   # Fig 4: VAR impulse response functions
│   ├── figA1_registry_timeseries.png      # Appendix: Registry time series
│   ├── figA2_parallel_trends.png          # Appendix: Parallel trends check
│   ├── figA3_var_fevd.png                 # Appendix: Forecast error variance decomp.
│   └── figA4_registry_jicpa_integrated.png  # Appendix: Registry × JICPA integrated
│
├── results/
│   ├── did_jicpa_results.txt        # Numerical output from analysis1
│   ├── its_registry_results.txt     # Numerical output from analysis2
│   └── var_bridge_results.txt       # Numerical output from analysis3
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data

### Raw data (not included in this repository)

Raw data files must be placed in `data/raw/` before running the scripts. Both sources are publicly available.

| File | Source | Description |
|---|---|---|
| `監査実施状況2013_2023.csv` | [JICPA](https://jicpa.or.jp/specialized_field/audit/audit_survey/) | Annual Survey of Audit Practices (2013–2023); encoding: Shift-JIS |
| `FEH_00250002_*.csv` | [Ministry of Justice e-Stat](https://www.e-stat.go.jp/stat-search/files?tclass=000001048465) | Monthly commercial registration statistics; encoding: UTF-8 |
| `マクロ指標.csv` | Various (see below) | Macro control variables (Nikkei, CPI, USD/JPY, lending rate, CGPI) |

Macro indicator sources: Nikkei 225 (Yahoo Finance), CPI (Statistics Bureau of Japan), USD/JPY (Bank of Japan), Lending rate (Bank of Japan), Corporate Goods Price Index (Bank of Japan).

### Processed data (included)

- `jicpa_panel_2013_2023.csv`: Panel of average audit fee, hourly rate, estimated hours, and company count by audit category and fiscal year (N=33).
- `registry_panel_2009_2025.csv`: Monthly panel of kansayaku (auditor-related), yakuin (director-related), and torishimariyaku (board-related) registration changes with derived log series (N=204).

---

## Replication

### Environment

```bash
pip install -r requirements.txt
```

Alternatively, all three scripts are written to run in **Google Colab** with minimal setup (upload the raw data files when prompted).

### Running order

```bash
# Step 1: Layer 2 — DID analysis on JICPA data (Tables 1–3, robustness checks)
python code/analysis1_did_jicpa.py

# Step 2: Layer 1 — ITS + macro controls on registry data (Table 4, Figs 2–3)
python code/analysis2_its_registry.py

# Step 3: VAR/Granger causality + bridge analysis (Tables 5–6, Fig 4)
python code/analysis3_var_bridge.py
```

Each script saves figures to `figures/` and a text summary to `results/`.

### Data paths

By default, scripts expect raw data in `data/raw/`. If running in Google Colab, upload files directly to `/content/` and the scripts will detect them automatically.

---

## Identification Strategy

### Layer 2: Difference-in-Differences

```
Y_it = α + β₁Post_t + β₂Treat_i + β₃(Post_t × Treat_i) + γ_t + ε_it
```

- **Treatment**: FIEA Consolidated audits (listed companies subject to Supreme Court ruling)
- **Control**: Companies Act audits (non-listed; outside Listed Company Audit Firm Registration System)
- **Post**: fiscal years 2020–2023
- **Outcomes**: `avgFee` (thousand JPY), `hourlyRate` (JPY) — directly measured
- `estimatedHours` reported descriptively only (algebraically derived; see paper §4.2)
- HC3 heteroskedasticity-consistent standard errors throughout

### Layer 1: Interrupted Time Series

```
log(Y_t) = α + β₁t + β₂PostKAM_t + β₃(t·PostKAM_t) + β₄PostSC_t + β₅(t·PostSC_t) + month FE + ε_t
```

- **Treatment series**: `kansayaku` (auditor-related registration changes)
- **Control series**: `yakuin` (director-related registration changes)
- **Interventions**: KAM (April 2015), SC Decision (November 2020)
- Macro controls added in sensitivity analyses (5 lagged standardized indicators)

### Bridge (analysis3)

VAR(6) on HP-cycle-detrended log-differenced monthly series; Granger causality tests split pre/post SC decision. Annual registry turnover rates merged into JICPA panel to test pricing leverage channel.

---

## Key Results

| Table | Analysis | Main coefficient | p-value |
|---|---|---|---|
| Table 3 | DID: avgFee (FIEA Consol. vs Companies Act) | +3,630 thousand JPY | 0.033** |
| Table 3 | DID: hourlyRate (FIEA Consol. vs Companies Act) | −395 JPY | 0.009*** |
| Table 4 | ITS: KAM level change (kansayaku) | −0.519 log pts (−40.5%) | 0.001*** |
| Table 4 | ITS: SC level change (kansayaku) | +0.422 log pts (+52.5%) | <0.001*** |
| Table 4 | ITS: SC level change (yakuin, control) | −0.012 log pts | 0.575 (n.s.) |
| Table 5 | Bridge: Kansa YoY → hourlyRate | +144 JPY/SD | 0.050* |
| Table 6b | Granger: tori→kansa, post-SC (min p, lag 4) | F = 6.446 | <0.001*** |

---

## Software

- Python 3.10+
- See `requirements.txt` for package versions

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{arai2025judicial,
  title   = {Judicial Decisions as Legitimizing Mechanisms: 
             Evidence from Japan's Audit Market},
  author  = {Arai, [co-authors]},
  journal = {[Journal]},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

Code: MIT License  
Data: The processed data files are derived from publicly available government and JICPA sources. Users are responsible for complying with the terms of the original data providers.
