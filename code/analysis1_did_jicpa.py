"""
analysis1_did_jicpa.py
======================
Replication code for:
  "Judicial Decisions as Legitimizing Mechanisms: Evidence from Japan's Audit Market"

LAYER 2 ANALYSIS — Difference-in-Differences on JICPA Survey Data
  Data : JICPA Survey of Audit Practices, 2013–2023
         Place raw CSV (Shift-JIS) in data/raw/ or upload to /content/ on Colab

  PART 1  Data loading and panel construction
  PART 2  Descriptive statistics and time-series plots (Tables 1–2)
  PART 3  Main DID estimates — avgFee and hourlyRate (Table 3)
            Strategy 1 : Basic DID (directly measured variables only)
            Strategy 2 : Stepwise log-transformation decomposition
            Strategy 3 : Two-stage KAM + SC sequential DID
  PART 4  Robustness checks
            4.1  Pre-trend / event-time tests
            4.2  Group-specific linear trends
            4.3  Alternative cutoff points (2019, 2021)
            4.4  Log vs level specifications
            4.5  Placebo tests (false treatment dates: 2016, 2017, 2018)
  PART 5  Results summary (saved to results/did_jicpa_results.txt)

  Key outputs
    figures/fig1_coevolution_three_stage.png  (Panel B)
    figures/figA2_parallel_trends.png
    results/did_jicpa_results.txt

NOTE: estimatedHours = avgFee × 1000 ÷ hourlyRate is algebraically derived
      and is reported for descriptive purposes only. Causal inference is
      restricted to directly measured avgFee and hourlyRate.
"""

import re, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False

DATA_PATH  = "/mnt/user-data/uploads/監査実施状況2013_2023.csv"
OUT        = "/mnt/user-data/outputs/"
os.makedirs(OUT, exist_ok=True)

print("=" * 65)
print("  Audit DID Analysis — Verification Code")
print("=" * 65)


# ════════════════════════════════════════════════════════════
# PART 1  データ整備
# ════════════════════════════════════════════════════════════
print("\n[PART 1] Data Preparation")

raw = pd.read_csv(DATA_PATH, encoding="shift_jis", header=None)
print(f"  Raw shape: {raw.shape}")

# -- 年度ブロック開始行を検出 --
YEAR_ROWS = []
for i, row in raw.iterrows():
    m = re.match(r"^(\d{4})年度", str(row[0]).strip())
    if m:
        yr = int(m.group(1))
        YEAR_ROWS.append((yr, i))
YEAR_ROWS = sorted(YEAR_ROWS, key=lambda x: x[1])   # 行番号順（出現順）でソート
print(f"  Year blocks (row order): {[y for y,_ in YEAR_ROWS]}")

def to_cat(s):
    """監査区分名 → 統一キー（空白を無視）"""
    s = re.sub(r"[\s\u3000]+", "", str(s).strip())
    if re.search(r"金商法.*連結", s): return "FIEA_Consol"
    if re.search(r"金商法.*個別", s): return "FIEA_Indiv"
    if s == "会社法":                  return "CompanyAct"
    return None

def to_float(s):
    if pd.isna(s): return np.nan
    try: return float(str(s).strip().replace(",","").replace("%",""))
    except: return np.nan

records = []
for idx, (yr, sr) in enumerate(YEAR_ROWS):
    er = YEAR_ROWS[idx+1][1] if idx+1 < len(YEAR_ROWS) else len(raw)
    for _, row in raw.iloc[sr:er].iterrows():
        cat = to_cat(row[0])
        if cat is None: continue
        fee  = to_float(row[14])   # 千円/社
        rate = to_float(row[15])   # 円/時間
        hours = (fee*1000/rate) if (pd.notna(fee) and pd.notna(rate) and rate>0) else np.nan
        records.append({"year": yr, "category": cat,
                        "n_companies": to_float(row[1]),
                        "avgFee": fee, "hourlyRate": rate,
                        "estimatedHours": hours})

panel = pd.DataFrame(records).sort_values(["category","year"]).reset_index(drop=True)
print(f"  Panel: {panel.shape}  |  Years: {sorted(panel['year'].unique())}")
print(f"  Categories: {panel['category'].unique().tolist()}")

# -- Treatment / Post flags --
panel["treat"]    = panel["category"].isin(["FIEA_Consol","FIEA_Indiv"]).astype(int)
panel["post2020"] = (panel["year"] >= 2020).astype(int)
panel["did"]      = panel["treat"] * panel["post2020"]
panel["year_c"]   = panel["year"] - 2016   # センタリング

for v in ["avgFee","hourlyRate","estimatedHours"]:
    panel[f"log_{v}"] = np.log(panel[v].replace(0, np.nan))

print("\n  avgFee pivot (thousand JPY):")
pv = panel.pivot_table(index="year", columns="category", values="avgFee")
print(pv.map(lambda x: f"{x:,.0f}" if pd.notna(x) else "").to_string())


# ════════════════════════════════════════════════════════════
# PART 2  記述統計・推移グラフ
# ════════════════════════════════════════════════════════════
print("\n[PART 2] Descriptive Statistics & Plots")

desc = []
for cat in ["FIEA_Consol","FIEA_Indiv","CompanyAct"]:
    for v in ["avgFee","hourlyRate","estimatedHours"]:
        s = panel.loc[panel["category"]==cat, v].describe()
        desc.append({"Cat":cat,"Var":v,"Mean":s["mean"],"SD":s["std"],
                     "Min":s["min"],"Max":s["max"],"N":int(s["count"])})
desc_df = pd.DataFrame(desc)
print(desc_df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
desc_df.to_csv(OUT+"table1_descriptive.csv", index=False, encoding="utf-8-sig")

# -- Time-series figure --
fig, axes = plt.subplots(3,1, figsize=(10,12))
clr = {"FIEA_Consol":"#1f77b4","FIEA_Indiv":"#ff7f0e","CompanyAct":"#2ca02c"}
mrk = {"FIEA_Consol":"o","FIEA_Indiv":"s","CompanyAct":"^"}
lbl = {"FIEA_Consol":"FIEA Consol (Treat)","FIEA_Indiv":"FIEA Indiv (Treat)",
       "CompanyAct":"Companies Act (Control)"}

for ax,(var,ylabel) in zip(axes,[
        ("avgFee","Avg Audit Fee (thousand JPY)"),
        ("hourlyRate","Hourly Rate (JPY)"),
        ("estimatedHours","Estimated Hours")]):
    for cat in ["FIEA_Consol","FIEA_Indiv","CompanyAct"]:
        sub = panel[panel["category"]==cat].sort_values("year")
        ax.plot(sub["year"], sub[var], label=lbl[cat],
                color=clr[cat], marker=mrk[cat], lw=2, ms=6)
    ax.axvline(2014.5, color="gray", ls=":", lw=1.5, alpha=0.8)
    ax.axvline(2019.5, color="red",  ls="--",lw=1.5, alpha=0.9)
    ymax = panel[var].max()
    ax.text(2014.7, ymax*0.97, "KAM\n(2015)", fontsize=7, color="gray", va="top")
    ax.text(2019.7, ymax*0.97, "SC\n(2020)", fontsize=7, color="red",  va="top")
    ax.set_title(ylabel, fontsize=11); ax.set_xlabel("Fiscal Year")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.grid(alpha=0.3)
    if var=="avgFee": ax.legend(fontsize=9)
plt.tight_layout()
fig.suptitle("JICPA Audit Survey 2013–2023", y=1.00, fontsize=12)
fig.savefig(OUT+"fig1_time_series.png", dpi=150, bbox_inches="tight"); plt.close()
print("  -> fig1_time_series.png")


# ════════════════════════════════════════════════════════════
# PART 3  本推計
# ════════════════════════════════════════════════════════════
print("\n[PART 3] Main DID Estimation")

def show(mod, var="did", unit=""):
    c=mod.params[var]; se=mod.bse[var]; p=mod.pvalues[var]
    st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
    print(f"      DID = {c:+12,.2f} {unit}  SE={se:9,.2f}  p={p:.3f} {st}")
    return c,se,p,st

est = panel.copy()

print("\n  [S1] Basic DID — all treat vs CompanyAct")
res_s1={}
for dep in ["avgFee","hourlyRate"]:
    print(f"    {dep}")
    mod = smf.ols(f"{dep} ~ treat+post2020+did", data=est).fit(cov_type="HC3")
    res_s1[dep]=mod; show(mod, unit="千円" if "Fee" in dep else "円")

print("\n  [S1-Primary] FIEA_Consol vs CompanyAct")
pp = panel[panel["category"].isin(["FIEA_Consol","CompanyAct"])].copy()
pp["treat"] = (pp["category"]=="FIEA_Consol").astype(int)
pp["did"]   = pp["treat"]*pp["post2020"]
res_pp={}
for dep in ["avgFee","hourlyRate"]:
    print(f"    {dep}")
    mod = smf.ols(f"{dep} ~ treat+post2020+did", data=pp).fit(cov_type="HC3")
    res_pp[dep]=mod; show(mod, unit="千円" if "Fee" in dep else "円")

print("\n  [S2] estimatedHours  [algebraic dependency — B-1]")
mod_h = smf.ols("estimatedHours ~ treat+post2020+did", data=est).fit(cov_type="HC3")
show(mod_h, unit="時間")

print("\n  [S3] Log-transformed DID")
res_log={}
for dep in ["log_avgFee","log_hourlyRate","log_estimatedHours"]:
    print(f"    {dep}")
    mod = smf.ols(f"{dep} ~ treat+post2020+did", data=est).fit(cov_type="HC3")
    res_log[dep]=mod
    c=mod.params["did"]; p=mod.pvalues["did"]
    st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
    print(f"      DID = {c:+.4f} ({(np.exp(c)-1)*100:+.1f}%)  p={p:.3f} {st}")


# ════════════════════════════════════════════════════════════
# PART 4  ロバストネス検定
# ════════════════════════════════════════════════════════════
print("\n[PART 4] Robustness Checks")

years = sorted(est["year"].unique())

# ── 4-1  Event study ────────────────────────────────────────
print("\n  [4-1] Event Study (avgFee, base=2019)")
BASE = 2019
es = est.copy()
for yr in years:
    es[f"y{yr}"]  = (es["year"]==yr).astype(int)
    es[f"td{yr}"] = (es["treat"]*(es["year"]==yr)).astype(int)

yvars  = [f"y{yr}"  for yr in years if yr!=BASE]
tdvars = [f"td{yr}" for yr in years if yr!=BASE]
fml = "avgFee ~ treat + " + "+".join(yvars) + "+" + "+".join(tdvars)
mod_es = smf.ols(fml, data=es).fit(cov_type="HC3")

ec_rows=[]
for yr in years:
    if yr==BASE:
        ec_rows.append({"year":yr,"coef":0.0,"ci_lo":0.0,"ci_hi":0.0,"p":np.nan})
    else:
        vn=f"td{yr}"
        if vn in mod_es.params:
            c=mod_es.params[vn]; se=mod_es.bse[vn]; p=mod_es.pvalues[vn]
            ec_rows.append({"year":yr,"coef":c,
                            "ci_lo":c-1.96*se,"ci_hi":c+1.96*se,"p":p})
ec_df = pd.DataFrame(ec_rows).sort_values("year")
print(ec_df[["year","coef","ci_lo","ci_hi","p"]].to_string(
      index=False, float_format=lambda x: f"{x:+,.1f}"))

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.fill_between(ec_df["year"], ec_df["ci_lo"], ec_df["ci_hi"],
                 alpha=0.2, color="#1f77b4")
ax2.plot(ec_df["year"], ec_df["coef"], "o-", color="#1f77b4",
         lw=2, ms=7, label="DID coef (avgFee, base=2019)")
ax2.axhline(0, color="black", lw=0.8, ls="--")
ax2.axvline(2014.5, color="gray", lw=1.5, ls=":",  alpha=0.8, label="KAM (2015)")
ax2.axvline(2019.5, color="red",  lw=1.5, ls="--", alpha=0.9, label="SC Decision (2020)")
ax2.set_title("Event Study: Year-by-Year DID (avgFee, Base=2019)", fontsize=12)
ax2.set_xlabel("Fiscal Year"); ax2.set_ylabel("DID Coefficient (thousand JPY)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:+,.0f}"))
ax2.legend(); ax2.grid(alpha=0.3); fig2.tight_layout()
fig2.savefig(OUT+"fig2_event_study.png", dpi=150, bbox_inches="tight"); plt.close()
print("  -> fig2_event_study.png")

# ── 4-2  グループ固有線形トレンド ────────────────────────────
print("\n  [4-2] Group-specific Linear Trends")
est2 = est.copy()
est2["treat_yearc"] = est2["treat"]*est2["year_c"]
for dep in ["avgFee","hourlyRate"]:
    mod_t = smf.ols(f"{dep} ~ treat+post2020+did+year_c+treat_yearc",
                    data=est2).fit(cov_type="HC3")
    c=mod_t.params["did"]; p=mod_t.pvalues["did"]
    st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
    print(f"    {dep:20s}  DID(trend-adj) = {c:+10,.1f}  p={p:.3f} {st}")

# ── 4-3  代替カットオフ ──────────────────────────────────────
print("\n  [4-3] Alternative Cutoffs (2019, 2021)")
for cutoff in [2019, 2021]:
    tmp=est.copy()
    tmp["post_alt"]=(tmp["year"]>=cutoff).astype(int)
    tmp["did_alt"]=tmp["treat"]*tmp["post_alt"]
    for dep in ["avgFee","hourlyRate"]:
        mod=smf.ols(f"{dep} ~ treat+post_alt+did_alt",data=tmp).fit(cov_type="HC3")
        c=mod.params["did_alt"]; p=mod.pvalues["did_alt"]
        st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
        print(f"    Cutoff={cutoff}  {dep:15s}  DID={c:+10,.1f}  p={p:.3f} {st}")

# ── 4-4  プラセボテスト ──────────────────────────────────────
print("\n  [4-4] Placebo Tests — using ONLY pre-2020 data")
print("  (Reviewer B-2: test parallel trends assumption)")
print()
pre = est[est["year"] < 2020].copy()   # 2013–2019 のみ

placebo_log = []
for fake in [2016, 2017, 2018]:
    pre_f = pre.copy()
    pre_f["post_f"] = (pre_f["year"]>=fake).astype(int)
    pre_f["did_f"]  = pre_f["treat"]*pre_f["post_f"]
    for dep in ["avgFee","hourlyRate"]:
        try:
            mod=smf.ols(f"{dep} ~ treat+post_f+did_f", data=pre_f).fit(cov_type="HC3")
            c=mod.params["did_f"]; se=mod.bse["did_f"]; p=mod.pvalues["did_f"]
            st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
            placebo_log.append({"FakeYear":fake,"Dep":dep,"DID":c,"p":p,"sig":st})
            print(f"    Fake={fake}  {dep:15s}  DID={c:+10,.1f}  p={p:.3f} {st}")
        except Exception as e:
            print(f"    Fake={fake}  {dep}: ERROR {e}")

print()
warn_shown = False
for r in placebo_log:
    if r["sig"] in ["*","**","***"]:
        if not warn_shown:
            print("  !! Significant placebo effect(s) detected:")
            warn_shown = True
        print(f"     FakeYear={r['FakeYear']}  {r['Dep']}  "
              f"DID={r['DID']:+,.1f}  p={r['p']:.3f} {r['sig']}")
if warn_shown:
    print("  -> Pre-treatment trends differ between groups (B-2 issue).")
    print("     Consider: Callaway-Sant'Anna (2021), group-specific trends,")
    print("     or re-framing KAM (2015) as a first-stage event.")
else:
    print("  -> No significant placebo effects (parallel trends plausible).")

# -- Rolling placebo figure --
roll=[]
for cutoff in range(2014, 2020):
    d = est[est["year"]<2020].copy()
    d["pc"]=(d["year"]>=cutoff).astype(int); d["dc"]=d["treat"]*d["pc"]
    for dep in ["avgFee","hourlyRate"]:
        try:
            mod=smf.ols(f"{dep} ~ treat+pc+dc",data=d).fit(cov_type="HC3")
            c=mod.params["dc"]; se=mod.bse["dc"]
            roll.append({"cutoff":cutoff,"dep":dep,"coef":c,
                         "ci_lo":c-1.96*se,"ci_hi":c+1.96*se})
        except: pass

if roll:
    rp=pd.DataFrame(roll)
    fig3,ax3s=plt.subplots(1,2,figsize=(12,5))
    for ax3,dep in zip(ax3s,["avgFee","hourlyRate"]):
        s=rp[rp["dep"]==dep]
        ax3.fill_between(s["cutoff"],s["ci_lo"],s["ci_hi"],alpha=0.25,color="#d62728")
        ax3.plot(s["cutoff"],s["coef"],"s-",color="#d62728",lw=2,ms=7,label=f"Placebo DID")
        ax3.axhline(0,color="black",lw=0.8,ls="--")
        ax3.set_title(f"Rolling Placebo — {dep}\n(pre-2020 data)", fontsize=11)
        ax3.set_xlabel("Fake Cutoff Year"); ax3.set_ylabel("DID Coefficient")
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:+,.0f}"))
        ax3.legend(); ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(OUT+"fig3_rolling_placebo.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  -> fig3_rolling_placebo.png")


# ════════════════════════════════════════════════════════════
# PART 5  結果サマリー
# ════════════════════════════════════════════════════════════
print("\n[PART 5] Summary")

def fmtd(mod, v="did"):
    c=mod.params[v]; p=mod.pvalues[v]
    st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
    return f"{c:+,.1f} {st} (p={p:.3f})"

lines=[
    "="*65,
    "  RESULTS SUMMARY",
    "="*65,
    "",
    "[S1] Basic DID (all treat vs CompanyAct)",
    f"  avgFee       {fmtd(res_s1['avgFee'])} 千円",
    f"  hourlyRate   {fmtd(res_s1['hourlyRate'])} 円",
    "",
    "[S1p] Primary DID (FIEA_Consol vs CompanyAct)",
    f"  avgFee       {fmtd(res_pp['avgFee'])} 千円",
    f"  hourlyRate   {fmtd(res_pp['hourlyRate'])} 円",
    "",
    "[S2] estimatedHours DID",
    f"  hours        {fmtd(mod_h)} 時間",
    "  [Caution: algebraic dependency (Reviewer B-1)]",
    "",
    "[S3] Log-transformed DID",
    f"  log_avgFee       {fmtd(res_log['log_avgFee'])}",
    f"  log_hourlyRate   {fmtd(res_log['log_hourlyRate'])}",
    f"  log_estHours     {fmtd(res_log['log_estimatedHours'])}",
    "",
    "[Robustness — ALEM key issues]",
    "  B-1 (tautology): report avgFee & hourlyRate as primary;",
    "      estimatedHours as supplementary with explicit caveat.",
    "  B-2 (parallel trends): see fig2 (event study) & fig3 (placebo).",
    "      If pre-trends differ: add group-specific trends (4-2),",
    "      or adopt Callaway-Sant'Anna (2021) staggered DiD.",
    "  KAM 2015: consider modelling as first-stage shock in 2-period DiD.",
    "="*65,
]
txt="\n".join(lines)
print(txt)
with open(OUT+"results_summary.txt","w",encoding="utf-8") as f:
    f.write(txt+"\n")

panel.to_csv(OUT+"panel_data_clean.csv", index=False, encoding="utf-8-sig")

print("\n  Output files:")
for fn in ["panel_data_clean.csv","table1_descriptive.csv",
           "fig1_time_series.png","fig2_event_study.png",
           "fig3_rolling_placebo.png","results_summary.txt"]:
    print(f"   {fn}")
print("\nDone.")
