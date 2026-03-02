# ================================================================
# analysis2_its_registry.py
# ================================================================
# Replication code for:
#   "Judicial Decisions as Legitimizing Mechanisms: Evidence from
#    Japan's Audit Market"
#
# LAYER 1 ANALYSIS — Interrupted Time Series on Registry Data
#   Data : Ministry of Justice monthly commercial registration statistics
#          (FEH_00250002_*.csv, UTF-8)  +  macro indicator CSV (Shift-JIS)
#          Place raw files in data/raw/ or upload to /content/ on Colab
#
#   Analysis 1  Interrupted Time Series (ITS)
#     — Outcome: log(kansayaku)  [auditor-related registration changes]
#     — Control: log(yakuin)     [director-related registration changes]
#     — Interventions: KAM (2015-04), SC Decision (2020-11)
#     — Baseline model + macro-controlled model
#     — Two-stage counterfactual: what would kansayaku look like
#       without the SC ruling? (Fig 2)
#
#   Analysis 4  Monthly Event Study (Announcement Effect)
#     — Window: ±12 months around SC Decision (base = 2020-10)
#     — Applied to ITS residuals (seasonal and KAM trend removed)
#     — Treatment series, control series, and log ratio (Fig 3)
#
#   Key outputs
#     figures/fig2_its_counterfactual.png
#     figures/fig3_event_study.png
#     figures/figA1_registry_timeseries.png
#     results/its_registry_results.txt
# ================================================================
#
# 分析①  Interrupted Time Series (ITS)
#   — 監査人関連登記変更（kansayaku）に対する
#     KAM導入（2015-04）・最高裁判決（2020-11）の
#     水準変化・トレンド変化を推計
#   — 比較対照：役員等に関する変更（yakuin）
#   — マクロ指標（株価・為替・CPI・金利・企業物価）を
#     追加コントロールとして組み込む
#
# 分析④  Announcement Effect（月次イベントスタディ）
#   — 判決月（2020-11）前後±12ヶ月
#   — ITS残差（季節性・KAMトレンド除去済み）に適用
#   — 治療系列・対照系列・比率の3系列で比較
#
# 比較対照の選択根拠
#   yakuin（役員等に関する変更）：
#     件数が最大で統計的に安定、季節性が治療系列と共通、
#     監査規制と無関係、コーポレートガバナンス改革の影響も
#     group-specific trendで制御可能
#   mokuteki（目的の変更）：感度分析に使用
#
# Google Colab での実行方法
#   1. 以下の2ファイルをColab左パネル「ファイル」からアップロード
#        監査関連件数.csv   ← 法務省登記統計
#        マクロ指標.csv     ← 株価・CPI・為替・金利・企業物価
#   2. 必要ライブラリのインストール（下記セルを実行）
#        !pip install statsmodels -q
#   3. このスクリプトをセルに貼り付けて実行
# ================================================================

# ----------------------------------------------------------------
# 0.  ライブラリ
# ----------------------------------------------------------------
import re, warnings, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import rcParams
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
rcParams["font.family"]       = "sans-serif"
rcParams["axes.unicode_minus"] = False
rcParams["axes.spines.top"]   = False
rcParams["axes.spines.right"] = False

# Google Colab では /content/ 以下にファイルが置かれる
BASE      = "/content/"
REG_FILE  = BASE + "監査関連件数.csv"
MACRO_FILE= BASE + "マクロ指標.csv"
OUT       = BASE   # 図・結果はルートに保存

# 介入点
KAM_DATE  = pd.Timestamp("2015-04-01")
SC_DATE   = pd.Timestamp("2020-11-01")
ISQM_DATE = pd.Timestamp("2022-04-01")

print("=" * 65)
print("  Registry + Macro Analysis  (ITS & Announcement Effect)")
print("=" * 65)


# ================================================================
# STEP 1  登記統計の読み込み・整形
# ================================================================
print("\n[STEP 1] Load Registry Data")

raw_reg = pd.read_csv(REG_FILE, encoding="utf-8",
                      header=None, skiprows=13, on_bad_lines="skip")
raw_reg.columns = [
    "code","subcode","unit","time_code","time_sub","yearmonth",
    "cat_label","total","mokuteki","shihonkin",
    "yakuin","torishimariyaku","kansayaku"
]
reg = raw_reg.iloc[4:].copy()

def to_int(s):
    try:    return int(str(s).replace(",","").strip())
    except: return np.nan

def parse_ym(s):
    m = re.match(r"(\d{4})年(\d{1,2})月", str(s))
    if m: return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return pd.NaT

for c in ["total","mokuteki","shihonkin","yakuin","torishimariyaku","kansayaku"]:
    reg[c] = reg[c].apply(to_int)
reg["date"] = reg["yearmonth"].apply(parse_ym)
reg = reg.dropna(subset=["date","kansayaku","yakuin"]).sort_values("date").reset_index(drop=True)

print(f"  Registry: {reg['date'].min():%Y-%m} ~ {reg['date'].max():%Y-%m}  (N={len(reg)})")


# ================================================================
# STEP 2  マクロ指標の読み込み・整形
# ================================================================
print("\n[STEP 2] Load Macro Data")

raw_mac = pd.read_csv(MACRO_FILE, encoding="UTF-8-SIG", header=None)

# col0=日付, col4=日経終値, col8=TOPIX終値,
# col9=CPI, col10=ドル円, col11=貸出金利, col12=企業物価
def to_float(s):
    try:    return float(str(s).replace(",","").strip())
    except: return np.nan

mac_rows = raw_mac[raw_mac[0].str.match(r"\d{4}年\d+月", na=False)].copy()
mac = pd.DataFrame({
    "date":     mac_rows[0].apply(parse_ym).values,
    "nikkei":   mac_rows[4].apply(to_float).values,
    "topix":    mac_rows[8].apply(to_float).values,
    "cpi":      mac_rows[9].apply(to_float).values,
    "usdjpy":   mac_rows[10].apply(to_float).values,
    "loanrate": mac_rows[11].apply(to_float).values,
    "ppi":      mac_rows[12].apply(to_float).values,
})
mac = mac.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

print(f"  Macro:    {mac['date'].min():%Y-%m} ~ {mac['date'].max():%Y-%m}  (N={len(mac)})")
print(f"  Columns:  {list(mac.columns[1:])}")


# ================================================================
# STEP 3  データ統合・分析変数の構築
# ================================================================
print("\n[STEP 3] Merge & Construct Variables")

data = pd.merge(reg, mac, on="date", how="left")

# 共通期間のみ使用
data = data[data["nikkei"].notna()].copy().reset_index(drop=True)
print(f"  Merged:   {data['date'].min():%Y-%m} ~ {data['date'].max():%Y-%m}  (N={len(data)})")

# --- 対数変換 ---
for v in ["kansayaku","yakuin","mokuteki","torishimariyaku",
          "nikkei","topix","cpi","usdjpy"]:
    data[f"log_{v}"] = np.log(data[v].replace(0, np.nan))

data["log_ratio"]  = data["log_kansayaku"] - data["log_yakuin"]

# --- 時間変数 ---
origin   = data["date"].min()
data["t"] = ((data["date"].dt.year  - origin.year)*12
           + (data["date"].dt.month - origin.month))
data["month"] = data["date"].dt.month

# --- 介入ダミー ---
data["post_kam"]  = (data["date"] >= KAM_DATE ).astype(int)
data["post_sc"]   = (data["date"] >= SC_DATE  ).astype(int)
data["post_isqm"] = (data["date"] >= ISQM_DATE).astype(int)

# --- トレンド変化（介入後の経過月数） ---
t_kam = data.loc[data["date"] == KAM_DATE, "t"].values
t_sc  = data.loc[data["date"] == SC_DATE,  "t"].values
t_kam = t_kam[0] if len(t_kam) else data.loc[data["date"] >= KAM_DATE, "t"].min()
t_sc  = t_sc[0]  if len(t_sc)  else data.loc[data["date"] >= SC_DATE,  "t"].min()

data["t_post_kam"] = np.where(data["post_kam"]==1, data["t"] - t_kam, 0)
data["t_post_sc"]  = np.where(data["post_sc"] ==1, data["t"] - t_sc,  0)

# --- 月固定効果ダミー ---
month_fe_vars = []
for m in range(2, 13):
    vn = f"m{m:02d}"
    data[vn] = (data["month"] == m).astype(int)
    month_fe_vars.append(vn)
month_fe = " + ".join(month_fe_vars)

# --- マクロコントロール（1期ラグ・標準化） ---
for v in ["log_nikkei","log_cpi","log_usdjpy","loanrate","ppi"]:
    data[f"{v}_lag1"] = data[v].shift(1)
    mu = data[f"{v}_lag1"].mean()
    sd = data[f"{v}_lag1"].std()
    data[f"{v}_std"]  = (data[f"{v}_lag1"] - mu) / (sd if sd > 0 else 1)

macro_ctrl = " + ".join([f"{v}_std" for v in
                         ["log_nikkei","log_cpi","log_usdjpy","loanrate","ppi"]])

print(f"  Macro controls (standardized lag-1): {macro_ctrl}")


# ================================================================
# STEP 4  記述統計・推移グラフ
# ================================================================
print("\n[STEP 4] Descriptive Plots")

fig, axes = plt.subplots(3, 1, figsize=(13, 13))
fig.suptitle("Monthly Registry Changes & Macro Indicators (Japan)",
             fontsize=13, y=1.001)

VLINES = [
    (KAM_DATE,  "gray",  ":",  "KAM intro (2015-04)"),
    (SC_DATE,   "red",   "--", "SC Decision (2020-11)"),
    (ISQM_DATE, "navy",  "-.", "ISQM reform (2022-04)"),
]

# Panel A: 治療系列 vs 対照系列（水準）
ax  = axes[0]
ax2 = ax.twinx()
ax.fill_between(data["date"], data["kansayaku"], alpha=0.15, color="#d62728")
ax.plot(data["date"], data["kansayaku"],
        color="#d62728", lw=2, label="Kansayaku (Auditor-related) — left")
ax2.plot(data["date"], data["yakuin"],
         color="#1f77b4", lw=1.5, ls="--", alpha=0.7,
         label="Yakuin (Directors) — right")
for dt, col, ls, lbl in VLINES:
    ax.axvline(dt, color=col, ls=ls, lw=1.4, alpha=0.85)
ax.set_title("Panel A: Registration Changes — Treatment vs Control (Level)", fontsize=11)
ax.set_ylabel("Kansayaku", color="#d62728")
ax2.set_ylabel("Yakuin",   color="#1f77b4")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
h1,l1 = ax.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Panel B: 対数系列 + 比率
ax = axes[1]
ax3 = ax.twinx()
for v, col, lbl, lw in [
    ("log_kansayaku", "#d62728", "log(Kansayaku)", 2.0),
    ("log_yakuin",    "#1f77b4", "log(Yakuin)",    1.5),
    ("log_mokuteki",  "#2ca02c", "log(Mokuteki)",  1.2),
]:
    ax.plot(data["date"], data[v], color=col, lw=lw, label=lbl, alpha=0.85)
ax3.plot(data["date"], data["log_ratio"], color="#9467bd", lw=1.8,
         ls="-.", alpha=0.8, label="log(Ratio) — right")
for dt, col, ls, _ in VLINES:
    ax.axvline(dt, color=col, ls=ls, lw=1.4, alpha=0.85)
ax.set_title("Panel B: Log-transformed Series & Ratio", fontsize=11)
ax.set_ylabel("log(cases)")
ax3.set_ylabel("log(Kansayaku/Yakuin)", color="#9467bd")
h1,l1 = ax.get_legend_handles_labels()
h3,l3 = ax3.get_legend_handles_labels()
ax.legend(h1+h3, l1+l3, fontsize=8, ncol=2, loc="upper left")
ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Panel C: マクロ指標（標準化）
ax = axes[2]
macro_plot = [
    ("log_nikkei", "#e377c2", "log(Nikkei)"),
    ("log_usdjpy", "#8c564b", "log(USD/JPY)"),
    ("log_cpi",    "#17becf", "log(CPI)"),
    ("loanrate",   "#bcbd22", "Loan rate"),
]
for v, col, lbl in macro_plot:
    s = data[v].dropna()
    s_z = (s - s.mean()) / s.std()
    ax.plot(data.loc[s.index, "date"], s_z, color=col, lw=1.4,
            label=lbl, alpha=0.85)
for dt, col, ls, lbl in VLINES:
    ax.axvline(dt, color=col, ls=ls, lw=1.4, alpha=0.85, label=lbl)
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.set_title("Panel C: Macro Indicators (standardized)", fontsize=11)
ax.set_ylabel("Standardized value (z-score)")
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig.savefig(OUT + "fig_reg_A_timeseries.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("  -> fig_reg_A_timeseries.png")


# ================================================================
# STEP 5  分析①  Interrupted Time Series (ITS)
# ================================================================
print("\n[STEP 5] Analysis ①: ITS")
print("  Y_t = α + β1·t + β2·PostKAM + β3·t×PostKAM")
print("           + β4·PostSC  + β5·t×PostSC")
print("           + Σ month_FE  [+ Macro controls]")

def run_its(depvar, label, df, add_macro=False):
    macro_part = f" + {macro_ctrl}" if add_macro else ""
    fml = (f"{depvar} ~ t + post_kam + t_post_kam"
           f" + post_sc + t_post_sc + {month_fe}{macro_part}")
    mod = smf.ols(fml, data=df.dropna(subset=[depvar]+
          ([v+"_std" for v in ["log_nikkei","log_cpi",
            "log_usdjpy","loanrate","ppi"]] if add_macro else [])
          )).fit(cov_type="HC3")

    tag = "(+Macro)" if add_macro else "       "
    print(f"\n  [{label}] {tag}  N={mod.nobs:.0f}  R²={mod.rsquared:.3f}")
    rows = [
        ("Baseline trend (t)",      "t"),
        ("KAM level change (β2)",   "post_kam"),
        ("KAM trend change (β3)",   "t_post_kam"),
        ("SC  level change (β4)",   "post_sc"),
        ("SC  trend change (β5)",   "t_post_sc"),
    ]
    res = {}
    for name, var in rows:
        if var in mod.params:
            c=mod.params[var]; p=mod.pvalues[var]
            st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else "  "))
            pct = f"({(np.exp(c)-1)*100:+.1f}%)" if depvar.startswith("log") else ""
            print(f"    {name:35s} {c:+.4f} {pct:10s} p={p:.3f} {st}")
            res[var] = (c, p, st)
    return mod, res

# ---- 各系列 ×（マクロなし / マクロあり）----
its_results = {}
for depvar, lbl in [
    ("log_kansayaku",       "Kansayaku [TREAT]"),
    ("log_yakuin",          "Yakuin     [CTRL1]"),
    ("log_mokuteki",        "Mokuteki   [CTRL2]"),
    ("log_torishimariyaku", "Torishimariyaku [CTRL3]"),
    ("log_ratio",           "log(Ratio)  [KANSA/YAKUIN]"),
]:
    mod_base,  res_base  = run_its(depvar, lbl, data, add_macro=False)
    mod_macro, res_macro = run_its(depvar, lbl, data, add_macro=True)
    its_results[depvar] = {
        "base":  (mod_base,  res_base),
        "macro": (mod_macro, res_macro),
    }


# ================================================================
# STEP 6  ITS 反事実図（マクロなし・マクロあり比較）
# ================================================================
print("\n[STEP 6] ITS Counterfactual Plot")

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle("ITS: Actual vs Counterfactual (shaded = SC effect gap)",
              fontsize=12)

plot_items = [
    ("log_kansayaku",       "log(Kansayaku)\n[Treatment]",      "#d62728"),
    ("log_yakuin",          "log(Yakuin)\n[Control — Directors]","#1f77b4"),
    ("log_torishimariyaku", "log(Torishimariyaku)\n[Control — Board]","#ff7f0e"),
    ("log_ratio",           "log(Kansayaku/Yakuin)\n[Ratio]",   "#9467bd"),
    ("log_mokuteki",        "log(Mokuteki)\n[Control — Purpose]","#2ca02c"),
]

for i, (depvar, title, color) in enumerate(plot_items):
    row, col = divmod(i, 3)
    ax = axes2[row, col]

    for spec, ls, lw, lbl in [("base","--",1.4,"ITS (no macro)"),
                               ("macro","-",1.8,"ITS (+macro ctrl)")]:
        mod = its_results[depvar][spec][0]
        cf  = data.copy()
        cf["post_sc"] = 0; cf["t_post_sc"] = 0
        fitted  = mod.predict(data)
        cf_pred = mod.predict(cf)

        ax.scatter(data["date"], data[depvar],
                   s=6, alpha=0.45, color=color, zorder=4)
        ax.plot(data["date"], fitted,  color=color,  lw=lw, ls=ls,  alpha=0.9, label=lbl)
        ax.plot(data["date"], cf_pred, color="gray", lw=lw, ls=":", alpha=0.7)

    # SC後のギャップに網かけ（macroモデルで）
    mod_m = its_results[depvar]["macro"][0]
    cf2 = data.copy(); cf2["post_sc"]=0; cf2["t_post_sc"]=0
    fitted_m  = mod_m.predict(data)
    cf_pred_m = mod_m.predict(cf2)
    post_mask = data["date"] >= SC_DATE
    ax.fill_between(data.loc[post_mask,"date"],
                    fitted_m[post_mask], cf_pred_m[post_mask],
                    alpha=0.2, color=color, label="SC gap (+macro)")

    ax.axvline(KAM_DATE, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.axvline(SC_DATE,  color="red",  ls="--",lw=1.4, alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("log(cases)"); ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))

# 空きセルを非表示
axes2[1, 2].set_visible(False)
plt.tight_layout()
fig2.savefig(OUT + "fig_reg_B_its_counterfactual.png",
             dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("  -> fig_reg_B_its_counterfactual.png")


# ================================================================
# STEP 7  分析④  月次イベントスタディ（アナウンスメント効果）
# ================================================================
print("\n[STEP 7] Analysis ④: Monthly Event Study (±12m around SC Decision)")

WINDOW = 12
BASE_K = -1   # 基準月（判決の1ヶ月前）

data["event_t"] = ((data["date"].dt.year  - SC_DATE.year)*12
                 + (data["date"].dt.month - SC_DATE.month))

def run_event_study(depvar, label, add_macro=False, window=WINDOW):
    """
    ITS残差（季節性＋KAMトレンド除去済み）を従属変数として
    判決前後windowヶ月のダミーを推計
    """
    # Step 1: KAM以前のデータでベースラインITSを推計
    macro_part = f" + {macro_ctrl}" if add_macro else ""
    fml_pre = f"{depvar} ~ t + post_kam + t_post_kam + {month_fe}{macro_part}"
    sub_pre = data[data["date"] < SC_DATE].dropna(
        subset=[depvar] +
        ([v+"_std" for v in ["log_nikkei","log_cpi","log_usdjpy","loanrate","ppi"]]
         if add_macro else [])
    )
    mod_pre = smf.ols(fml_pre, data=sub_pre).fit(cov_type="HC3")
    data["resid_its"] = data[depvar] - mod_pre.predict(data)

    # Step 2: 残差にイベントダミーを回帰
    ev_vars = []
    for k in range(-window, window+1):
        if k == BASE_K: continue
        vn = f"ev_{'p' if k>=0 else 'n'}{abs(k):02d}"
        data[vn] = (data["event_t"] == k).astype(int)
        ev_vars.append((k, vn))

    fml_ev = "resid_its ~ " + " + ".join(vn for _, vn in ev_vars)
    mod_ev = smf.ols(fml_ev, data=data).fit(cov_type="HC3")

    rows = [{"k": BASE_K, "coef": 0., "ci_lo": 0., "ci_hi": 0., "p": np.nan}]
    for k, vn in ev_vars:
        if vn in mod_ev.params:
            c=mod_ev.params[vn]; se=mod_ev.bse[vn]; p=mod_ev.pvalues[vn]
            rows.append({"k":k,"coef":c,"ci_lo":c-1.96*se,"ci_hi":c+1.96*se,"p":p})
    df_ev = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    tag = "(+Macro)" if add_macro else "       "
    print(f"\n  [{label}] {tag}")
    sig = df_ev[(df_ev["p"] < 0.10) & (df_ev["k"] >= 0)]
    if len(sig):
        print("    Significant post-SC months (p<0.10):")
        for _, r in sig.iterrows():
            st="***" if r['p']<.01 else("**" if r['p']<.05 else "*")
            print(f"      k={int(r['k']):+3d}  coef={r['coef']:+.4f}  p={r['p']:.3f} {st}")
    else:
        print("    No significant post-SC effects detected (p<0.10)")
    return df_ev

ev_results = {}
for depvar, lbl in [
    ("log_kansayaku", "Kansayaku [TREAT]"),
    ("log_yakuin",    "Yakuin     [CTRL]"),
    ("log_ratio",     "log(Ratio)"),
]:
    ev_base  = run_event_study(depvar, lbl, add_macro=False)
    ev_macro = run_event_study(depvar, lbl, add_macro=True)
    ev_results[depvar] = {"base": ev_base, "macro": ev_macro}


# ---- イベントスタディ図 ----------------------------------------
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
fig3.suptitle(
    "Monthly Event Study: ITS Residuals around SC Decision (2020-11)\n"
    "Top row = no macro controls  |  Bottom row = +macro controls\n"
    "Base month = -1 (Oct 2020)",
    fontsize=11)

ev_items = [
    ("log_kansayaku", "Kansayaku\n[Treatment]",            "#d62728"),
    ("log_yakuin",    "Yakuin\n[Control — Directors]",     "#1f77b4"),
    ("log_ratio",     "log(Kansayaku/Yakuin)\n[Ratio]",    "#9467bd"),
]

for row_i, spec in enumerate(["base", "macro"]):
    for col_i, (depvar, title, color) in enumerate(ev_items):
        ax = axes3[row_i, col_i]
        df = ev_results[depvar][spec]

        pre  = df[df["k"] <  0]
        post = df[df["k"] >= 0]

        ax.fill_between(pre["k"],  pre["ci_lo"],  pre["ci_hi"],
                        alpha=0.18, color="gray")
        ax.plot(pre["k"],  pre["coef"],  "o-", color="gray",
                lw=1.8, ms=4, label="Pre-SC")
        ax.fill_between(post["k"], post["ci_lo"], post["ci_hi"],
                        alpha=0.22, color=color)
        ax.plot(post["k"], post["coef"], "o-", color=color,
                lw=2.2, ms=5, label="Post-SC")

        ax.axhline(0,    color="black", lw=0.7, ls="--")
        ax.axvline(-0.5, color="red",   lw=1.4, ls="--",
                   alpha=0.85, label="SC Decision")

        # 有意点マーク
        sig = df[(df["p"] < 0.10) & (df["k"] != BASE_K)]
        for _, r in sig.iterrows():
            mk = ("***" if r["p"]<.01 else ("**" if r["p"]<.05 else "*"))
            ax.annotate(mk, xy=(r["k"], r["coef"]),
                        xytext=(0,6), textcoords="offset points",
                        ha="center", fontsize=8, color=color)

        spec_lbl = "No macro controls" if spec=="base" else "+Macro controls"
        ax.set_title(f"{title}\n({spec_lbl})", fontsize=10)
        ax.set_xlabel("Months relative to SC Decision")
        ax.set_ylabel("Coeff. (log-scale ITS residual)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xticks(range(-WINDOW, WINDOW+1, 3))

plt.tight_layout()
fig3.savefig(OUT + "fig_reg_C_event_study.png",
             dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("\n  -> fig_reg_C_event_study.png")


# ================================================================
# STEP 8  DID型アナウンスメント効果（治療 vs 対照 パネル）
# ================================================================
print("\n[STEP 8] DID-type Announcement Effect (Panel)")
print("  Panel: Kansayaku (treat=1) vs Yakuin (treat=0)")

base_cols  = ["date","t","post_sc","post_kam","t_post_kam","month"]
macro_cols = [v+"_std" for v in
              ["log_nikkei","log_cpi","log_usdjpy","loanrate","ppi"]]

def make_panel(data):
    tk = data[base_cols + macro_cols + ["log_kansayaku"]].copy()
    tk.rename(columns={"log_kansayaku":"logY"}, inplace=True)
    tk["treat"] = 1
    ty = data[base_cols + macro_cols + ["log_yakuin"]].copy()
    ty.rename(columns={"log_yakuin":"logY"}, inplace=True)
    ty["treat"] = 0
    p = pd.concat([tk,ty], ignore_index=True)
    p["did"]     = p["treat"] * p["post_sc"]
    p["did_kam"] = p["treat"] * p["post_kam"]
    for m in range(2,13):
        p[f"m{m:02d}"] = (p["month"]==m).astype(int)
    return p

panel = make_panel(data)
mfe   = " + ".join([f"m{m:02d}" for m in range(2,13)])

models = {
    "M1 Basic DID":
        smf.ols(f"logY ~ treat+post_sc+did+{mfe}", data=panel),
    "M2 Two-stage DID (KAM+SC)":
        smf.ols(f"logY ~ treat+post_kam+did_kam+post_sc+did+{mfe}", data=panel),
    "M3 DID + trend":
        smf.ols(f"logY ~ treat+post_sc+did+t+post_kam+{mfe}", data=panel),
    "M4 DID + macro":
        smf.ols(f"logY ~ treat+post_sc+did+{mfe}+{macro_ctrl}", data=panel),
    "M5 Two-stage + macro":
        smf.ols(f"logY ~ treat+post_kam+did_kam+post_sc+did+{mfe}+{macro_ctrl}",
                data=panel),
}

did_summary = []
for mname, fml_obj in models.items():
    mod = fml_obj.fit(cov_type="HC3")
    print(f"\n  {mname}  (R²={mod.rsquared:.3f})")
    row = {"Model": mname}
    for var, nm in [("did","DID_SC"), ("did_kam","DID_KAM")]:
        if var in mod.params:
            c=mod.params[var]; p=mod.pvalues[var]
            st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
            pct=(np.exp(c)-1)*100
            print(f"    {nm:12s}: coef={c:+.4f} ({pct:+.1f}%)  p={p:.3f} {st}")
            row[nm+"_coef"]=c; row[nm+"_p"]=p; row[nm+"_sig"]=st
    did_summary.append(row)

did_df = pd.DataFrame(did_summary)


# ================================================================
# STEP 9  平行トレンド確認図
# ================================================================
print("\n[STEP 9] Parallel Trends Check")

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("Parallel Trends: Kansayaku vs Yakuin", fontsize=12)

ma12_k = data.set_index("date")["log_kansayaku"].rolling(12, center=True).mean()
ma12_y = data.set_index("date")["log_yakuin"].rolling(12, center=True).mean()

ax = axes4[0]
ax.plot(ma12_k.index, ma12_k, color="#d62728", lw=2, label="log(Kansayaku) 12m-MA")
ax.plot(ma12_y.index, ma12_y, color="#1f77b4", lw=2, label="log(Yakuin)    12m-MA")
for dt, col, ls, lbl in VLINES:
    ax.axvline(dt, color=col, ls=ls, lw=1.4, alpha=0.85, label=lbl)
ax.set_title("Level (12-month moving average)")
ax.set_ylabel("log(cases)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

ax = axes4[1]
diff = ma12_k - ma12_y
pre_kam_mean = diff[diff.index < KAM_DATE].mean()
ax.plot(diff.index, diff.values, color="#9467bd", lw=2,
        label="log(Kansayaku) − log(Yakuin)")
ax.axhline(pre_kam_mean, color="gray", ls="--", lw=1,
           label=f"Pre-KAM mean = {pre_kam_mean:.3f}")
for dt, col, ls, lbl in VLINES:
    ax.axvline(dt, color=col, ls=ls, lw=1.4, alpha=0.85)
ax.set_title("Difference Series\n(Flat pre-2015 → parallel trends hold)", fontsize=11)
ax.set_ylabel("log-difference"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
fig4.savefig(OUT + "fig_reg_D_parallel_trends.png",
             dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("  -> fig_reg_D_parallel_trends.png")


# ================================================================
# STEP 10  結果サマリー出力
# ================================================================
print("\n[STEP 10] Results Summary")

def fmt_its(res, var, depvar="log"):
    r = res.get(var)
    if r is None: return "n/a"
    c,p,st = r
    pct = f"({(np.exp(c)-1)*100:+.1f}%)" if depvar.startswith("log") else ""
    return f"{c:+.4f} {pct} {st} [p={p:.3f}]"

# ITSのpost_sc係数（baseとmacro）
def its_sc(key, spec="base"):
    res = its_results[key][spec][1]
    return fmt_its(res, "post_sc", key)

lines = [
    "="*65,
    "  RESULTS SUMMARY — Registry + Macro Analysis",
    "="*65, "",
    "[Analysis ①] ITS: SC level change (post_sc β4)",
    "  Series             No-macro           +Macro controls",
    f"  Kansayaku [TREAT] {its_sc('log_kansayaku','base'):30s} {its_sc('log_kansayaku','macro')}",
    f"  Yakuin    [CTRL1] {its_sc('log_yakuin','base'):30s} {its_sc('log_yakuin','macro')}",
    f"  Mokuteki  [CTRL2] {its_sc('log_mokuteki','base'):30s} {its_sc('log_mokuteki','macro')}",
    f"  log(Ratio)        {its_sc('log_ratio','base'):30s} {its_sc('log_ratio','macro')}",
    "",
    "[Analysis ④] DID: Kansayaku vs Yakuin",
]
for row in did_summary:
    sc = row.get("DID_SC_coef", np.nan)
    sp = row.get("DID_SC_p",   np.nan)
    ss = row.get("DID_SC_sig", "")
    km = row.get("DID_KAM_coef", None)
    kp = row.get("DID_KAM_p",   None)
    ks = row.get("DID_KAM_sig", "")
    sc_str = f"DID_SC={sc:+.4f}({(np.exp(sc)-1)*100:+.1f}%) {ss}[p={sp:.3f}]" if not np.isnan(sc) else ""
    km_str = f" | DID_KAM={km:+.4f}({(np.exp(km)-1)*100:+.1f}%) {ks}[p={kp:.3f}]" if km is not None else ""
    lines.append(f"  {row['Model']:30s} {sc_str}{km_str}")

lines += [
    "",
    "[Macro controls (lag-1, standardized)]",
    f"  {macro_ctrl}",
    "",
    "[Interpretation]",
    "  ITS post_sc (Kansayaku) > 0 & sig",
    "    -> Auditor-related changes increased after SC Decision",
    "    -> Consistent with small-firm exit / auditor rotation pressure",
    "  ITS post_sc (Yakuin/Mokuteki) n.s.",
    "    -> Effect is specific to audit regulation, not macro/general trend",
    "  DID_SC < 0 & sig  -> Kansayaku fell relative to Yakuin (consolidation)",
    "  DID_SC > 0 & sig  -> Kansayaku rose relative to Yakuin (more turnover)",
    "  Stable DID after adding macro ctrl -> macro factors not confounding",
    "="*65,
]
txt = "\n".join(lines)
print(txt)

with open(OUT + "registry_results_summary.txt","w",encoding="utf-8") as f:
    f.write(txt + "\n")

data.to_csv(OUT + "registry_panel_clean.csv", index=False, encoding="utf-8-sig")
did_df.to_csv(OUT + "did_summary_table.csv",  index=False, encoding="utf-8-sig")

print("\n  Output files saved to /content/:")
for fn in ["fig_reg_A_timeseries.png",
           "fig_reg_B_its_counterfactual.png",
           "fig_reg_C_event_study.png",
           "fig_reg_D_parallel_trends.png",
           "registry_results_summary.txt",
           "registry_panel_clean.csv",
           "did_summary_table.csv"]:
    print(f"   {fn}")
print("\nDone.")
