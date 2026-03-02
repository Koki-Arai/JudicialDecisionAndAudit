# ================================================================
# analysis3_var_bridge.py
# ================================================================
# Replication code for:
#   "Judicial Decisions as Legitimizing Mechanisms: Evidence from
#    Japan's Audit Market"
#
# BRIDGE ANALYSIS — VAR/Granger Causality + Layer 1 × Layer 2 Integration
#   Data : (1) Ministry of Justice monthly registry CSV  [Layer 1]
#          (2) JICPA Survey of Audit Practices CSV       [Layer 2]
#          Place raw files in data/raw/ or upload to /content/ on Colab
#
#   Analysis 2  VAR(6) + Granger Causality (Table 6a, 6b)
#     — Tests whether torishimariyaku (board-related changes) Granger-
#       causes kansayaku (auditor-related changes)
#     — VAR estimated on HP-cycle-detrended log-differenced monthly series
#     — Sub-period split at SC Decision (pre: 2009–2020, post: 2020–2025)
#     — Impulse response functions (Fig 4) and FEVD (Fig A3)
#
#   Analysis 3  Integrated Registry × JICPA Panel (Table 5)
#     — Annual registry turnover rate merged into JICPA DID panel
#     — Model D: hourlyRate DID + kansa_yoy  → bridge channel test
#     — Model E: two-stage mediation (kansa_yoy → avgFee pathway)
#     — Fig A4: integrated 4-panel visualization
#
#   Key outputs
#     figures/fig4_var_irf.png
#     figures/figA3_var_fevd.png
#     figures/figA4_registry_jicpa_integrated.png
#     results/var_bridge_results.txt
# ================================================================
#
# 【論文再整理の位置づけ】
#   JICPA調査データ（年次）= 大手法人の価格・時間調整を捉える
#   登記統計データ（月次）= 小規模事務所の市場退出を捉える
#   この2つを繋ぐのが分析②と分析③
#
# 分析②  VAR（ベクトル自己回帰）+ グレンジャー因果検定
#   — 機関設計変更（torishimariyaku）が先行し
#     監査人交代（kansayaku）が後続するか
#   — 「機関設計変更 → 監査人再選定 → 登記変更」の
#     伝播メカニズムを検証
#   — 介入前後でVARを分割推計し、構造変化を確認
#
# 分析③  登記統計とJICPAデータの統合分析
#   — 登記変化率（市場混乱度の代理変数）を年次集計して
#     JICPAパネルにマージ
#   — 「登記変化が大きい年 = 市場再編圧力が高い年」に
#     価格・時間調整も大きいかを検証
#   — 二段階推計（登記変化 → 費用変化）で
#     市場再編が価格調整を媒介するかを確認
#
# Google Colab での実行方法
#   1. 以下の3ファイルをColabにアップロード
#        監査関連件数.csv
#        マクロ指標.csv
#        （registry_analysis_colab.py の実行後に生成される）
#        registry_panel_clean.csv  ← 既に生成済みならアップロード
#        （なければ本コード内で再生成します）
#   2. !pip install statsmodels -q
#   3. このスクリプトを実行
# ================================================================

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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy import stats

warnings.filterwarnings("ignore")
rcParams["font.family"]        = "sans-serif"
rcParams["axes.unicode_minus"] = False
rcParams["axes.spines.top"]    = False
rcParams["axes.spines.right"]  = False

BASE       = "/content/"
REG_FILE   = BASE + "監査関連件数.csv"
MACRO_FILE = BASE + "マクロ指標.csv"
OUT        = BASE

KAM_DATE   = pd.Timestamp("2015-04-01")
SC_DATE    = pd.Timestamp("2020-11-01")
ISQM_DATE  = pd.Timestamp("2022-04-01")

print("=" * 65)
print("  Analysis ② VAR/Granger  &  Analysis ③ Integrated DID")
print("  Two-layer market mechanism: Large firms vs Small firms")
print("=" * 65)


# ================================================================
# STEP 0  データ再構築（registry_analysis_colab.py と共通処理）
# ================================================================
print("\n[STEP 0] Reconstruct Panel Data")

# ---- 登記統計 ----
def parse_ym(s):
    m = re.match(r"(\d{4})年(\d{1,2})月", str(s))
    if m: return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return pd.NaT

def to_int(s):
    try:    return int(str(s).replace(",","").strip())
    except: return np.nan

def to_float(s):
    try:    return float(str(s).replace(",","").strip())
    except: return np.nan

raw_reg = pd.read_csv(REG_FILE, encoding="utf-8",
                      header=None, skiprows=13, on_bad_lines="skip")
raw_reg.columns = [
    "code","subcode","unit","time_code","time_sub","yearmonth",
    "cat_label","total","mokuteki","shihonkin",
    "yakuin","torishimariyaku","kansayaku"
]
reg = raw_reg.iloc[4:].copy()
for c in ["total","mokuteki","shihonkin","yakuin","torishimariyaku","kansayaku"]:
    reg[c] = reg[c].apply(to_int)
reg["date"] = reg["yearmonth"].apply(parse_ym)
reg = reg.dropna(subset=["date","kansayaku","yakuin"]).sort_values("date").reset_index(drop=True)

# ---- マクロ指標 ----
raw_mac = pd.read_csv(MACRO_FILE, encoding="UTF-8-SIG", header=None)
mac_rows = raw_mac[raw_mac[0].str.match(r"\d{4}年\d+月", na=False)].copy()
mac = pd.DataFrame({
    "date":     mac_rows[0].apply(parse_ym).values,
    "nikkei":   mac_rows[4].apply(to_float).values,
    "topix":    mac_rows[8].apply(to_float).values,
    "cpi":      mac_rows[9].apply(to_float).values,
    "usdjpy":   mac_rows[10].apply(to_float).values,
    "loanrate": mac_rows[11].apply(to_float).values,
    "ppi":      mac_rows[12].apply(to_float).values,
}).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# ---- 統合・変数構築 ----
monthly = pd.merge(reg, mac, on="date", how="left")
monthly = monthly[monthly["nikkei"].notna()].copy().reset_index(drop=True)

for v in ["kansayaku","yakuin","mokuteki","torishimariyaku","nikkei","cpi","usdjpy"]:
    monthly[f"log_{v}"] = np.log(monthly[v].replace(0, np.nan))
monthly["log_ratio"]  = monthly["log_kansayaku"] - monthly["log_yakuin"]

origin = monthly["date"].min()
monthly["t"]          = ((monthly["date"].dt.year - origin.year)*12
                        +(monthly["date"].dt.month - origin.month))
monthly["month"]      = monthly["date"].dt.month
monthly["post_kam"]   = (monthly["date"] >= KAM_DATE ).astype(int)
monthly["post_sc"]    = (monthly["date"] >= SC_DATE  ).astype(int)
monthly["post_isqm"]  = (monthly["date"] >= ISQM_DATE).astype(int)
monthly["year"]       = monthly["date"].dt.year

print(f"  Monthly panel: {monthly['date'].min():%Y-%m} ~ {monthly['date'].max():%Y-%m}  N={len(monthly)}")

# ---- JICPAデータ（audit_did_analysis.py の出力があれば使用、なければ直接構築）----
# 監査実施状況データの列位置:
#   col0=監査区分, col1=会社数, col14=1社当たり平均報酬(千円), col15=時間当たり単価(円)
AUDIT_FILE = BASE + "監査実施状況2013_2023.csv"

def load_jicpa(path):
    raw = pd.read_csv(path, encoding="shift_jis", header=None)
    YEAR_ROWS = []
    for i, row in raw.iterrows():
        m2 = re.match(r"^(\d{4})年度", str(row[0]).strip())
        if m2:
            YEAR_ROWS.append((int(m2.group(1)), i))
    YEAR_ROWS = sorted(YEAR_ROWS, key=lambda x: x[1])

    def to_cat(s):
        s = re.sub(r"[\s\u3000]+", "", str(s).strip())
        if re.search(r"金商法.*連結", s): return "FIEA_Consol"
        if re.search(r"金商法.*個別", s): return "FIEA_Indiv"
        if s == "会社法":                  return "CompanyAct"
        return None

    records = []
    for idx, (yr, sr) in enumerate(YEAR_ROWS):
        er = YEAR_ROWS[idx+1][1] if idx+1 < len(YEAR_ROWS) else len(raw)
        for _, row in raw.iloc[sr:er].iterrows():
            cat = to_cat(row[0])
            if cat is None: continue
            fee  = to_float(row[14])
            rate = to_float(row[15])
            hours = (fee*1000/rate) if (pd.notna(fee) and pd.notna(rate) and rate>0) else np.nan
            records.append({"year":yr,"category":cat,"avgFee":fee,
                            "hourlyRate":rate,"estimatedHours":hours,
                            "n_companies":to_float(row[1])})
    panel = pd.DataFrame(records).sort_values(["category","year"]).reset_index(drop=True)
    panel["treat"]    = panel["category"].isin(["FIEA_Consol","FIEA_Indiv"]).astype(int)
    panel["post2020"] = (panel["year"] >= 2020).astype(int)
    panel["did"]      = panel["treat"] * panel["post2020"]
    for v in ["avgFee","hourlyRate","estimatedHours"]:
        panel[f"log_{v}"] = np.log(panel[v].replace(0, np.nan))
    return panel

try:
    jicpa = load_jicpa(AUDIT_FILE)
    print(f"  JICPA panel:  {jicpa['year'].min()} ~ {jicpa['year'].max()}  N={len(jicpa)}")
    HAS_JICPA = True
except Exception as e:
    print(f"  JICPA file not found ({e}) — Analysis ③ will run with simulated merge")
    HAS_JICPA = False


# ================================================================
# STEP 1  単位根検定（VAR前の前処理）
# ================================================================
print("\n[STEP 1] Unit Root Tests (ADF) — Pre-processing for VAR")

vars_for_var = ["log_kansayaku", "log_yakuin",
                "log_torishimariyaku", "log_ratio"]

adf_results = {}
print(f"  {'Variable':30s}  {'ADF stat':>9}  {'p-value':>8}  {'I(0)?':>6}")
print("  " + "-"*60)
for v in vars_for_var:
    s = monthly[v].dropna()
    res = adfuller(s, autolag="AIC")
    stat, p = res[0], res[1]
    stationary = "Yes" if p < 0.05 else "No "
    print(f"  {v:30s}  {stat:9.3f}  {p:8.4f}  {stationary}")
    adf_results[v] = (stat, p, stationary)

# 差分系列でも確認
print(f"\n  {'Variable (1st diff)':30s}  {'ADF stat':>9}  {'p-value':>8}  {'I(0)?':>6}")
print("  " + "-"*60)
diff_vars = {}
for v in vars_for_var:
    s = monthly[v].dropna().diff().dropna()
    res = adfuller(s, autolag="AIC")
    stat, p = res[0], res[1]
    stationary = "Yes" if p < 0.05 else "No "
    diff_vars[f"d_{v}"] = monthly[v].diff()
    print(f"  d_{v:27s}  {stat:9.3f}  {p:8.4f}  {stationary}")


# ================================================================
# STEP 2  分析②  VAR + グレンジャー因果検定
# ================================================================
print("\n[STEP 2] Analysis ②: VAR & Granger Causality")
print("  Hypothesis: torishimariyaku (board changes) Granger-causes")
print("              kansayaku (auditor changes)")
print("  — 機関設計変更が先行し監査人交代を誘発するか")

# HP フィルタで季節性・トレンド除去済みサイクル成分を使用
def get_cycle(series, lamb=14400):  # 月次: λ=14400 が標準
    clean = series.dropna()
    _, cycle = hpfilter(clean, lamb=lamb)
    return cycle

# 全期間・介入前後の3パターンで推計
periods = {
    "Full (2009-2025)":
        monthly,
    "Pre-SC (2009-2020)":
        monthly[monthly["date"] < SC_DATE],
    "Post-SC (2020-2025)":
        monthly[monthly["date"] >= SC_DATE],
}

var_results    = {}
granger_results = {}

for period_name, df_sub in periods.items():
    print(f"\n  --- Period: {period_name} (N={len(df_sub)}) ---")

    # サイクル成分を抽出
    try:
        c_kansa = get_cycle(df_sub["log_kansayaku"].dropna().values)
        c_tori  = get_cycle(df_sub["log_torishimariyaku"].dropna().values)
        c_yakuin= get_cycle(df_sub["log_yakuin"].dropna().values)
    except Exception as e:
        print(f"    HP filter failed: {e}, using differenced series")
        c_kansa = df_sub["log_kansayaku"].diff().dropna().values
        c_tori  = df_sub["log_torishimariyaku"].diff().dropna().values
        c_yakuin= df_sub["log_yakuin"].diff().dropna().values

    n = min(len(c_kansa), len(c_tori), len(c_yakuin))
    var_df = pd.DataFrame({
        "kansa": c_kansa[-n:],
        "tori":  c_tori[-n:],
        "yakuin": c_yakuin[-n:],
    }).dropna()

    # VAR ラグ次数選択（最大12まで）
    model = VAR(var_df)
    ic_res = model.select_order(maxlags=12)
    best_lag = ic_res.aic
    best_lag = max(1, min(best_lag, 6))  # 1〜6に制限
    print(f"    Optimal lag (AIC): {best_lag}")

    # VAR推計
    result = model.fit(best_lag)
    var_results[period_name] = result

    # グレンジャー因果: tori → kansa（「機関設計変更が監査人変更を引き起こすか」）
    print(f"    Granger Causality: torishimariyaku → kansayaku")
    gc_data = var_df[["kansa","tori"]].dropna()
    gc_res  = grangercausalitytests(gc_data, maxlag=best_lag, verbose=False)
    for lag in range(1, best_lag+1):
        f_test = gc_res[lag][0]["ssr_ftest"]
        f_stat, p_val = f_test[0], f_test[1]
        st = "***" if p_val<.01 else("**" if p_val<.05 else("*" if p_val<.10 else ""))
        print(f"      lag={lag}  F={f_stat:.3f}  p={p_val:.3f} {st}")

    # 逆方向: kansa → tori（因果の方向性確認）
    print(f"    Granger Causality: kansayaku → torishimariyaku (reverse check)")
    gc_rev  = grangercausalitytests(var_df[["tori","kansa"]].dropna(),
                                    maxlag=best_lag, verbose=False)
    for lag in range(1, best_lag+1):
        f_test = gc_rev[lag][0]["ssr_ftest"]
        f_stat, p_val = f_test[0], f_test[1]
        st = "***" if p_val<.01 else("**" if p_val<.05 else("*" if p_val<.10 else ""))
        print(f"      lag={lag}  F={f_stat:.3f}  p={p_val:.3f} {st}")

    granger_results[period_name] = gc_res


# ================================================================
# STEP 3  インパルス応答関数（IRF）の可視化
# ================================================================
print("\n[STEP 3] Impulse Response Functions (IRF)")

fig_irf, axes_irf = plt.subplots(1, 3, figsize=(16, 5))
fig_irf.suptitle(
    "Impulse Response: Shock to torishimariyaku → kansayaku\n"
    "(Full / Pre-SC / Post-SC periods, HP-filtered cycles)",
    fontsize=11)

irf_colors = {"Full (2009-2025)": "#1f77b4",
              "Pre-SC (2009-2020)": "#2ca02c",
              "Post-SC (2020-2025)": "#d62728"}

for ax, (period_name, result) in zip(axes_irf, var_results.items()):
    try:
        irf = result.irf(periods=18)
        # tori(列1) への kansa(列0) の応答
        resp = irf.irfs[:, 0, 1]  # kansaがtoriショックに応答
        stderr = irf.stderr(orth=False)[:, 0, 1] if hasattr(irf, 'stderr') else np.zeros(len(resp))

        periods_x = np.arange(len(resp))
        ax.plot(periods_x, resp, color=irf_colors[period_name], lw=2,
                label="IRF: tori shock → kansa response")
        ax.fill_between(periods_x,
                        resp - 1.96*stderr, resp + 1.96*stderr,
                        alpha=0.2, color=irf_colors[period_name])
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"{period_name}", fontsize=10)
        ax.set_xlabel("Months after shock")
        ax.set_ylabel("Response of log(kansayaku)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f"IRF failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
        ax.set_title(period_name, fontsize=10)

plt.tight_layout()
fig_irf.savefig(OUT + "fig_analysis2_irf.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("  -> fig_analysis2_irf.png")


# ================================================================
# STEP 4  VAR 予測誤差分散分解（FEVD）
# ================================================================
print("\n[STEP 4] Forecast Error Variance Decomposition (FEVD)")

fig_fevd, axes_fevd = plt.subplots(1, 3, figsize=(16, 5))
fig_fevd.suptitle(
    "FEVD: Variance of kansayaku explained by torishimariyaku shock\n"
    "(Full / Pre-SC / Post-SC, HP-filtered)",
    fontsize=11)

for ax, (period_name, result) in zip(axes_fevd, var_results.items()):
    try:
        fevd = result.fevd(periods=18)
        # kansa（列0）の変動のうちtori（列1）が説明する割合
        decomp = fevd.decomp[0]  # kansaの分散分解
        n_vars = decomp.shape[1]
        var_names = ["kansa", "tori", "yakuin"][:n_vars]
        colors_fevd = ["#1f77b4","#d62728","#2ca02c","#ff7f0e"]
        bottom = np.zeros(18)
        for i, (vn, col) in enumerate(zip(var_names, colors_fevd)):
            share = decomp[:18, i]
            ax.bar(range(18), share, bottom=bottom, color=col,
                   alpha=0.75, label=f"Explained by {vn}", width=0.8)
            bottom += share
        ax.set_title(f"{period_name}", fontsize=10)
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_ylabel("Variance share of kansayaku")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2, axis="y")
    except Exception as e:
        ax.text(0.5, 0.5, f"FEVD failed:\n{e}", transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
        ax.set_title(period_name, fontsize=10)

plt.tight_layout()
fig_fevd.savefig(OUT + "fig_analysis2_fevd.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("  -> fig_analysis2_fevd.png")


# ================================================================
# STEP 5  分析③  登記統計とJICPAデータの統合
# ================================================================
print("\n[STEP 5] Analysis ③: Integrated Registry × JICPA Panel")
print("  — 登記変化率（市場混乱度）がJICAP費用変動を媒介するか")

# ---- 登記データを年次集計 ----
# 年度（4月〜3月）ベースに集計（JICPAの年度と合わせる）
def get_fiscal_year(date):
    """4月始まり年度を返す（2020年4月〜2021年3月 → 2020年度）"""
    if date.month >= 4:
        return date.year
    else:
        return date.year - 1

monthly["fiscal_year"] = monthly["date"].apply(get_fiscal_year)

annual_reg = monthly.groupby("fiscal_year").agg(
    kansayaku_mean   = ("kansayaku",       "mean"),
    yakuin_mean      = ("yakuin",           "mean"),
    tori_mean        = ("torishimariyaku",  "mean"),
    kansayaku_sum    = ("kansayaku",        "sum"),
    yakuin_sum       = ("yakuin",           "sum"),
    ratio_mean       = ("log_ratio",        "mean"),
    nikkei_mean      = ("nikkei",           "mean"),
    usdjpy_mean      = ("usdjpy",           "mean"),
    cpi_mean         = ("cpi",              "mean"),
    loanrate_mean    = ("loanrate",         "mean"),
).reset_index()

# 市場混乱度の代理変数：前年比変化率
annual_reg["kansa_yoy"]  = annual_reg["kansayaku_sum"].pct_change()
annual_reg["yakuin_yoy"] = annual_reg["yakuin_sum"].pct_change()
annual_reg["ratio_yoy"]  = annual_reg["ratio_mean"].diff()  # log差分 ≈ 変化率

# ログ変換
for v in ["kansayaku_mean","yakuin_mean","tori_mean","nikkei_mean","cpi_mean","usdjpy_mean"]:
    annual_reg[f"log_{v}"] = np.log(annual_reg[v].replace(0, np.nan))

print(f"\n  Annual registry panel: {annual_reg['fiscal_year'].min()} ~ {annual_reg['fiscal_year'].max()}")
print(annual_reg[["fiscal_year","kansayaku_sum","yakuin_sum","ratio_mean","kansa_yoy"]].to_string(index=False))

# ---- JICPAパネルが利用可能な場合のみ結合 ----
if HAS_JICPA:
    jicpa_annual = jicpa.rename(columns={"year":"fiscal_year"})
    integrated = pd.merge(jicpa_annual, annual_reg, on="fiscal_year", how="inner")
    integrated["post2020"] = (integrated["fiscal_year"] >= 2020).astype(int)
    integrated["did"]      = integrated["treat"] * integrated["post2020"]
    integrated["post_kam"] = (integrated["fiscal_year"] >= 2015).astype(int)
    integrated["did_kam"]  = integrated["treat"] * integrated["post_kam"]

    # 標準化
    for v in ["kansa_yoy","ratio_yoy","log_kansayaku_mean"]:
        mu = integrated[v].mean()
        sd = integrated[v].std()
        integrated[f"{v}_std"] = (integrated[v] - mu) / (sd if sd>0 else 1)

    print(f"\n  Integrated panel: N={len(integrated)}")
    print(f"  Years: {sorted(integrated['fiscal_year'].unique())}")

    # ---- Model A: 基本DID（ベースライン） ----
    print("\n  [Model A] Baseline DID (avgFee ~ treat×post2020)")
    mod_A = smf.ols("avgFee ~ treat + post2020 + did", data=integrated).fit(cov_type="HC3")
    c=mod_A.params["did"]; p=mod_A.pvalues["did"]
    st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
    print(f"    DID coef = {c:+,.1f} 千円  p={p:.3f} {st}  R²={mod_A.rsquared:.3f}")

    # ---- Model B: 登記変化率をコントロールとして追加 ----
    print("\n  [Model B] DID + Registry Turnover Rate (kansa_yoy_std)")
    mod_B = smf.ols("avgFee ~ treat + post2020 + did + kansa_yoy_std",
                    data=integrated.dropna(subset=["kansa_yoy_std"])).fit(cov_type="HC3")
    for var, nm in [("did","DID"), ("kansa_yoy_std","Kansa_YoY (std)")]:
        c=mod_B.params[var]; p=mod_B.pvalues[var]
        st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
        print(f"    {nm:25s}  coef={c:+,.2f}  p={p:.3f} {st}")
    print(f"    R²={mod_B.rsquared:.3f}")

    # ---- Model C: 比率変化（log_ratio_yoy）をコントロール ----
    print("\n  [Model C] DID + Registry Ratio Change (ratio_yoy_std)")
    mod_C = smf.ols("avgFee ~ treat + post2020 + did + ratio_yoy_std",
                    data=integrated.dropna(subset=["ratio_yoy_std"])).fit(cov_type="HC3")
    for var, nm in [("did","DID"), ("ratio_yoy_std","Ratio_YoY (std)")]:
        c=mod_C.params[var]; p=mod_C.pvalues[var]
        st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
        print(f"    {nm:25s}  coef={c:+,.2f}  p={p:.3f} {st}")
    print(f"    R²={mod_C.rsquared:.3f}")

    # ---- Model D: hourlyRate も同様 ----
    print("\n  [Model D] DID on hourlyRate + Registry Turnover")
    mod_D = smf.ols("hourlyRate ~ treat + post2020 + did + kansa_yoy_std",
                    data=integrated.dropna(subset=["kansa_yoy_std"])).fit(cov_type="HC3")
    for var, nm in [("did","DID"), ("kansa_yoy_std","Kansa_YoY (std)")]:
        c=mod_D.params[var]; p=mod_D.pvalues[var]
        st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
        print(f"    {nm:25s}  coef={c:+,.2f}  p={p:.3f} {st}")
    print(f"    R²={mod_D.rsquared:.3f}")

    # ---- Model E: 2段階推計（Mediation Analysis）----
    # Stage 1: 登記変化率 ~ post2020 + treat + ...
    # Stage 2: avgFee ~ DID + 登記変化率（予測値）
    print("\n  [Model E] Two-stage Mediation: Registry change mediates fee change?")
    stage1 = smf.ols("kansa_yoy ~ post2020 + treat + did",
                     data=integrated.dropna(subset=["kansa_yoy"])).fit(cov_type="HC3")
    integrated["kansa_yoy_hat"] = stage1.predict(integrated)

    stage2 = smf.ols("avgFee ~ treat + post2020 + did + kansa_yoy_hat",
                     data=integrated.dropna(subset=["kansa_yoy_hat"])).fit(cov_type="HC3")

    print(f"  Stage 1: kansa_yoy ~ post2020 + treat + DID")
    c1=stage1.params.get("did",np.nan); p1=stage1.pvalues.get("did",np.nan)
    st1="***" if p1<.01 else("**" if p1<.05 else("*" if p1<.10 else ""))
    print(f"    DID (on kansa_yoy) = {c1:+.4f}  p={p1:.3f} {st1}")

    print(f"  Stage 2: avgFee ~ treat + post2020 + DID + kansa_yoy_hat")
    for var, nm in [("did","DID (direct)"), ("kansa_yoy_hat","Kansa_YoY (mediated)")]:
        c=stage2.params.get(var,np.nan); p=stage2.pvalues.get(var,np.nan)
        if not np.isnan(c):
            st="***" if p<.01 else("**" if p<.05 else("*" if p<.10 else ""))
            print(f"    {nm:30s}  coef={c:+,.2f}  p={p:.3f} {st}")

    # ---- 統合パネル可視化 ----
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle(
        "Analysis ③: Registry Turnover × JICPA Fee Panel\n"
        "Three-stage institutional evolution: KAM(2015) → SC(2020) → ISQM(2022)",
        fontsize=11)

    VLINES_Y = [
        (2015, "gray", ":", "KAM (2015)"),
        (2020, "red",  "--","SC Decision (2020)"),
        (2022, "navy", "-.","ISQM reform (2022)"),
    ]

    # Panel A: 年次登記変化率
    ax = axes5[0, 0]
    ax.bar(annual_reg["fiscal_year"], annual_reg["kansa_yoy"]*100,
           color=["#d62728" if y >= 2020 else "#aec7e8"
                  for y in annual_reg["fiscal_year"]],
           alpha=0.75, label="Kansayaku YoY (%)")
    ax.plot(annual_reg["fiscal_year"], annual_reg["kansa_yoy"]*100,
            "o-", color="#d62728", lw=1.5, ms=5)
    for yr, col, ls, lbl in VLINES_Y:
        ax.axvline(yr-0.5, color=col, ls=ls, lw=1.5, alpha=0.85, label=lbl)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Registry: Kansayaku Year-on-Year Change (%)", fontsize=10)
    ax.set_ylabel("YoY change (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel B: avgFeeの推移（3グループ）
    ax = axes5[0, 1]
    clr = {"FIEA_Consol":"#1f77b4","FIEA_Indiv":"#ff7f0e","CompanyAct":"#2ca02c"}
    for cat in ["FIEA_Consol","FIEA_Indiv","CompanyAct"]:
        sub = jicpa[jicpa["category"]==cat].sort_values("year")
        ax.plot(sub["year"], sub["avgFee"], "o-", color=clr[cat],
                lw=2, ms=5, label=cat)
    for yr, col, ls, lbl in VLINES_Y:
        ax.axvline(yr-0.5, color=col, ls=ls, lw=1.5, alpha=0.85)
    ax.set_title("JICPA: Avg Audit Fee by Category (thousand JPY)", fontsize=10)
    ax.set_ylabel("Avg Fee (thousand JPY)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel C: 散布図（登記変化率 vs avgFee変化）
    ax = axes5[1, 0]
    consol = integrated[integrated["category"]=="FIEA_Consol"].dropna(subset=["kansa_yoy","avgFee"])
    consol_sorted = consol.sort_values("fiscal_year")
    fee_yoy = consol_sorted["avgFee"].pct_change() * 100
    kansa_pct = consol_sorted["kansa_yoy"] * 100
    years_c = consol_sorted["fiscal_year"].values

    sc = ax.scatter(kansa_pct.values, fee_yoy.values,
                    c=years_c, cmap="RdYlBu_r", s=80, zorder=5)
    plt.colorbar(sc, ax=ax, label="Fiscal Year")
    # 回帰直線
    valid = pd.DataFrame({"x": kansa_pct.values, "y": fee_yoy.values}).dropna()
    if len(valid) > 3:
        slope, intercept, r, p_val, se = stats.linregress(valid["x"], valid["y"])
        xr = np.linspace(valid["x"].min(), valid["x"].max(), 50)
        ax.plot(xr, intercept + slope*xr, "r--", lw=1.5,
                label=f"OLS slope={slope:.2f} (r={r:.2f}, p={p_val:.3f})")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_title("Registry YoY (%) vs FIEA_Consol Fee YoY (%)", fontsize=10)
    ax.set_xlabel("Kansayaku YoY (%)")
    ax.set_ylabel("FIEA_Consol avgFee YoY (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Panel D: 2段階メカニズム図
    ax = axes5[1, 1]
    # DID係数の比較（モデルA vs B vs C）
    model_names = ["Model A\n(Basic DID)", "Model B\n(+kansa_yoy)", "Model C\n(+ratio_yoy)"]
    did_coefs = [mod_A.params.get("did",np.nan),
                 mod_B.params.get("did",np.nan),
                 mod_C.params.get("did",np.nan)]
    did_ses   = [mod_A.bse.get("did",np.nan),
                 mod_B.bse.get("did",np.nan),
                 mod_C.bse.get("did",np.nan)]
    colors_bar = ["#1f77b4","#ff7f0e","#2ca02c"]

    for i, (nm, c_val, se_val, col) in enumerate(
            zip(model_names, did_coefs, did_ses, colors_bar)):
        if not np.isnan(c_val):
            ax.bar(i, c_val, yerr=1.96*se_val, color=col, alpha=0.75,
                   capsize=6, error_kw={"linewidth":2})
            ax.text(i, c_val + (1.96*se_val + 200) * np.sign(c_val),
                    f"{c_val:+,.0f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(3)); ax.set_xticklabels(model_names, fontsize=9)
    ax.set_title("DID Coefficient Comparison\n(avgFee, FIEA_Consol vs CompanyAct)", fontsize=10)
    ax.set_ylabel("DID coef (thousand JPY)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:+,.0f}"))
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    fig5.savefig(OUT + "fig_analysis3_integrated.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close()
    print("\n  -> fig_analysis3_integrated.png")

else:
    print("\n  JICPA data not available — skipping integrated DID")
    print("  Please upload 監査実施状況2013_2023.csv to /content/")


# ================================================================
# STEP 6  制度的共進化の視覚化（論文の核心図）
# ================================================================
print("\n[STEP 6] Three-Stage Institutional Co-evolution — Summary Figure")

fig6, axes6 = plt.subplots(3, 1, figsize=(14, 13))
fig6.suptitle(
    "Three-Stage Institutional Co-evolution in Japan's Audit Market\n"
    "Stage 1: KAM intro (2015)  →  Stage 2: SC Decision (2020)"
    "  →  Stage 3: ISQM reform (2022)",
    fontsize=12, y=1.002)

VLINES_DT = [
    (KAM_DATE,  "gray", ":",  "Stage 1: KAM (2015-04)"),
    (SC_DATE,   "red",  "--", "Stage 2: SC Decision (2020-11)"),
    (ISQM_DATE, "navy", "-.", "Stage 3: ISQM (2022-04)"),
]

# Panel A: 登記系列（月次）— 小規模事務所の反応
ax = axes6[0]
ax.fill_between(monthly["date"], monthly["log_ratio"],
                monthly["log_ratio"].rolling(12,center=True).mean().min(),
                where=monthly["log_ratio"] > monthly["log_ratio"].rolling(12,center=True).mean().min(),
                alpha=0.1, color="#d62728")
ax.plot(monthly["date"],
        monthly["log_ratio"].rolling(12, center=True).mean(),
        color="#d62728", lw=2.5, label="log(Kansayaku/Yakuin) — 12m MA")
ax.plot(monthly["date"], monthly["log_ratio"],
        color="#d62728", lw=0.6, alpha=0.4)
for dt, col, ls, lbl in VLINES_DT:
    ax.axvline(dt, color=col, ls=ls, lw=1.8, alpha=0.9, label=lbl)
ax.set_title("Layer 1 (Small firms): Registry-based Auditor Turnover Ratio\n"
             "↑ = relatively more auditor changes vs director changes", fontsize=10)
ax.set_ylabel("log(Kansayaku/Yakuin)")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Panel B: 大手法人の反応（JICPAあり → 監査報酬・時間単価 / なし → 登記水準の代替表示）
ax = axes6[1]
if HAS_JICPA:
    ax2b = ax.twinx()
    consol = jicpa[jicpa["category"]=="FIEA_Consol"].sort_values("year")
    ctrl   = jicpa[jicpa["category"]=="CompanyAct"].sort_values("year")
    ax.plot(consol["year"], consol["avgFee"], "o-",
            color="#1f77b4", lw=2.5, ms=7, label="FIEA Consol: avgFee (left)")
    ax.plot(ctrl["year"],   ctrl["avgFee"],   "s-",
            color="#2ca02c", lw=2, ms=6, alpha=0.75, label="CompanyAct: avgFee (left)")
    ax2b.plot(consol["year"], consol["hourlyRate"], "^--",
              color="#ff7f0e", lw=2, ms=6, alpha=0.85, label="FIEA Consol: hourlyRate (right)")
    ax.set_ylabel("Avg Fee (thousand JPY)", color="#1f77b4")
    ax2b.set_ylabel("Hourly Rate (JPY)", color="#ff7f0e")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax2b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    h1,l1 = ax.get_legend_handles_labels()
    h2,l2 = ax2b.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
    ax.set_title("Layer 2 (Large firms): JICPA Audit Fee & Hourly Rate\n"
                 "Fee ↑ + Hourly Rate suppressed = 'time accumulation / unit cost suppression' strategy",
                 fontsize=10)
else:
    # JICPAなし: kansayaku・yakuin・torishimariyakuの月次水準（12ヶ月移動平均）を表示
    ax2b = ax.twinx()
    ma12_k = monthly.set_index("date")["kansayaku"].rolling(12, center=True).mean()
    ma12_t = monthly.set_index("date")["torishimariyaku"].rolling(12, center=True).mean()
    ma12_y = monthly.set_index("date")["yakuin"].rolling(12, center=True).mean()
    ax.fill_between(ma12_k.index, ma12_k, alpha=0.15, color="#d62728")
    ax.plot(ma12_k.index, ma12_k, color="#d62728", lw=2,
            label="Kansayaku (auditor-related) — left")
    ax.plot(ma12_t.index, ma12_t, color="#ff7f0e", lw=1.8, ls="--",
            label="Torishimariyaku (board) — left")
    ax2b.plot(ma12_y.index, ma12_y, color="#1f77b4", lw=1.8, ls=":",
              alpha=0.8, label="Yakuin (directors) — right")
    ax.set_ylabel("Monthly cases (12m MA)", color="#d62728")
    ax2b.set_ylabel("Yakuin monthly cases (12m MA)", color="#1f77b4")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax2b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    h1,l1 = ax.get_legend_handles_labels()
    h2,l2 = ax2b.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.set_title("Layer 2 (proxy): Registry-based Series Breakdown\n"
                 "[Upload 監査実施状況2013_2023.csv for JICPA audit fee / hourly rate data]",
                 fontsize=10)
for yr, col, ls in [(2015,"gray",":"),(2020,"red","--"),(2022,"navy","-."),]:
    ax.axvline(pd.Timestamp(f"{yr}-04-01") if HAS_JICPA is False
               else yr-0.5,
               color=col, ls=ls, lw=1.8, alpha=0.9)
ax.grid(alpha=0.25)

# Panel C: 年次登記変化率（市場混乱度の指標）
ax = axes6[2]
yoy = annual_reg.dropna(subset=["kansa_yoy"])
bars = ax.bar(yoy["fiscal_year"], yoy["kansa_yoy"]*100,
              color=["#d62728" if r > 0 else "#1f77b4"
                     for r in yoy["kansa_yoy"]],
              alpha=0.7, label="Kansayaku YoY (%)")
ax.plot(yoy["fiscal_year"], yoy["kansa_yoy"]*100, "o-",
        color="#333333", lw=1.5, ms=5, zorder=5)
for yr, col, ls in [(2015,"gray",":"),(2020,"red","--"),(2022,"navy","-."),]:
    ax.axvline(yr-0.5, color=col, ls=ls, lw=1.8, alpha=0.9)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("Bridge: Annual Registry Turnover Rate (%)\n"
             "Connects small-firm exit (Layer 1) to market conditions affecting large firms (Layer 2)",
             fontsize=10)
ax.set_ylabel("YoY change (%)"); ax.set_xlabel("Fiscal Year")
ax.grid(alpha=0.25, axis="y")
# 主要イベントをテキストで注記
for yr, label in [(2015,"KAM"),(2020,"SC Dec."),(2022,"ISQM")]:
    if yr in yoy["fiscal_year"].values:
        val = yoy.loc[yoy["fiscal_year"]==yr, "kansa_yoy"].values[0]*100
        ax.annotate(label, xy=(yr, val),
                    xytext=(yr+0.2, val + (5 if val >= 0 else -8)),
                    fontsize=8, color="red" if yr==2020 else "gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

plt.tight_layout()
fig6.savefig(OUT + "fig_main_coevolution.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close()
print("  -> fig_main_coevolution.png  [Key figure for paper reframing]")


# ================================================================
# STEP 7  結果サマリー出力
# ================================================================
print("\n[STEP 7] Summary")

lines = [
    "="*65,
    "  ANALYSIS ② & ③ — RESULTS SUMMARY",
    "  Two-layer Institutional Co-evolution Mechanism",
    "="*65, "",
    "[Analysis ②] VAR / Granger Causality",
]

for period_name in var_results:
    res = granger_results.get(period_name, {})
    if not res: continue
    min_p = min(res[lag][0]["ssr_ftest"][1] for lag in res)
    st = "***" if min_p<.01 else("**" if min_p<.05 else("*" if min_p<.10 else "n.s."))
    lines.append(f"  {period_name:30s}  tori→kansa min-p={min_p:.3f} {st}")

lines += [
    "",
    "  Interpretation:",
    "  If tori→kansa significant (esp. pre-SC):",
    "    -> Board restructuring preceded auditor rotation",
    "    -> Mechanism: governance change triggers auditor re-appointment",
    "  If effect strengthens post-SC:",
    "    -> SC decision accelerated the mechanism",
    "",
]

if HAS_JICPA:
    c_A = mod_A.params.get("did",np.nan); p_A = mod_A.pvalues.get("did",np.nan)
    c_B = mod_B.params.get("did",np.nan); p_B = mod_B.pvalues.get("did",np.nan)
    lines += [
        "[Analysis ③] Integrated Registry × JICPA DID",
        f"  Model A (Basic DID)               DID={c_A:+,.1f}  p={p_A:.3f}",
        f"  Model B (+kansa_yoy)               DID={c_B:+,.1f}  p={p_B:.3f}",
        "",
        "  Mediation interpretation:",
        "  If DID shrinks from A to B:",
        "    -> Registry turnover partially mediates the fee effect",
        "    -> Small-firm exit → market concentration → large-firm pricing power",
        "  If DID stable A to B:",
        "    -> Registry dynamics are independent channel, not confounder",
    ]

lines += [
    "",
    "[Figures produced]",
    "  fig_analysis2_irf.png         — IRF: tori shock → kansa response",
    "  fig_analysis2_fevd.png        — FEVD: variance decomposition",
    "  fig_analysis3_integrated.png  — Registry × JICPA integrated panel",
    "  fig_main_coevolution.png      — KEY FIGURE: 3-stage co-evolution",
    "",
    "[Paper Reframing Suggestion]",
    "  New title candidate:",
    "  'Judicial Legitimization in a Three-Stage Institutional Evolution:",
    "   Evidence from Japan's Audit Market (2015-2023)'",
    "",
    "  New core narrative (2 layers × 3 stages):",
    "  Layer 1 (Small firms, Registry data):",
    "    KAM → -40% kansayaku level [ITS β2=-0.52***]",
    "    SC  → +53% kansayaku rebound [ITS β4=+0.42***]",
    "    Interpretation: regulatory cost forced small firms out (KAM),",
    "    SC decision triggered further auditor rotation",
    "  Layer 2 (Large firms, JICPA data):",
    "    KAM → gradual fee/hour divergence begins",
    "    SC  → DID: +3,630千円 fee / -395円 hourlyRate [p<0.05]",
    "    Interpretation: 'time accumulation / unit cost suppression' strategy",
    "  Bridge (Analysis ③):",
    "    Small-firm exit → market concentration → large-firm pricing leverage",
    "="*65,
]

txt = "\n".join(lines)
print(txt)

with open(OUT+"analysis_2_3_summary.txt","w",encoding="utf-8") as f:
    f.write(txt+"\n")

annual_reg.to_csv(OUT+"annual_registry_panel.csv", index=False, encoding="utf-8-sig")
if HAS_JICPA:
    integrated.to_csv(OUT+"integrated_panel.csv", index=False, encoding="utf-8-sig")

print("\n  Output files saved to /content/:")
for fn in ["fig_analysis2_irf.png", "fig_analysis2_fevd.png",
           "fig_analysis3_integrated.png", "fig_main_coevolution.png",
           "analysis_2_3_summary.txt", "annual_registry_panel.csv",
           "integrated_panel.csv"]:
    print(f"   {fn}")
print("\nDone.")
