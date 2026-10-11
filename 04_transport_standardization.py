"""
v5: g-formula only, compared across mutually exclusive outcome-model algorithms.
  Pipeline: (1) fit outcome model in LASER-201 only -> (2) choose algorithm by internal validation
            (model_comparison.py / cv_comparison.py) -> (3) predict for SMC patients -> (4) expected benefit.
  For every algorithm: transported outcome in SMC (g-formula) and benefit vs SMC observed,
  with bootstrap 95% CIs (trial and target resampled separately; all models refit in each replicate).
  IOSW (no outcome model) is reported only as a supplementary check.
Colab: !pip install scikit-survival
"""
import numpy as np, pandas as pd, warnings, os
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026); B = int(os.environ.get("B_TR", 200))

d = pd.read_csv("../data/data_fin_r1.csv")
d["orr"] = np.where(d.orr_type.isin([1, 2]), 1.0, 0.0); d.loc[d.orr_type == 99, "orr"] = np.nan
X = pd.DataFrame({"age": d.age, "male": (d.sex == 1).astype(int), "ever_smk": d.smoking.isin([1, 2]).astype(int),
    "ecog1": (d.ecog_status >= 1).astype(int), "brain_met": d.brain_met2, "liver_met": d.liver_met, "del19": d.del19,
    "tki_ge6m": (d.g_duration_tki == 2).astype(int), "hb": d.hb, "egfr": d.renal})
CONT = ["age", "hb", "egfr"]
trial = (d.data == 1).values
for c in X: X[c] = X[c].fillna(X.loc[trial, c].median())
dt, dh = d[trial].reset_index(drop=True), d[~trial].reset_index(drop=True)
Xt, Xh = X[trial].reset_index(drop=True), X[~trial].reset_index(drop=True)

def std(A, Bm):
    sc = StandardScaler().fit(A[CONT]); A, Bm = A.copy(), Bm.copy()
    A[CONT] = sc.transform(A[CONT]); Bm[CONT] = sc.transform(Bm[CONT]); return A, Bm

# Coxnet penalties selected once by 5-fold CV in model_comparison.py (pre-specified covariates), fixed here
ALPHA = {("PFS", 1.0): None, ("PFS", 0.5): None, ("OS", 1.0): None, ("OS", 0.5): None}
def _pick(Z, y, l1):
    from sklearn.model_selection import KFold
    from sksurv.metrics import concordance_index_censored
    path = CoxnetSurvivalAnalysis(l1_ratio=l1, n_alphas=30, alpha_min_ratio=0.01).fit(Z, y).alphas_
    best, ba = -1, path[0]
    for a in path:
        cs = []
        for tr, te in KFold(5, shuffle=True, random_state=1).split(Z):
            try:
                m = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=[a]).fit(Z.iloc[tr], y[tr])
                cs.append(concordance_index_censored(y[te]["e"], y[te]["t"], m.predict(Z.iloc[te]))[0])
            except Exception: pass
        if cs and np.mean(cs) > best: best, ba = np.mean(cs), a
    return ba

OUT = {"PFS": dict(e="pfs_yn", t="pfs_tt", tau=24, tp=[6, 12]), "OS": dict(e="os_yn", t="os_tt", tau=36, tp=[12, 24])}
Zt0, _ = std(Xt, Xh)
for out, o in OUT.items():
    y = Surv.from_arrays(dt[o["e"]] == 1, dt[o["t"]], "e", "t")
    for l1 in (1.0, 0.5): ALPHA[(out, l1)] = _pick(Zt0, y, l1)
print("coxnet alphas", ALPHA)

def SURV(out):
    return {
        "Cox (unpenalized)": lambda: CoxPHSurvivalAnalysis(alpha=1e-4),
        "Cox ridge (L2)": lambda: CoxPHSurvivalAnalysis(alpha=5.0),
        "Cox lasso (L1)": lambda: CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[ALPHA[(out, 1.0)]], fit_baseline_model=True),
        "Cox elastic net": lambda: CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[ALPHA[(out, 0.5)]], fit_baseline_model=True),
        "Random survival forest": lambda: RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, max_features="sqrt", random_state=1),
        "Gradient boosting survival": lambda: GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1)}
CLF = {
    "Logistic (unpenalized)": lambda: LogisticRegression(C=1e6, max_iter=5000),
    "Ridge logistic (L2)": lambda: LogisticRegression(C=0.5, max_iter=5000),
    "Lasso logistic (L1)": lambda: LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000),
    "Elastic net logistic": lambda: LogisticRegressionCV(Cs=10, cv=5, penalty="elasticnet", solver="saga", l1_ratios=[0.5], scoring="roc_auc", max_iter=5000),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=1),
    "Gradient boosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1),
    "Support vector machine": lambda: SVC(probability=True, C=1.0, random_state=1)}

def km_curve(t, e, w, grid):
    ut = np.unique(t[e == 1]); S, ts, ss = 1.0, [0.0], [1.0]
    for u in ut:
        r = w[t >= u].sum(); dd = w[(t == u) & (e == 1)].sum()
        if r > 0: S *= 1 - dd / r
        ts.append(u); ss.append(S)
    ts, ss = np.array(ts), np.array(ss)
    return ss[np.searchsorted(ts, grid, side="right") - 1]
def rmst(g, s): return np.sum(s[:-1] * np.diff(g))
def surv_at(m, Z, g):
    sf = m.predict_survival_function(Z, return_array=True)
    ut = getattr(m, "unique_times_", None)
    if ut is None: ut = m.event_times_
    idx = np.searchsorted(ut, g, side="right") - 1
    return np.where(idx[None, :] < 0, 1.0, sf[:, np.clip(idx, 0, None)])
def summ(g, curve, o): return np.r_[[curve[np.searchsorted(g, tp)] for tp in o["tp"]], rmst(g, curve)]
NAMES = lambda o: [f"S({tp})" for tp in o["tp"]] + [f"RMST({o['tau']})"]

def once(dt_, dh_, Xt_, Xh_, curves=False):
    Zt, Zh = std(Xt_, Xh_); res, cur = {}, {}
    # supplementary IOSW (participation model; covariates only)
    s = np.r_[np.ones(len(Zt)), np.zeros(len(Zh))]
    p = np.clip(LogisticRegression(C=1.0, max_iter=2000).fit(pd.concat([Zt, Zh]), s).predict_proba(Zt)[:, 1], .01, .99); w = (1 - p) / p
    for out, o in OUT.items():
        g = np.arange(0, o["tau"] + 1e-9, 0.1)
        et, tt = dt_[o["e"]].values, dt_[o["t"]].values; eh, th = dh_[o["e"]].values, dh_[o["t"]].values
        for lab, cv in [("SMC observed", km_curve(th, eh, np.ones(len(th)), g)), ("LASER-201 observed", km_curve(tt, et, np.ones(len(tt)), g)),
                        ("IOSW (supplementary)", km_curve(tt, et, w, g))]:
            for k, v in zip(NAMES(o), summ(g, cv, o)): res[(out, lab, k)] = v
            if curves: cur[(out, lab)] = (g, cv)
        y = Surv.from_arrays(et == 1, tt)
        for mn, make in SURV(out).items():
            try:
                curve = surv_at(make().fit(Zt, y), Zh, g).mean(0)
            except Exception:
                continue
            for k, v in zip(NAMES(o), summ(g, curve, o)): res[(out, mn, k)] = v
            if curves: cur[(out, mn)] = (g, curve)
    yo = dt_.orr.values; ok = ~np.isnan(dh_.orr.values)
    res[("ORR", "SMC observed", "ORR")] = dh_.orr.values[ok].mean()
    res[("ORR", "LASER-201 observed", "ORR")] = yo.mean()
    res[("ORR", "IOSW (supplementary)", "ORR")] = np.average(yo, weights=w)
    for mn, make in CLF.items():
        try: res[("ORR", mn, "ORR")] = make().fit(Zt, yo).predict_proba(Zh)[:, 1].mean()
        except Exception: pass
    return (pd.Series(res), cur) if curves else pd.Series(res)

def run(mask, label):
    dh_, Xh_ = dh[mask].reset_index(drop=True), Xh[mask].reset_index(drop=True)
    est, cur = once(dt, dh_, Xt, Xh_, curves=True); bt = []
    for b in range(B):
        i = rng.integers(0, len(dt), len(dt)); j = rng.integers(0, len(dh_), len(dh_))
        bt.append(once(dt.iloc[i].reset_index(drop=True), dh_.iloc[j].reset_index(drop=True), Xt.iloc[i].reset_index(drop=True), Xh_.iloc[j].reset_index(drop=True)))
        if b % 50 == 0: print(label, b, flush=True)
    bt = pd.concat(bt, axis=1); rows = []
    for key, v in est.items():
        out, mn, meas = key; sc = 100 if out == "ORR" or meas.startswith("S(") else 1
        r = dict(target=label, outcome=out, model=mn, measure=meas, value=v * sc, lo=bt.loc[key].quantile(.025) * sc, hi=bt.loc[key].quantile(.975) * sc)
        if mn != "SMC observed":
            dv = (bt.loc[key] - bt.loc[(out, "SMC observed", meas)]) * sc
            r.update(diff=(v - est[(out, "SMC observed", meas)]) * sc, diff_lo=dv.quantile(.025), diff_hi=dv.quantile(.975), n_boot=int(bt.loc[key].notna().sum()))
        rows.append(r)
    curv = [dict(target=label, outcome=o, curve=k, month=int(round(x)), surv=s) for (o, k), (g, c) in cur.items() for x, s in zip(g, c) if abs(x - round(x)) < 1e-9]
    return pd.DataFrame(rows), pd.DataFrame(curv)

rA, cA = run(dh.t790m.values == 1, "A: SMC T790M+")
rB, cB = run(np.ones(len(dh), bool), "B: SMC all")
pd.concat([rA, rB]).round(3).to_csv("../results/v5_gformula_by_model.csv", index=False)
pd.concat([cA, cB]).round(4).to_csv("../results/v5_curves.csv", index=False)
print("done")
