"""
LASER-201 (lazertinib, single-arm trial; data=1) -> SMC (SOC chemotherapy; data=2)  — v3: brain_met2 (prior to index) + delta sensitivity
Transportability analysis of PFS / OS / ORR  —  v2
  * pre-specified 10 covariates (see variable-selection rationale)
  * outcome-model internal validation: optimism-corrected C, calibration slope, IBS
  * estimators: IOSW (weighted KM), g-formula (Cox ridge / RSF / GBS), doubly robust (pseudo-observations / AIPW)
  * targets: A = SMC T790M+ (primary), B = all SMC (sensitivity)
Colab:  !pip install scikit-survival
"""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)
PATH, B_IV, B_TR = "../data/data_fin_r1.csv", 100, 200
DELTAS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]   # hazard multiplier: real-world vs trial conditions (lazertinib)

# ============================================================ 1. data
d = pd.read_csv(PATH)
d["orr"] = np.where(d.orr_type.isin([1, 2]), 1.0, 0.0)
d.loc[d.orr_type == 99, "orr"] = np.nan
X = pd.DataFrame({
    "age": d.age, "male": (d.sex == 1).astype(int), "ever_smk": d.smoking.isin([1, 2]).astype(int),
    "ecog1": (d.ecog_status >= 1).astype(int), "brain_met": d.brain_met2, "liver_met": d.liver_met,
    "del19": d.del19, "tki_ge6m": (d.g_duration_tki == 2).astype(int), "hb": d.hb, "egfr": d.renal})
CONT = ["age", "hb", "egfr"]
trial = (d.data == 1).values
for c in X: X[c] = X[c].fillna(X.loc[trial, c].median())
dt, dh = d[trial].reset_index(drop=True), d[~trial].reset_index(drop=True)
Xt_raw, Xh_raw = X[trial].reset_index(drop=True), X[~trial].reset_index(drop=True)

def std(A, Bm):
    sc = StandardScaler().fit(A[CONT]); A, Bm = A.copy(), Bm.copy()
    A[CONT] = sc.transform(A[CONT]); Bm[CONT] = sc.transform(Bm[CONT]); return A, Bm

OUT = {"PFS": dict(e="pfs_yn", t="pfs_tt", tau=24, tp=[6, 12], tcal=12),
       "OS":  dict(e="os_yn",  t="os_tt",  tau=36, tp=[12, 24], tcal=24)}
MODELS = {
    "Cox ridge": lambda: CoxPHSurvivalAnalysis(alpha=5.0),
    "RSF":       lambda: RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, max_features="sqrt", random_state=1),
    "GBS":       lambda: GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1)}

def surv_at(m, Xa, times):
    """matrix n x len(times) of predicted S(t|x), step function, flat beyond last event time"""
    sf = m.predict_survival_function(Xa, return_array=True)
    ut = m.unique_times_ if hasattr(m, "unique_times_") else m.event_times_
    idx = np.searchsorted(ut, times, side="right") - 1
    out = np.where(idx[None, :] < 0, 1.0, sf[:, np.clip(idx, 0, None)])
    return out

# ============================================================ 2. internal validation (trial only)
def cal_slope(S_t, y, tcal):
    """slope of Cox model on cloglog(1-S(tcal)) — 1 = ideal, <1 = predictions too extreme"""
    lp = np.log(-np.log(np.clip(S_t, 1e-6, 1 - 1e-6)))
    lp = (lp - lp.mean())
    m = CoxPHSurvivalAnalysis(alpha=0).fit(lp.reshape(-1, 1), y)
    return m.coef_[0]

def metrics(m, Xa, y, o):
    grid = np.arange(1, o["tau"] + 0.01, 1.0)
    c = concordance_index_censored(y["e"], y["t"], m.predict(Xa))[0]
    S = surv_at(m, Xa, grid)
    ibs = integrated_brier_score(y, y, S, grid)
    sl = cal_slope(S[:, list(grid).index(o["tcal"])], y, o["tcal"])
    return np.array([c, sl, ibs])

iv_rows = []
Xt_s, _ = std(Xt_raw, Xh_raw)
for out, o in OUT.items():
    y = Surv.from_arrays(dt[o["e"]] == 1, dt[o["t"]], "e", "t")
    for name, make in MODELS.items():
        m = make().fit(Xt_s, y); app = metrics(m, Xt_s, y, o); opt = []
        for _ in range(B_IV):
            i = rng.integers(0, len(y), len(y))
            yb = y[i]
            if yb["t"][yb["e"]].max() < o["tau"] or yb["t"].max() <= o["tau"]:
                pass
            try:
                mb = make().fit(Xt_s.iloc[i], yb)
                opt.append(metrics(mb, Xt_s.iloc[i], yb, o) - metrics(mb, Xt_s, y, o))
            except Exception:
                continue
        corr = app - np.nanmean(opt, axis=0)
        # calibration slope: bootstrap-model-in-original estimate (Harrell): 1 - optimism
        iv_rows.append(dict(outcome=out, model=name, C_apparent=app[0], C_corrected=corr[0],
                            slope_corrected=corr[1],
                            IBS_apparent=app[2], IBS_corrected=corr[2], n_boot=len(opt)))
        print(out, name, np.round(app, 3), "->", np.round(corr, 3), flush=True)

# ORR — ridge logistic: corrected AUC, calibration slope, Brier
yo = dt.orr.values
def lr(): return LogisticRegression(C=0.5, max_iter=2000)
def orr_metrics(m, Xa, yv):
    p = np.clip(m.predict_proba(Xa)[:, 1], 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    sl = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), yv).coef_[0, 0]
    return np.array([roc_auc_score(yv, p), sl, np.mean((p - yv) ** 2)])
m = lr().fit(Xt_s, yo); app = orr_metrics(m, Xt_s, yo); opt = []
for _ in range(B_IV):
    i = rng.integers(0, len(yo), len(yo))
    if len(np.unique(yo[i])) < 2: continue
    mb = lr().fit(Xt_s.iloc[i], yo[i]); opt.append(orr_metrics(mb, Xt_s.iloc[i], yo[i]) - orr_metrics(mb, Xt_s, yo))
corr = app - np.mean(opt, axis=0)
iv_rows.append(dict(outcome="ORR", model="Ridge logistic", C_apparent=app[0], C_corrected=corr[0],
                    slope_corrected=corr[1], IBS_apparent=app[2], IBS_corrected=corr[2], n_boot=len(opt)))
print("ORR", np.round(app, 3), "->", np.round(corr, 3), flush=True)
# marginal (null) reference for IBS / Brier
for out, o in OUT.items():
    y = Surv.from_arrays(dt[o["e"]] == 1, dt[o["t"]], "e", "t"); grid = np.arange(1, o["tau"] + 0.01, 1.0)
    from sksurv.nonparametric import kaplan_meier_estimator
    tk, sk = kaplan_meier_estimator(y["e"], y["t"])
    s0 = np.array([sk[tk <= g][-1] if (tk <= g).any() else 1 for g in grid])
    iv_rows.append(dict(outcome=out, model="Null (KM)", IBS_apparent=integrated_brier_score(y, y, np.tile(s0, (len(y), 1)), grid)))
iv_rows.append(dict(outcome="ORR", model="Null (prevalence)", IBS_apparent=np.mean((yo.mean() - yo) ** 2)))
pd.DataFrame(iv_rows).round(3).to_csv("../results/v3_internal_validation.csv", index=False)

# ============================================================ 3. transport estimators
def km_curve(t, e, w, grid):
    ut = np.unique(t[e == 1]); S, ts, ss = 1.0, [0.0], [1.0]
    for u in ut:
        r = w[t >= u].sum(); dd = w[(t == u) & (e == 1)].sum()
        if r > 0: S *= 1 - dd / r
        ts.append(u); ss.append(S)
    ts, ss = np.array(ts), np.array(ss)
    return ss[np.searchsorted(ts, grid, side="right") - 1]

def rmst(grid, s): return np.sum(s[:-1] * np.diff(grid))

def km_stats(t, e, o, w=None):
    w = np.ones(len(t)) if w is None else w
    g = np.arange(0, o["tau"] + 1e-9, 0.1); s = km_curve(t, e, w, g)
    return np.r_[[s[np.searchsorted(g, tp)] for tp in o["tp"]], rmst(g, s)]

def pseudo(t, e, o):
    n = len(t); full = km_stats(t, e, o); po = np.zeros((n, len(full)))
    for i in range(n):
        mask = np.arange(n) != i
        po[i] = n * full - (n - 1) * km_stats(t[mask], e[mask], o)
    return po

STAT = lambda o: [f"S({tp})" for tp in o["tp"]] + [f"RMST({o['tau']})"]

def transport_once(dt_, dh_, Xt_r, Xh_r, models=("Cox ridge", "RSF", "GBS"), curves=False):
    Xt_, Xh_ = std(Xt_r, Xh_r); res, cur = {}, {}
    Xall = pd.concat([Xt_, Xh_]); s = np.r_[np.ones(len(Xt_)), np.zeros(len(Xh_))]
    pm = LogisticRegression(C=1.0, max_iter=2000).fit(Xall, s)
    p = np.clip(pm.predict_proba(Xt_)[:, 1], 0.01, 0.99); w = (1 - p) / p
    res[("Diag", "ESS_trial", "")] = w.sum() ** 2 / (w ** 2).sum()
    res[("Diag", "max_w_share", "")] = w.max() / w.sum()
    for out, o in OUT.items():
        e_t, t_t = dt_[o["e"]].values, dt_[o["t"]].values
        e_h, t_h = dh_[o["e"]].values, dh_[o["t"]].values
        names = STAT(o)
        obs_h = km_stats(t_h, e_h, o); obs_t = km_stats(t_t, e_t, o); iosw = km_stats(t_t, e_t, o, w)
        for k, v in zip(names, obs_h): res[(out, "SMC observed", k)] = v
        for k, v in zip(names, obs_t): res[(out, "Trial observed", k)] = v
        for k, v in zip(names, iosw): res[(out, "IOSW", k)] = v
        g = np.arange(0, o["tau"] + 1e-9, 0.1)
        y = Surv.from_arrays(e_t == 1, t_t)
        for mn in models:
            m = MODELS[mn]().fit(Xt_, y)
            Sh = surv_at(m, Xh_, g); curve = Sh.mean(0)
            vals = np.r_[[curve[np.searchsorted(g, tp)] for tp in o["tp"]], rmst(g, curve)]
            for k, v in zip(names, vals): res[(out, f"g-formula ({mn})", k)] = v
            if mn == "Cox ridge":
                # doubly robust via pseudo-observations
                St = surv_at(m, Xt_, g)
                mt = np.c_[[St[:, np.searchsorted(g, tp)] for tp in o["tp"]]].T
                mt = np.c_[mt, np.array([rmst(g, r) for r in St])]
                po = pseudo(t_t, e_t, o)
                dr = vals + (w[:, None] * (po - mt)).sum(0) / w.sum()
                for k, v in zip(names, dr): res[(out, "Doubly robust", k)] = v
                if curves: cur[(out, "g-formula (Cox ridge)")] = (g, curve)
                # delta sensitivity: S_rw(t) = S_trial-conditions(t) ** delta  (proportional hazard multiplier)
                for dl in DELTAS:
                    cd = curve ** dl
                    v2 = np.r_[[cd[np.searchsorted(g, tp)] for tp in o["tp"]], rmst(g, cd)]
                    for k, v in zip(names, v2): res[(out, f"delta={dl}", k)] = v
        if curves:
            cur[(out, "SMC observed")] = (g, km_curve(t_h, e_h, np.ones(len(t_h)), g))
            cur[(out, "Trial observed")] = (g, km_curve(t_t, e_t, np.ones(len(t_t)), g))
            cur[(out, "IOSW")] = (g, km_curve(t_t, e_t, w, g))
    # ORR
    yo_ = dt_.orr.values; ok = ~np.isnan(dh_.orr.values)
    lrm = LogisticRegression(C=0.5, max_iter=2000).fit(Xt_, yo_)
    mh, mt = lrm.predict_proba(Xh_)[:, 1], lrm.predict_proba(Xt_)[:, 1]
    res[("ORR", "SMC observed", "ORR")] = dh_.orr.values[ok].mean()
    res[("ORR", "Trial observed", "ORR")] = yo_.mean()
    res[("ORR", "IOSW", "ORR")] = np.average(yo_, weights=w)
    res[("ORR", "g-formula (logistic)", "ORR")] = mh.mean()
    res[("ORR", "Doubly robust", "ORR")] = mh.mean() + (w * (yo_ - mt)).sum() / w.sum()
    for dl in DELTAS:                      # odds of response divided by delta
        od = mh / (1 - mh) / dl
        res[("ORR", f"delta={dl}", "ORR")] = (od / (1 + od)).mean()
    return (pd.Series(res), cur) if curves else pd.Series(res)

def run(mask, label):
    dh_, Xh_ = dh[mask].reset_index(drop=True), Xh_raw[mask].reset_index(drop=True)
    est, cur = transport_once(dt, dh_, Xt_raw, Xh_, curves=True)
    # participation-model c-statistic
    A, Bm = std(Xt_raw, Xh_); Xall = pd.concat([A, Bm]); s = np.r_[np.ones(len(A)), np.zeros(len(Bm))]
    pc = roc_auc_score(s, LogisticRegression(C=1.0, max_iter=2000).fit(Xall, s).predict_proba(Xall)[:, 1])
    bt = []
    for b in range(B_TR):
        i = rng.integers(0, len(dt), len(dt)); j = rng.integers(0, len(dh_), len(dh_))
        try:
            bt.append(transport_once(dt.iloc[i].reset_index(drop=True), dh_.iloc[j].reset_index(drop=True),
                                     Xt_raw.iloc[i].reset_index(drop=True), Xh_.iloc[j].reset_index(drop=True)))
        except Exception as ex:
            continue
        if b % 50 == 0: print(label, "boot", b, flush=True)
    bt = pd.concat(bt, axis=1)
    rows = []
    for key, v in est.items():
        out, est_name, stat = key
        r = dict(target=label, outcome=out, estimator=est_name, measure=stat, value=v,
                 lo=bt.loc[key].quantile(.025), hi=bt.loc[key].quantile(.975))
        if out in ("PFS", "OS", "ORR") and est_name not in ("SMC observed",):
            ref = (out, "SMC observed", stat)
            dv = bt.loc[key] - bt.loc[ref]
            r.update(diff=v - est[ref], diff_lo=dv.quantile(.025), diff_hi=dv.quantile(.975))
        rows.append(r)
    rows.append(dict(target=label, outcome="Diag", estimator="participation c-statistic", value=pc))
    # tipping-point delta (point estimate): benefit on RMST / ORR reaches zero
    for out, o in OUT.items():
        g, curve = cur[(out, "g-formula (Cox ridge)")]
        smc = est[(out, "SMC observed", f"RMST({o['tau']})")]
        fine = np.arange(1, 30.001, 0.01)
        bvals = np.array([rmst(g, curve ** dl) - smc for dl in fine])
        tp_ = fine[np.argmax(bvals <= 0)] if (bvals <= 0).any() else np.nan
        rows.append(dict(target=label, outcome=out, estimator="tipping delta (RMST benefit = 0)", value=tp_))
    A_, B_ = std(Xt_raw, Xh_)
    mh = LogisticRegression(C=0.5, max_iter=2000).fit(A_, dt.orr.values).predict_proba(B_)[:, 1]
    smc = est[("ORR", "SMC observed", "ORR")]
    fine = np.arange(1, 100.001, 0.01)
    bvals = np.array([((mh / (1 - mh) / dl) / (1 + mh / (1 - mh) / dl)).mean() - smc for dl in fine])
    rows.append(dict(target=label, outcome="ORR", estimator="tipping delta (ORR benefit = 0)", value=fine[np.argmax(bvals <= 0)] if (bvals <= 0).any() else np.nan))
    curv = []
    for (out, nm), (g, s) in cur.items():
        for gg, ss in zip(g, s):
            if abs(gg - round(gg)) < 1e-9: curv.append(dict(target=label, outcome=out, curve=nm, month=int(round(gg)), surv=ss))
    return pd.DataFrame(rows), pd.DataFrame(curv)

rA, cA = run(dh.t790m.values == 1, "A: SMC T790M+")
rB, cB = run(np.ones(len(dh), bool), "B: SMC all")
res = pd.concat([rA, rB]); cur = pd.concat([cA, cB])
res.round(4).to_csv("../results/v3_transport_results.csv", index=False); cur.round(4).to_csv("../results/v3_curves.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 300)
print(res.round(3).to_string())
