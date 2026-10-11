"""
Outcome-model comparison (statistical vs machine learning) in LASER-201, and the transported
estimate each model would give in SMC T790M+ (target A).
- covariate sets: 'pre10' (10 pre-specified, v3) and 'full' (all 26 raw variables, one-hot; 1st-round setting)
- performance: apparent vs Harrell bootstrap optimism-corrected C/AUC, calibration slope, IBS/Brier
- transported: g-formula point estimate in target A for each model
"""
import numpy as np, pandas as pd, warnings, time
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026); B = 100

d = pd.read_csv("../data/data_fin_r1.csv")
d["orr"] = np.where(d.orr_type.isin([1, 2]), 1.0, 0.0); d.loc[d.orr_type == 99, "orr"] = np.nan
trial = (d.data == 1).values; tA = ((d.data == 2) & (d.t790m == 1)).values

pre = pd.DataFrame({"age": d.age, "male": (d.sex == 1).astype(int), "ever_smk": d.smoking.isin([1, 2]).astype(int),
    "ecog1": (d.ecog_status >= 1).astype(int), "brain_met": d.brain_met2, "liver_met": d.liver_met, "del19": d.del19,
    "tki_ge6m": (d.g_duration_tki == 2).astype(int), "hb": d.hb, "egfr": d.renal})
excl = ["study_id", "pfs_yn", "pfs_tt", "os_yn", "os_tt", "orr_type", "orr", "data"]
catv = ["sex", "smoking", "ecog_status", "tnm_stage", "histology_type", "g_time_from_tki", "g_duration_tki"]
full = d.drop(columns=excl).copy()
full = pd.get_dummies(full, columns=catv, drop_first=True, dtype=float)
SETS = {}
for name, X in [("pre10", pre), ("full", full)]:
    X = X.copy()
    for c in X: X[c] = X[c].fillna(X.loc[trial, c].median())
    X = X.loc[:, X[trial].std() > 0]                      # drop columns constant in the trial
    cont = [c for c in X if X[c].nunique() > 2]            # standardize continuous only (same as main analysis std())
    Z = X.copy().reset_index(drop=True); sc = StandardScaler().fit(X.loc[trial, cont]); Z[cont] = sc.transform(X[cont])
    SETS[name] = Z
print({k: v.shape[1] for k, v in SETS.items()})

# --------------------------------------------------------------- survival models
def coxnet(l1):
    return lambda: CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=[ALPHA_NET[l1]], fit_baseline_model=True)
SURV = {
    "Cox (unpenalized)": ("stat", lambda: CoxPHSurvivalAnalysis(alpha=1e-4)),
    "Cox ridge (L2)": ("stat", lambda: CoxPHSurvivalAnalysis(alpha=5.0)),
    "Cox lasso (L1)": ("stat", None), "Cox elastic net": ("stat", None),
    "Random survival forest": ("ml", lambda: RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, max_features="sqrt", random_state=1)),
    "Gradient boosting survival": ("ml", lambda: GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1)),
}

def surv_at(m, X, grid):
    sf = m.predict_survival_function(X, return_array=True)
    ut = getattr(m, "unique_times_", None)
    if ut is None: ut = m.event_times_
    idx = np.searchsorted(ut, grid, side="right") - 1
    return np.where(idx[None, :] < 0, 1.0, sf[:, np.clip(idx, 0, None)])

def s_metrics(m, X, y, tau, tcal):
    grid = np.arange(1, tau + 0.01, 1.0)
    c = concordance_index_censored(y["e"], y["t"], m.predict(X))[0]
    S = surv_at(m, X, grid)
    ibs = integrated_brier_score(y, y, S, grid)
    st = np.clip(S[:, list(grid).index(tcal)], 1e-6, 1 - 1e-6); lp = np.log(-np.log(st)); lp = lp - lp.mean()
    sl = CoxPHSurvivalAnalysis(alpha=0).fit(lp.reshape(-1, 1), y).coef_[0]
    return np.array([c, sl, ibs])

def rmst(grid, s): return np.sum(s[:-1] * np.diff(grid))

# choose a single penalty for coxnet by 5-fold CV deviance-free proxy: largest C-index on CV (done once, documented)
from sklearn.model_selection import KFold
ALPHA_NET = {}
def pick_alpha(Z, y, l1):
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

OUT = {"PFS": ("pfs_yn", "pfs_tt", 24, 12), "OS": ("os_yn", "os_tt", 36, 24)}
rows, trows = [], []
for sname, Z in SETS.items():
    Zt, ZA = Z[trial].reset_index(drop=True), Z[tA].reset_index(drop=True)
    for out, (e, t, tau, tcal) in OUT.items():
        y = Surv.from_arrays(d.loc[trial, e].values == 1, d.loc[trial, t].values, "e", "t")
        for l1 in (1.0, 0.5): ALPHA_NET[l1] = pick_alpha(Zt, y, l1)
        SURV["Cox lasso (L1)"] = ("stat", coxnet(1.0)); SURV["Cox elastic net"] = ("stat", coxnet(0.5))
        smc_A = d.loc[tA]; tk, sk = kaplan_meier_estimator(smc_A[e] == 1, smc_A[t])
        g = np.arange(0, tau + 1e-9, 0.1); s_obs = np.array([sk[tk <= x][-1] if (tk <= x).any() else 1 for x in g])
        for mname, (kind, make) in SURV.items():
            t0 = time.time()
            try:
                m = make().fit(Zt, y); app = s_metrics(m, Zt, y, tau, tcal)
            except Exception as ex:
                print("fail", sname, out, mname, ex); continue
            opt = []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y))
                try:
                    mb = make().fit(Zt.iloc[i], y[i])
                    opt.append(s_metrics(mb, Zt.iloc[i], y[i], tau, tcal) - s_metrics(mb, Zt, y, tau, tcal))
                except Exception: pass
            corr = app - np.nanmean(opt, axis=0)
            # transported (g-formula) in target A
            curve = surv_at(m, ZA, g).mean(0)
            tr_rmst = rmst(g, curve)
            rows.append(dict(covset=sname, outcome=out, model=mname, kind=kind, n_feat=Zt.shape[1],
                             disc_app=app[0], disc_corr=corr[0], optimism=app[0] - corr[0],
                             slope_corr=corr[1], score_app=app[2], score_corr=corr[2], n_boot=len(opt)))
            trows.append(dict(covset=sname, outcome=out, model=mname, transported=tr_rmst, smc_obs=rmst(g, s_obs),
                              benefit=tr_rmst - rmst(g, s_obs)))
            print(sname, out, mname, np.round(app, 3), np.round(corr, 3), f"{time.time()-t0:.0f}s", flush=True)
        # null model score
        tk0, sk0 = kaplan_meier_estimator(y["e"], y["t"]); grid = np.arange(1, tau + 0.01, 1.0)
        s0 = np.array([sk0[tk0 <= x][-1] if (tk0 <= x).any() else 1 for x in grid])
        rows.append(dict(covset=sname, outcome=out, model="Null (no covariates)", kind="ref", score_app=integrated_brier_score(y, y, np.tile(s0, (len(y), 1)), grid)))

# --------------------------------------------------------------- ORR classifiers
yo_all = d.orr.values
CLF = {
    "Logistic (unpenalized)": ("stat", lambda: LogisticRegression(C=1e6, max_iter=5000)),
    "Ridge logistic (L2)": ("stat", lambda: LogisticRegression(C=0.5, max_iter=5000)),
    "Lasso logistic (L1)": ("stat", lambda: LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000)),
    "Elastic net logistic": ("stat", lambda: LogisticRegressionCV(Cs=10, cv=5, penalty="elasticnet", solver="saga", l1_ratios=[0.5], scoring="roc_auc", max_iter=5000)),
    "Random forest": ("ml", lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=1)),
    "Gradient boosting": ("ml", lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=1)),
    "Support vector machine": ("ml", lambda: SVC(probability=True, C=1.0, random_state=1)),
}
def c_metrics(m, X, yv):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    sl = LogisticRegression(C=1e6, max_iter=5000).fit(lp.reshape(-1, 1), yv).coef_[0, 0]
    return np.array([roc_auc_score(yv, p), sl, np.mean((p - yv) ** 2)])
for sname, Z in SETS.items():
    Zt, ZA = Z[trial].reset_index(drop=True), Z[tA].reset_index(drop=True)
    yv = yo_all[trial]; obsA = np.nanmean(yo_all[tA])
    for mname, (kind, make) in CLF.items():
        t0 = time.time()
        m = make().fit(Zt, yv); app = c_metrics(m, Zt, yv); opt = []
        for _ in range(B):
            i = rng.integers(0, len(yv), len(yv))
            if len(np.unique(yv[i])) < 2: continue
            try:
                mb = make().fit(Zt.iloc[i], yv[i]); opt.append(c_metrics(mb, Zt.iloc[i], yv[i]) - c_metrics(mb, Zt, yv))
            except Exception: pass
        corr = app - np.nanmean(opt, axis=0)
        trp = m.predict_proba(ZA)[:, 1].mean()
        rows.append(dict(covset=sname, outcome="ORR", model=mname, kind=kind, n_feat=Zt.shape[1], disc_app=app[0], disc_corr=corr[0],
                         optimism=app[0] - corr[0], slope_corr=corr[1], score_app=app[2], score_corr=corr[2], n_boot=len(opt)))
        trows.append(dict(covset=sname, outcome="ORR", model=mname, transported=trp * 100, smc_obs=obsA * 100, benefit=(trp - obsA) * 100))
        print(sname, "ORR", mname, np.round(app, 3), np.round(corr, 3), f"{time.time()-t0:.0f}s", flush=True)
    rows.append(dict(covset=sname, outcome="ORR", model="Null (prevalence)", kind="ref", score_app=np.mean((yv.mean() - yv) ** 2)))

pd.DataFrame(rows).round(3).to_csv("../results/model_comparison_performance.csv", index=False)
pd.DataFrame(trows).round(2).to_csv("../results/model_comparison_transported.csv", index=False)
print("done")
