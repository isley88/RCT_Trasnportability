"""
Repeated 5-fold cross-validation (20 repeats) of the same candidate models.
Rationale: Harrell's bootstrap optimism correction is known to under-correct highly adaptive learners
(random forest, boosting) because bootstrap samples contain duplicated patients that trees memorise.
Out-of-fold predictions avoid this, so the statistical-vs-ML comparison is judged on both methods.
Risk score for discrimination = 1 - S(t_cal | x) for every survival model (comparable across folds).
"""
import numpy as np, pandas as pd, warnings, importlib.util, sys
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
warnings.filterwarnings("ignore")
R = 20

spec = importlib.util.spec_from_file_location("mc", "model_comparison_defs.py"); mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
d, SETS, trial, SURV, CLF, OUT, surv_at, pick_alpha, coxnet, ALPHA_NET = mc.d, mc.SETS, mc.trial, mc.SURV, mc.CLF, mc.OUT, mc.surv_at, mc.pick_alpha, mc.coxnet, mc.ALPHA_NET

def slope_surv(st, y):
    st = np.clip(st, 1e-6, 1 - 1e-6); lp = np.log(-np.log(st)); lp = lp - lp.mean()
    return CoxPHSurvivalAnalysis(alpha=0).fit(lp.reshape(-1, 1), y).coef_[0]

rows = []
for sname, Z in SETS.items():
    Zt = Z[trial].reset_index(drop=True)
    for out, (e, t, tau, tcal) in OUT.items():
        y = Surv.from_arrays(d.loc[trial, e].values == 1, d.loc[trial, t].values, "e", "t")
        for l1 in (1.0, 0.5): ALPHA_NET[l1] = pick_alpha(Zt, y, l1)
        SURV["Cox lasso (L1)"] = ("stat", coxnet(1.0)); SURV["Cox elastic net"] = ("stat", coxnet(0.5))
        grid = np.arange(1, tau + 0.01, 1.0); ic = list(grid).index(tcal)
        for mname, (kind, make) in SURV.items():
            res = []
            for rep in range(R):
                S = np.full((len(y), len(grid)), np.nan)
                try:
                    for tr, te in KFold(5, shuffle=True, random_state=rep).split(Zt):
                        m = make().fit(Zt.iloc[tr], y[tr]); S[te] = surv_at(m, Zt.iloc[te], grid)
                    gmax = y["t"].max()
                    c = concordance_index_censored(y["e"], y["t"], 1 - S[:, ic])[0]
                    ibs = integrated_brier_score(y, y, S, grid)
                    res.append([c, slope_surv(S[:, ic], y), ibs])
                except Exception:
                    pass
            if res:
                r = np.nanmean(res, axis=0)
                rows.append(dict(covset=sname, outcome=out, model=mname, kind=kind, cv_disc=r[0], cv_slope=r[1], cv_score=r[2], reps=len(res)))
                print(sname, out, mname, np.round(r, 3), flush=True)
    yv = d.orr.values[trial]
    for mname, (kind, make) in CLF.items():
        res = []
        for rep in range(R):
            p = np.full(len(yv), np.nan)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(Zt, yv):
                m = make().fit(Zt.iloc[tr], yv[tr]); p[te] = m.predict_proba(Zt.iloc[te])[:, 1]
            p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
            sl = LogisticRegression(C=1e6, max_iter=5000).fit(lp.reshape(-1, 1), yv).coef_[0, 0]
            res.append([roc_auc_score(yv, p), sl, np.mean((p - yv) ** 2)])
        r = np.mean(res, axis=0)
        rows.append(dict(covset=sname, outcome="ORR", model=mname, kind=kind, cv_disc=r[0], cv_slope=r[1], cv_score=r[2], reps=R))
        print(sname, "ORR", mname, np.round(r, 3), flush=True)
pd.DataFrame(rows).round(3).to_csv("../results/model_comparison_cv.csv", index=False)
print("done")
