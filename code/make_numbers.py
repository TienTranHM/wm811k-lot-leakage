"""Collect results into LaTeX macros and tables for the paper."""
import sys, glob, json, os
import numpy as np
import pandas as pd
from common import CLASSES

prefix, splitdir, outdir, paper = sys.argv[1:5]
allm = pd.read_csv(prefix + "_all_meta.csv")
m = pd.read_csv(prefix + "_labeled_meta.csv")
d = m[m.label != "none"]
N = {}
N["nAll"] = f"{len(allm):,}"
N["nLots"] = f"{allm.lot.nunique():,}"
N["nLabeled"] = f"{len(m):,}"
N["pctLabeled"] = f"{len(m) / len(allm):.1%}".replace("%", "\\%")
N["nLabLots"] = f"{m.lot.nunique():,}"
N["medLot"] = f"{m.groupby('lot').size().median():.0f}"
N["nNone"] = f"{(m.label == 'none').sum():,}"
N["nDefect"] = f"{len(d):,}"
N["nEdgeRing"] = f"{(m.label == 'Edge-Ring').sum():,}"
N["nNearFull"] = f"{(m.label == 'Near-full').sum():,}"
N["nSizes"] = f"{m.groupby(['h', 'w']).ngroups}"
ct = pd.crosstab(m.label, m.tt)
N["origERtrain"] = f"{ct.loc['Edge-Ring', 'Training']:,}"
N["origERtest"] = f"{ct.loc['Edge-Ring', 'Test']:,}"
r8 = pd.read_csv(f"{splitdir}/split_random_8cls.csv")
N["nSharedLots"] = f"{len(set(r8.lot[r8.split == 'train']) & set(r8.lot[r8.split == 'test'])):,}"
N["nFeat"] = "284"
gl = d.groupby("lot").label
multi = gl.size() > 1
N["nMultiLots"] = f"{int(multi.sum()):,}"
N["lotPure"] = f"{100 * (gl.nunique()[multi] == 1).mean():.1f}"
N["medDefLot"] = f"{gl.size()[multi].median():.0f}"

# results
R = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{outdir}/res_*.json")])
R = R[~R.relabeled]
g = R.groupby(["model", "scen", "split"]).agg(f1=("macro_f1", "mean"), f1sd=("macro_f1", "std"),
                                               acc=("acc", "mean"), n=("seed", "count")).reset_index()
def leak(model):
    s = g[(g.model == model) & (g.scen == 8)].set_index("split")
    return f"{100 * (s.loc['random', 'f1'] - s.loc['lot', 'f1']):.1f}" if {"lot", "random"} <= set(s.index) else "\\todo{}"
N["leakLGB"], N["leakCNN"] = leak("lgbm"), leak("cnn")

# leakage over split seeds: paired (lot sK, random sK) runs with model seed 0
pairs = [("lot", "random")] + [(f"lots{s}", f"randoms{s}") for s in range(1, 5)]
R0 = R[(R.scen == 8) & (R.seed == 0)].set_index(["model", "split"]).macro_f1
P = []
for mod in ("lgbm", "cnn", "resnet"):
    for k, (a, b) in enumerate(pairs):
        if (mod, a) in R0.index and (mod, b) in R0.index:
            P.append(dict(model=mod, split_seed=k, lot=R0[(mod, a)], random=R0[(mod, b)]))
P = pd.DataFrame(P)
P["gap"] = 100 * (P.random - P.lot)
for mod, key in (("lgbm", "LGB"), ("cnn", "CNN"), ("resnet", "RES")):
    s = P[P.model == mod]
    N[f"gap{key}n"] = f"{len(s)}"
    N[f"gap{key}mean"] = f"{s.gap.mean():.1f}" if len(s) else "\\todo{}"
    N[f"gap{key}min"] = f"{s.gap.min():.1f}" if len(s) else "\\todo{}"
    N[f"gap{key}max"] = f"{s.gap.max():.1f}" if len(s) else "\\todo{}"
    N[f"gap{key}pos"] = f"{int((s.gap > 0).sum())}"
P.to_csv(f"{outdir}/leak_pairs.csv", index=False)
for mod, key in (("lgbm", "LGB"), ("cnn", "CNN")):
    s = P[P.model == mod]
    N[f"lotLo{key}"], N[f"lotHi{key}"] = f"{100 * s.lot.min():.1f}", f"{100 * s.lot.max():.1f}"
    N[f"lotRange{key}"] = f"{100 * (s.lot.max() - s.lot.min()):.1f}"
sd = g[(g.scen == 8) & (g.n > 1)].f1sd.max()
N["seedSDmax"] = f"{100 * sd:.1f}"
d9 = g[(g.model == "lgbm") & (g.scen == 9)].set_index("split").f1
N["gapDnine"] = f"{100 * (d9['random'] - d9['lot']):.1f}" if {"lot", "random"} <= set(d9.index) else "\\todo{}"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.4, 2.1))
labels = {"lgbm": "LightGBM", "cnn": "CNN", "resnet": "ResNet-18"}
mods = [m for m in ("lgbm", "cnn", "resnet") if (P.model == m).any()]
for i, mod in enumerate(mods):
    s = P[P.model == mod]
    jit = np.linspace(-0.12, 0.12, len(s)) if len(s) > 1 else np.zeros(1)
    for (_, r), j in zip(s.iterrows(), jit):
        ax.plot([i + j - 0.18, i + j + 0.18], [100 * r.lot, 100 * r.random], color="0.6", lw=0.7)
    ax.scatter(i + jit - 0.18, 100 * s.lot, marker="o", color="black", s=12, label="lot-grouped" if i == 0 else None, zorder=3)
    ax.scatter(i + jit + 0.18, 100 * s.random, marker="^", facecolor="white", edgecolor="black", s=14, label="random" if i == 0 else None, zorder=3)
ax.set_xticks(range(len(mods))); ax.set_xticklabels([labels[m] for m in mods])
ax.set_ylabel("Test macro-F1 (%)"); ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{paper}/figures/fig_leak.pdf"); plt.close(fig)

# nearest-neighbour mechanism
nnp = f"{outdir}/nn_leak.json"
if os.path.exists(nnp):
    nn = pd.DataFrame(json.load(open(nnp)))
    rnd, lot = nn[nn.kind.str.startswith("random")], nn[nn.kind.str.startswith("lot")]
    N["nnSameLot"] = f"{100 * rnd.same_lot.mean():.1f}"
    N["nnAgreeSame"] = f"{100 * rnd.label_agree_same_lot.mean():.1f}"
    N["nnAgreeOther"] = f"{100 * rnd.label_agree_other_lot.mean():.1f}"
    N["nnFoneLot"] = f"{100 * lot.nn_macro_f1.mean():.1f}"
    N["nnFoneRand"] = f"{100 * rnd.nn_macro_f1.mean():.1f}"
    N["nnSplits"] = f"{len(rnd)}"

# original Training/Test split
for mod, key in (("lgbm", "LGB"), ("cnn", "CNN")):
    v = R0.get((mod, "orig"))
    N[f"orig{key}"] = f"{100 * v:.1f}" if v is not None else "\\todo{}"

names = {"lgbm": "LightGBM (features)", "cnn": "4-block CNN", "resnet": "ResNet-18"}
lines = ["\\begin{table}[t]\\centering\\caption{Test macro-F1 and accuracy (\\%) under lot-grouped and random splits, mean $\\pm$ s.d. over seeds ($n$).}\\label{tab:results}",
         "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llcccc}\\toprule",
         "Task & Model & Split & Macro-F1 & Acc. & $n$\\\\\\midrule"]
for scen in (8, 9):
    for mod in ("lgbm", "cnn", "resnet"):
        for sp in ("lot", "random"):
            row = g[(g.model == mod) & (g.scen == scen) & (g.split == sp)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            sd = f"$\\pm${100 * r.f1sd:.1f}" if r.n > 1 else ""
            lines.append(f"D{scen} & {names[mod]} & {sp} & {100 * r.f1:.1f}{sd} & {100 * r.acc:.1f} & {r.n}\\\\")
    lines.append("\\midrule")
lines[-1] = "\\bottomrule\\end{tabular}\\end{table}"
open(f"{paper}/tab_results.tex", "w").write("\n".join(lines) + "\n")

iss_path = f"{outdir}/label_issues_8cls.csv"
if os.path.exists(iss_path):
    iss = pd.read_csv(iss_path)
    N["nIssues"] = f"{len(iss):,}"
    N["pctIssues"] = f"{len(iss) / len(d):.1%}".replace("%", "\\%")
    N["nIssuesAgree"] = f"{int(iss.both_models_agree.sum()):,}"
    cnt = d.label.value_counts()
    by = iss.groupby("given").size().reindex(CLASSES[:8]).fillna(0).astype(int)
    rate = by / cnt.reindex(CLASSES[:8])
    N["topIssueClass"] = rate.idxmax()
    top_sugg = iss.groupby("given").suggested.agg(lambda s: s.value_counts().index[0])
    t = ["\\begin{table}[t]\\centering\\caption{Wafers flagged as suspect by confident learning (D8); flags are not verified.}\\label{tab:issues}",
         "\\begin{tabular}{lrrrl}\\toprule Given label & Wafers & Flagged & \\% & Most suggested\\\\\\midrule"]
    for c in CLASSES[:8]:
        t.append(f"{c} & {cnt[c]:,} & {by[c]:,} & {100 * rate[c]:.1f} & {top_sugg.get(c, '--')}\\\\")
    t.append(f"\\midrule Total & {len(d):,} & {len(iss):,} & {100 * len(iss) / len(d):.1f} & \\\\\\bottomrule\\end{{tabular}}\\end{{table}}")
    open(f"{paper}/tab_issues.tex", "w").write("\n".join(t) + "\n")
else:
    for k in ("nIssues", "pctIssues", "nIssuesAgree", "topIssueClass"):
        N[k] = "\\todo{}"
    open(f"{paper}/tab_issues.tex", "w").write("\\todo{label issue table}\n")

with open(f"{paper}/numbers.tex", "w") as f:
    for k, v in N.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n".replace("\\xspace", ""))
print(N)
print(g.to_string())
