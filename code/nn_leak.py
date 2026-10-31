"""Why random splits leak: 1-NN analysis on hand-crafted features (D8).
For each split kind, find each test wafer's nearest training wafer and report how often it comes from the
same lot, how often it has the same label, and 1-NN macro-F1."""
import sys, json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from common import load, features

prefix, splitdir, outdir = sys.argv[1:4]
X, y, meta = load(prefix, 8)
F = features(X)
lots = meta["lot"].values
res = []
for kind in ["lot", "random"] + [f"{k}s{s}" for s in range(1, 5) for k in ("lot", "random")]:
    sp = pd.read_csv(f"{splitdir}/split_{kind}_8cls.csv")["split"].values
    tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
    sc = StandardScaler().fit(F[tr])
    nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(F[tr]))
    dist, idx = nn.kneighbors(sc.transform(F[te]))
    j = tr[idx[:, 0]]
    same_lot = lots[j] == lots[te]
    r = dict(kind=kind, same_lot=float(same_lot.mean()), nn_macro_f1=float(f1_score(y[te], y[j], average="macro")),
             median_dist=float(np.median(dist)),
             label_agree_same_lot=float((y[j] == y[te])[same_lot].mean()) if same_lot.any() else None,
             label_agree_other_lot=float((y[j] == y[te])[~same_lot].mean()))
    res.append(r)
    print(r)
json.dump(res, open(f"{outdir}/nn_leak.json", "w"), indent=1)
# within-lot label purity for defect wafers
d = meta.assign(y=y)
g = d.groupby("lot").y
multi = g.size() > 1
print("defect lots with >1 defect wafer:", int(multi.sum()), "share whose wafers all have one label:",
      float((g.nunique()[multi] == 1).mean()))
