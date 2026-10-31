"""Make fixed splits: lot-grouped (no lot shared across train/val/test) and wafer-random, 70/10/20, stratified."""
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from common import load

prefix, outdir = sys.argv[1], sys.argv[2]
for scen in (8, 9):
    X, y, meta = load(prefix, scen)
    for kind in ("lot", "random"):
        if kind == "lot":
            folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(X, y, meta["lot"]))
        else:
            folds = list(StratifiedKFold(10, shuffle=True, random_state=0).split(X, y))
        part = np.empty(len(y), dtype=object)
        for f, (_, idx) in enumerate(folds):
            part[idx] = "test" if f < 2 else ("val" if f == 2 else "train")
        out = meta[["orig_index", "lot", "label"]].copy()
        out["split"] = part
        out.to_csv(f"{outdir}/split_{kind}_{scen}cls.csv", index=False)
        lots = {s: set(out.lot[out.split == s]) for s in ("train", "val", "test")}
        print(scen, kind, out.split.value_counts().to_dict(),
              "lots shared train/test:", len(lots["train"] & lots["test"]))
