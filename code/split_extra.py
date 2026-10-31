"""Extra D8 splits: lot/random with split seeds 1..4, and the dataset's original Training/Test split
(10% of training lots held out for validation)."""
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from common import load

prefix, outdir = sys.argv[1], sys.argv[2]
X, y, meta = load(prefix, 8)
for s in range(1, 5):
    for kind in ("lot", "random"):
        if kind == "lot":
            folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=s).split(X, y, meta["lot"]))
        else:
            folds = list(StratifiedKFold(10, shuffle=True, random_state=s).split(X, y))
        part = np.empty(len(y), dtype=object)
        for f, (_, idx) in enumerate(folds):
            part[idx] = "test" if f < 2 else ("val" if f == 2 else "train")
        out = meta[["orig_index", "lot", "label"]].copy()
        out["split"] = part
        out.to_csv(f"{outdir}/split_{kind}s{s}_8cls.csv", index=False)

part = np.where(meta["tt"].values == "Test", "test", "train").astype(object)
tr = np.where(part == "train")[0]
_, va = next(GroupShuffleSplit(1, test_size=0.1, random_state=0).split(tr, groups=meta["lot"].values[tr]))
part[tr[va]] = "val"
out = meta[["orig_index", "lot", "label"]].copy()
out["split"] = part
out.to_csv(f"{outdir}/split_orig_8cls.csv", index=False)
print(out.split.value_counts().to_dict())
