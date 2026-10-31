"""Load LSWMD.pkl, keep labeled wafers, resize to 64x64, save compact npz + metadata csv."""
import sys, pickle, types
import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]

# LSWMD.pkl was written with pandas 0.x; alias old module paths
import pandas.core.indexes as _idx
sys.modules.setdefault("pandas.indexes", _idx)
for name in ["base", "range", "numeric", "multi"]:
    try:
        sys.modules.setdefault(f"pandas.indexes.{name}", __import__(f"pandas.core.indexes.{name}", fromlist=["x"]))
    except ImportError:
        pass

from pandas.compat import pickle_compat
with open(src, "rb") as fh:  # written by Python 2
    df = pickle_compat.Unpickler(fh, encoding="latin1").load()
print(df.shape, list(df.columns))

def unwrap(v):
    a = np.asarray(v)
    return str(a.ravel()[0]) if a.size else ""

df["label"] = df["failureType"].map(unwrap)
tt_col = [c for c in df.columns if c.lower().startswith("tr") and "label" in c.lower()][0]
df["tt"] = df[tt_col].map(unwrap)
df["lot"] = df["lotName"].astype(str)
df["widx"] = df["waferIndex"].astype(int)
df["h"] = df["waferMap"].map(lambda m: m.shape[0])
df["w"] = df["waferMap"].map(lambda m: m.shape[1])

print("all wafers", len(df), "lots", df.lot.nunique())
print(df.label.value_counts())
print(df.tt.value_counts())
df[["lot", "widx", "label", "tt", "h", "w"]].to_csv(dst + "_all_meta.csv", index=True)

lab = df[df.label != ""].copy()
S = 64

def resize_nn(m):
    h, w = m.shape
    r = (np.arange(S) * h / S).astype(int)
    c = (np.arange(S) * w / S).astype(int)
    return m[np.ix_(r, c)].astype(np.uint8)

X = np.stack([resize_nn(np.asarray(m)) for m in lab["waferMap"]])
classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
y = lab["label"].map({c: i for i, c in enumerate(classes)}).values
assert not np.isnan(y.astype(float)).any(), lab.label.unique()
meta = lab[["lot", "widx", "label", "tt", "h", "w"]].reset_index().rename(columns={"index": "orig_index"})
np.savez_compressed(dst + "_labeled64.npz", X=X, y=y.astype(np.int8))
meta.to_csv(dst + "_labeled_meta.csv", index=False)
print("saved", X.shape)
