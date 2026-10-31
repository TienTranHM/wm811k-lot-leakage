import numpy as np
import pandas as pd

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def load(prefix, scenario):
    d = np.load(prefix + "_labeled64.npz")
    X, y = d["X"], d["y"].astype(int)
    meta = pd.read_csv(prefix + "_labeled_meta.csv")
    if scenario == 8:
        keep = y != 8
        X, y, meta = X[keep], y[keep], meta[keep].reset_index(drop=True)
    return X, y, meta


def onehot(X):
    # die states: 0 = off-wafer, 1 = pass, 2 = fail
    return np.stack([(X == k) for k in (0, 1, 2)], axis=1).astype(np.float32)


def features(X):
    """Hand-crafted features: region fail density (Wu et al. style), radial and angular profiles, 16x16 fail map."""
    n, S, _ = X.shape
    fail = (X == 2).astype(np.float32)
    on = (X > 0).astype(np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    c = (S - 1) / 2
    r = np.hypot(yy - c, xx - c) / (S / 2)
    ang = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    feats = []
    denom_all = on.sum((1, 2)) + 1e-6
    feats.append(fail.sum((1, 2)) / denom_all)
    for k in range(10):  # radial rings
        m = (r >= k / 10) & (r < (k + 1) / 10)
        feats.append((fail * m).sum((1, 2)) / ((on * m).sum((1, 2)) + 1e-6))
    for k in range(8):  # angular sectors
        m = (ang >= k / 8) & (ang < (k + 1) / 8)
        feats.append((fail * m).sum((1, 2)) / ((on * m).sum((1, 2)) + 1e-6))
    for i in range(3):  # 3x3 grid regions
        for j in range(3):
            sl = (slice(i * S // 3, (i + 1) * S // 3), slice(j * S // 3, (j + 1) * S // 3))
            feats.append(fail[:, sl[0], sl[1]].sum((1, 2)) / (on[:, sl[0], sl[1]].sum((1, 2)) + 1e-6))
    F = np.stack(feats, 1)
    small = fail.reshape(n, 16, 4, 16, 4).mean((2, 4)).reshape(n, -1)
    return np.concatenate([F, small], 1)
