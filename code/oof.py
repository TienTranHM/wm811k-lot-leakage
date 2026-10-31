"""Out-of-fold probabilities (5-fold, grouped by lot) from LightGBM and a CNN on the 8 defect classes,
then cleanlab to rank likely label errors. Writes oof probs, issue list, and review grids."""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import load, onehot, features, CLASSES

prefix, outdir = sys.argv[1], sys.argv[2]
X, y, meta = load(prefix, 8)
K = 8
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=1).split(X, y, meta["lot"]))
t0 = time.time()

import lightgbm as lgb
F = features(X)
P_lgb = np.zeros((len(y), K))
for f, (tr, te) in enumerate(folds):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, class_weight="balanced",
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=f, verbose=-1)
    clf.fit(F[tr], y[tr])
    P_lgb[te] = clf.predict_proba(F[te])
    print("lgbm fold", f, f"{time.time() - t0:.0f}s", flush=True)

import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(4)
Xt, yt = torch.from_numpy(onehot(X)), torch.from_numpy(y).long()

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

P_cnn = np.zeros((len(y), K))
for f, (tr, te) in enumerate(folds):
    torch.manual_seed(f); np.random.seed(f)
    net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, K))
    counts = np.bincount(y[tr], minlength=K)
    w = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    epochs = 10
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(tr) // 128 + 1))
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = torch.rot90(Xt[b], np.random.randint(4), (2, 3))
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, (3,))
            loss = Fn.cross_entropy(net(xb), yt[b], weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        P_cnn[te] = torch.cat([Fn.softmax(net(Xt[te[i:i + 512]]), 1) for i in range(0, len(te), 512)]).numpy()
    print("cnn fold", f, f"{time.time() - t0:.0f}s", flush=True)

np.savez(f"{outdir}/oof_8cls.npz", P_lgb=P_lgb, P_cnn=P_cnn, y=y)

from cleanlab.filter import find_label_issues
P = (P_lgb + P_cnn) / 2
issues = find_label_issues(y, P, return_indices_ranked_by="self_confidence")
agree = (P_lgb.argmax(1) == P_cnn.argmax(1)) & (P_cnn.argmax(1) != y)
rows = []
for rank, i in enumerate(issues):
    rows.append(dict(rank=rank, idx=i, orig_index=meta.orig_index[i], lot=meta.lot[i], given=CLASSES[y[i]],
                     suggested=CLASSES[P[i].argmax()], p_given=P[i, y[i]], p_suggested=P[i].max(),
                     both_models_agree=bool(agree[i])))
iss = pd.DataFrame(rows)
iss.to_csv(f"{outdir}/label_issues_8cls.csv", index=False)
print("cleanlab issues:", len(iss), "of", len(y), f"({len(iss) / len(y):.1%});",
      "with both models agreeing on another class:", int(iss.both_models_agree.sum()))
print(iss.groupby("given").size().to_string())

# review grids: up to 25 strongest candidates per given class
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cmap = matplotlib.colors.ListedColormap(["white", "#c8d6e5", "#c0392b"])
for c in range(K):
    sub = iss[(iss.given == CLASSES[c]) & iss.both_models_agree].head(25)
    if len(sub) == 0:
        continue
    fig, axs = plt.subplots(5, 5, figsize=(10, 11))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axs.ravel(), sub.iterrows()):
        ax.imshow(X[r.idx], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"#{r.orig_index}\n-> {r.suggested} ({r.p_suggested:.2f})", fontsize=8)
    fig.suptitle(f"Given label: {CLASSES[c]} (model suggests otherwise)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/review_{CLASSES[c]}.png", dpi=90)
    plt.close(fig)
print("done", f"{time.time() - t0:.0f}s")
