"""Train one baseline on one split; write metrics json and test probabilities.

usage: train.py PREFIX SPLITDIR OUTDIR MODEL(lgbm|cnn|resnet) SPLIT(lot|random) SCEN(8|9) SEED [labels.npy]
"""
import sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from common import load, onehot, features

prefix, splitdir, outdir, model, kind, scen, seed = sys.argv[1:8]
scen, seed = int(scen), int(seed)
X, y, meta = load(prefix, scen)
if len(sys.argv) > 8:  # optional corrected labels, aligned with load(prefix, scen)
    y = np.load(sys.argv[8])
sp = pd.read_csv(f"{splitdir}/split_{kind}_{scen}cls.csv")["split"].values
tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
K = 8 if scen == 8 else 9
t0 = time.time()

if model == "lgbm":
    import lightgbm as lgb
    F = features(X)
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, class_weight="balanced",
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    clf.fit(F[tr], y[tr], eval_set=[(F[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    P = clf.predict_proba(F[te])
else:
    import torch, torch.nn as nn, torch.nn.functional as Fn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(4)
    Xt = torch.from_numpy(onehot(X))
    yt = torch.from_numpy(y).long()

    if model == "cnn":
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, K))
    else:
        import torchvision
        net = torchvision.models.resnet18(num_classes=K)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        net.layer1 = nn.Sequential(nn.MaxPool2d(2), net.layer1)  # keep compute low on CPU

    counts = np.bincount(y[tr], minlength=K)
    w = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 15 if scen == 8 else 6
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(tr) // 128 + 1))

    def predict(idx):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                out.append(Fn.softmax(net(Xt[idx[i:i + 512]]), 1))
        return torch.cat(out).numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = Xt[b]
            k = np.random.randint(4)
            xb = torch.rot90(xb, k, (2, 3))
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, (3,))
            loss = Fn.cross_entropy(net(xb), yt[b], weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        f1v = f1_score(y[va], predict(va).argmax(1), average="macro")
        print(f"ep{ep} val macroF1 {f1v:.4f} {time.time() - t0:.0f}s", flush=True)
        if f1v > best:
            best, bad = f1v, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    P = predict(te)

pred = P.argmax(1)
res = dict(model=model, split=kind, scen=scen, seed=seed, relabeled=len(sys.argv) > 8,
           macro_f1=f1_score(y[te], pred, average="macro"), acc=accuracy_score(y[te], pred),
           per_class_f1=f1_score(y[te], pred, average=None, labels=list(range(K))).tolist(),
           cm=confusion_matrix(y[te], pred, labels=list(range(K))).tolist(),
           n_train=len(tr), n_test=len(te), seconds=time.time() - t0)
tag = f"{model}_{kind}_{scen}_{seed}" + ("_relab" if res["relabeled"] else "")
json.dump(res, open(f"{outdir}/res_{tag}.json", "w"))
print(tag, "macroF1 %.4f acc %.4f %.0fs" % (res["macro_f1"], res["acc"], res["seconds"]))
