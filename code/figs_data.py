import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import CLASSES
prefix, out = sys.argv[1], sys.argv[2]
m = pd.read_csv(prefix + "_labeled_meta.csv"); X = np.load(prefix + "_labeled64.npz")["X"]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
# Fig: one example per class
fig, axs = plt.subplots(1, 9, figsize=(7.0, 1.1))
cm = matplotlib.colors.ListedColormap(["white", "#bfbfbf", "black"])
for ax, c in zip(axs, CLASSES):
    i = np.where(m.label.values == c)[0][3]
    ax.imshow(X[i], cmap=cm, vmin=0, vmax=2, interpolation="nearest"); ax.set_title(c, fontsize=7); ax.axis("off")
fig.tight_layout(); fig.savefig(out + "/fig_examples.pdf"); plt.close(fig)
# Fig: class counts in original Training/Test split (log scale)
ct = pd.crosstab(m.label, m.tt).reindex(CLASSES)
fig, ax = plt.subplots(figsize=(3.4, 2.0))
xx = np.arange(len(CLASSES))
ax.bar(xx - 0.2, ct["Training"], 0.4, label="Training", color="0.3"); ax.bar(xx + 0.2, ct["Test"], 0.4, label="Test", color="0.7")
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_ylabel("Wafers"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(out + "/fig_class_counts.pdf"); plt.close(fig)
print(ct)
