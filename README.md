# Lot-Level Data Leakage in WM-811K

Code and fixed data splits for the paper

> **Lot-Level Data Leakage in WM-811K: A Reproducible Benchmark Protocol for Wafer Map Defect Pattern Classification**

WM-811K is the most widely used public dataset for wafer map defect pattern classification, but published results on it are hard to compare because each paper splits and scores the data differently. A common choice is a random wafer-level split, which places wafers from the same lot in both the training and the test set. This repository measures how much that choice changes the reported scores, explains where the difference comes from, and provides fixed lot-grouped splits so that methods can be compared on equal terms.

## Main findings

All numbers are test macro-F1 (%) on the eight defect classes (D8).

| | LightGBM (284 features) | 4-block CNN |
|---|---|---|
| Random split minus lot-grouped split, mean over 5 split seeds | +1.0 (range +0.2 to +2.2) | +0.5 (range −0.4 to +2.6) |
| Range across the 5 lot-grouped splits | 82.0 – 83.7 | 85.6 – 88.1 |
| Original Training/Test split of the dataset | 67.5 | 73.0 |

- Under a random split, 11.8% of test wafers have their nearest training wafer in their own lot. Those neighbours share the label 95.8% of the time, against 81.9% for neighbours from other lots.
- The gain from a random split is consistent for the feature-based model but small and unstable for the CNN.
- The choice of lot-grouped split alone moves the score by up to 2.5 points. Results should therefore be reported as the mean over several fixed lot-grouped splits, not on a single split.
- The original Training/Test split is lot-disjoint but has a strong class shift between its two parts, so its scores are not comparable with those of re-split data.

## Repository layout

```
code/
  common.py         class names, data loading, one-hot input, hand-crafted features
  prep.py           reads LSWMD.pkl, keeps labeled wafers, resizes maps to 64x64
  split.py          lot-grouped and random splits (seed 0) for the 8- and 9-class tasks
  split_extra.py    extra splits for seeds 1-4 and the original Training/Test split
  train.py          trains one baseline (lgbm, cnn, resnet) on one split
  nn_leak.py        nearest-neighbour analysis of the leak
  oof.py            out-of-fold predictions and a confident-learning label screen
  figs_data.py      dataset figures
  make_numbers.py   collects all results into the tables, figures and numbers of the paper
splits/             fixed split files used in the paper
label_issues_8cls.csv   wafers flagged as suspect by the label screen (not verified)
```

Each split file has one row per labeled wafer with the columns `orig_index` (row index in `LSWMD.pkl`), `lot`, `label` and `split` (`train`, `val` or `test`). The file names follow `split_{kind}_{task}.csv`:

| kind | meaning |
|---|---|
| `lot`, `lots1` … `lots4` | lot-grouped, stratified, 70/10/20, split seeds 0–4 |
| `random`, `randoms1` … `randoms4` | wafer-level random, stratified, 70/10/20, split seeds 0–4 |
| `orig` | the dataset's own Training/Test labels, with 10% of training lots held out for validation |

Tasks: `8cls` uses the eight defect classes (25,519 wafers); `9cls` adds the `none` class (172,950 wafers).

## Getting the data

Download `LSWMD.pkl` (about 2.1 GB) from the WM-811K page on Kaggle (search for "WM-811K wafer map") and place it in `data/`. The file was written with Python 2 and pandas 0.x; `prep.py` handles the conversion, so no manual step is needed.

## Setup

Python 3.10 or later. A GPU is not required; every result in the paper was produced on a 4-core CPU.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The paper used Python 3.13, numpy 2.5, pandas 3.1, scikit-learn 1.9, LightGBM 4.7, PyTorch 2.15, torchvision 0.29, cleanlab 2.9 and matplotlib 3.11.

## Reproducing the results

Run the commands from the `code/` directory.

```bash
cd code
mkdir -p ../out ../paper/figures

# 1. Preprocess (writes ../data/wm_labeled64.npz and two metadata files)
python prep.py ../data/LSWMD.pkl ../data/wm

# 2. Splits. The files in splits/ are the ones used in the paper;
#    with the library versions listed above, these commands regenerate them.
python split.py ../data/wm ../data
python split_extra.py ../data/wm ../data

# 3. Baselines: train.py PREFIX SPLITDIR OUTDIR MODEL SPLIT TASK SEED
python train.py ../data/wm ../data ../out lgbm lot 8 0
python train.py ../data/wm ../data ../out cnn  random 8 0
# repeat for MODEL in {lgbm, cnn, resnet} and SPLIT in {lot, random, lots1..4, randoms1..4, orig};
# TASK 9 is available for the lot and random splits; the paper uses model SEED 0, 1, 2 on lot/random

# 4. Nearest-neighbour analysis and label screen
python nn_leak.py ../data/wm ../data ../out
python oof.py ../data/wm ../out

# 5. Tables, figures and numbers
python make_numbers.py ../data/wm ../data ../out ../paper
```

To use the published splits instead of regenerating them, pass `../splits` as `SPLITDIR`.

Approximate run times on a 4-core CPU: on D8, LightGBM takes under a minute per run, the CNN about 12 minutes and ResNet-18 about 55 minutes; LightGBM on D9 takes about 4 minutes.

### Training details

- **LightGBM**: 284 features (fail density overall, in 10 radial rings, 8 angular sectors and a 3x3 grid, plus a 16x16 down-sampled map), balanced class weights, early stopping on the validation set.
- **CNN and ResNet-18**: one-hot die states (3 channels) at 64x64, AdamW, one-cycle schedule with peak learning rate 2e-3, batch size 128, square-root inverse-frequency class weights, label smoothing 0.05, random 90° rotations and horizontal flips, 15 epochs on D8 (6 on D9). The weights from the epoch with the best validation macro-F1 are kept.

## Using the splits in your own work

1. Load `LSWMD.pkl` and select the rows given by `orig_index`.
2. Train on `train`, tune on `val` and report on `test` only.
3. Report macro-F1 as the primary metric, together with per-class F1 and accuracy.
4. Report the mean and standard deviation over the five lot-grouped splits (`lot`, `lots1` … `lots4`).
5. State the task (D8 or D9), the input resolution and any preprocessing.

## About the label screen

`label_issues_8cls.csv` lists defect wafers whose given label disagrees with out-of-fold model predictions, as ranked by confident learning. These flags have not been verified by inspection. The original labels were kept for every experiment, and a flag alone does not mean the label is wrong.

## Citation

If you use the splits or code, please cite:

```bibtex
@article{tran2026lotleakage,
  author  = {Tran, Hoang Minh Tien},
  title   = {Lot-Level Data Leakage in {WM-811K}: A Reproducible Benchmark Protocol for Wafer Map Defect Pattern Classification},
  journal = {(under review)},
  year    = {2026}
}
```

Please also cite the original dataset:

> M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer map failure pattern recognition and similarity ranking for large-scale data sets," *IEEE Transactions on Semiconductor Manufacturing*, vol. 28, no. 1, pp. 1–12, 2015.

## License

Code: MIT License (see `LICENSE`). The split files contain only row indices, lot names and labels taken from WM-811K; the dataset itself is not redistributed here and remains subject to its original terms.