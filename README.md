# Titanic Survival Prediction — An Iterative Model-Building Study

A documented study of **five successive modelling approaches** on the classic
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). The
focus of this repository is not the final score but the **methodology**: each
version is a deliberate experiment testing one hypothesis about what should
improve out-of-sample generalisation, measured the same honest way every time.

**Author:** Avdhesh Thakur ([Avi-techAI](https://github.com/Avi-techAI))
**Task:** Binary classification — predict passenger survival
**Evaluation metric:** Accuracy (fraction of the 418 test passengers correctly classified)

---

## Results at a glance

| Version | Approach | CV accuracy (5-fold) | Public leaderboard | Outcome |
|--------:|----------|:--------------------:|:------------------:|---------|
| v1 | Random Forest, basic features | 0.8227 | 0.77272 | Baseline |
| v2 | Gradient Boosting + extra features (`FarePerPerson`, `Deck`) | 0.8317 | 0.76076 | CV up, LB **down** (overfitting) |
| v3 | Tuned Random Forest, trimmed clean features | 0.8305 | **0.77990** | **Best leaderboard** |
| v4 | v3 + survival-grouped titles | 0.8271 | 0.77751 | No gain |
| v5 | Soft-voting ensemble of 5 models | **0.8372** | 0.77990 | **Best CV**, tied best LB |

**Central finding:** a higher cross-validation score did **not** reliably
translate into a higher leaderboard score. The simplest clean model (v3) and a
five-model ensemble (v5) both plateaued at **0.77990**, which is close to the
practical ceiling for honestly-built models on a dataset this small and noisy.

All of the code, with inline commentary for each version, lives in:

➡️ **[`notebooks/titanic_model_study.ipynb`](notebooks/titanic_model_study.ipynb)**

---

## What is in the notebook

The notebook is organised as one shared data-loading section followed by five
self-contained version sections. Each version section contains:

- a short **hypothesis** statement (what I expected the change to do),
- the **feature-engineering** and **model** code for that version,
- the **5-fold cross-validation accuracy** printed inline,
- a note on the **leaderboard outcome** and the interpretation.

| Section | What it covers |
|---------|----------------|
| `0. Setup` | Imports and a portable `DATA_DIR` loader (works on Kaggle or locally) |
| `v1` | Baseline Random Forest |
| `v2` | Gradient Boosting with two extra features |
| `v3` | Trimmed features + shallow tuned Random Forest (best LB) |
| `v4` | Survival-grouped title mapping |
| `v5` | Five-model soft-voting ensemble (best CV) |
| `Submission` | Fits the chosen model and writes `submission.csv` |
| `Closing notes` | The lessons drawn from the five experiments |

---

## Methodology

### Validation protocol

Every version is scored with **5-fold cross-validation** on the 891-row training
set via `sklearn.model_selection.cross_val_score(model, X, y, cv=5)`. This gives
an out-of-sample estimate *before* consuming a leaderboard submission. The public
leaderboard is computed on only a subset of the 418-row test set, so it has
limited power to separate models whose CV scores differ by less than ~0.01.

### Shared feature engineering

Train and test are concatenated so that every transformation (imputation,
encoding) is applied identically to both, then split back apart. Core features
used across versions:

- **`Title`** — extracted from `Name` with a regex (`,\s*([^.]+)\.`). In v1–v3,
  titles appearing fewer than 10 times are grouped as `Rare`; in v4/v5 they are
  mapped into survival-linked groups (`Master`, `Mr`, `Mrs`, `Miss`, `Officer`,
  `Royalty`).
- **`FamilySize`** = `SibSp + Parch + 1`, and **`IsAlone`** = `FamilySize == 1`.
- **`Age`** — imputed with the median age *within each title group* (more accurate
  than a single global median), with a global-median fallback.
- **`Fare`** — median-imputed (one value missing in the test set).
- **`Embarked`** — mode-imputed (two values missing).
- **`Sex`** — mapped to 0/1; **`Embarked`/`Title`** one-hot encoded (`drop_first=True`).
- **`Cabin`** — dropped in the strongest versions because it is missing for the
  majority of passengers.

### Per-version rationale

**v1 — baseline.** A `RandomForestClassifier(n_estimators=300, max_depth=6,
min_samples_leaf=3)` on the core features. Establishes the reference point
(CV 0.8227, LB 0.77272).

**v2 — more power, more features.** Added `FarePerPerson = Fare / FamilySize`
and `Deck` (first letter of `Cabin`, `U` for unknown), and switched to
`GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05)`.
CV rose to 0.8317 but the **leaderboard fell to 0.76076**. The `Deck` feature is
missing for ~77% of passengers, so it mostly supplied noise the stronger model
could latch onto — a textbook case of a higher training-fit score hiding worse
generalisation.

**v3 — simplify to generalise.** Dropped `Deck` and `FarePerPerson`, kept only
features with a clear survival link, and used a shallower, more regularised
`RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=4)`.
**Best leaderboard result: 0.77990.** Confirms that on this dataset a cleaner,
simpler model beats a more powerful one.

**v4 — smarter titles.** Replaced the "rare titles together" rule with a
hand-built survival-pattern mapping (`Master`, `Officer`, `Royalty`, …). A
reasonable idea, but CV (0.8271) and LB (0.77751) both came in slightly below
v3: the extra category granularity added variance without a matching signal gain.

**v5 — ensemble.** A `VotingClassifier(voting="soft")` over five diverse models —
Random Forest, Extra Trees, Gradient Boosting, Logistic Regression (scaled), and
a probability-calibrated SVM (scaled). This produced the **highest CV of the
study (0.8372)**, but the leaderboard matched v3 at 0.77990 — strong evidence
that ~0.78 is near the honest ceiling here.

---

## Key takeaways

1. **Measure out-of-sample, not training fit.** v2's higher CV masked worse
   generalisation; honest validation is what catches this.
2. **Complexity is not free.** More powerful models and more features increase
   variance; on small data that often hurts (v2, v4).
3. **Simplicity won.** The trimmed, shallow model (v3) gave the best leaderboard
   score and tied the far more complex ensemble (v5).
4. **Leaderboards have limited resolution.** Two very different models landing on
   the identical public score is a signal you have reached the data's ceiling.

---

## Repository structure

```
titanic-ml-study/
├── README.md                          # this file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── notebooks/
│   └── titanic_model_study.ipynb      # all five approaches, documented
└── submissions/
    └── submission_v3_best.csv         # the best-scoring submission (v3)
```

> **Note on the data:** the competition's `train.csv` and `test.csv` are not
> included here, in line with Kaggle's rules. Download them from the
> [competition Data tab](https://www.kaggle.com/competitions/titanic/data) and
> place them in a `data/` folder, or run the notebook directly on Kaggle where
> they are mounted automatically.

---

## How to run

**On Kaggle (recommended):** open the notebook in a Kaggle kernel with the
Titanic dataset attached and run all cells. The loader finds the data
automatically.

**Locally:**

```bash
pip install -r requirements.txt
# place train.csv and test.csv in ./data/
jupyter notebook notebooks/titanic_model_study.ipynb
```

The v5 ensemble (which trains and cross-validates five models, including an SVM)
is the slowest cell and may take a minute or two; the other versions run in
seconds.

---

## Tools

Python · pandas · NumPy · scikit-learn · Jupyter

---

## About me

Avdhesh Thakur — MSc Computer Science student and machine learning / AI
practitioner. I build data and ML projects like this one as part of an ongoing
portfolio.

- **GitHub:** [Avi-techAI](https://github.com/Avi-techAI)
- **Kaggle:** [avdheshthakur](https://www.kaggle.com/avdheshthakur)
- **LinkedIn:** [aavdesh-thakur](https://www.linkedin.com/in/aavdesh-thakur/)
- **Email:** avdhesh.t [at] outlook [dot] com
