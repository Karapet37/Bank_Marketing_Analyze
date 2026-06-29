# Bank Marketing — Data Analysis

A simple, beginner-friendly analysis of the **UCI Bank Marketing** dataset.
The question: *which clients subscribe to a term deposit after a marketing
call, and what is related to that decision?*

The script (`bank_marketing_analysis.py`) goes top to bottom:

1. Load the data.
2. Quick check — missing values, duplicates, `"unknown"` categories.
3. Target — how many clients subscribed (the classes are imbalanced).
4. Numeric features — means per class and a correlation heatmap.
5. Categorical features — subscription rate per group.
6. A simple logistic-regression model with ROC-AUC and the top features.

## Source / dataset

Dataset: **[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)**
(`bank-full.csv`, 45,211 contacts, 16 features + target `y`).

The dataset is **not** included in this repository (see `.gitignore`).
Download it from the link above and place it so the script finds it:

```
bank-marketing-analysis/
└── bank/
    └── bank-full.csv
```

## How to run

```bash
pip install -r requirements.txt
python bank_marketing_analysis.py
```

The script can be run from any folder — it switches to its own directory first,
loads `bank/bank-full.csv`, prints the results, and saves charts to `figures/`.

## Note on `duration`

The `duration` (call length) column is dropped before training the model,
because it is only known *after* the call finishes. Using it would let the model
"cheat", so it is excluded to keep the result realistic.

## Figures

| | |
|---|---|
| ![target](figures/01_target.png) | ![correlation](figures/02_correlation.png) |
| ![poutcome](figures/03_poutcome.png) | |
