# -*- coding: utf-8 -*-
# Bank Marketing - simple data analysis
# Dataset: UCI Bank Marketing (bank/bank-full.csv)
# Goal: see which clients say "yes" to a term deposit.

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, classification_report


# Work next to this script, not in whatever folder you ran python from.
# So "bank/bank-full.csv" and "figures/" are always found.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)


# 1. Load the data
# The file uses ";" as the separator.
df = pd.read_csv("bank/bank-full.csv", sep=";")

# Make a 0/1 column from the target so it is easy to work with.
df["subscribed"] = (df["y"] == "yes").astype(int)

print("Shape:", df.shape)
print(df.head())


# 2. Quick look at the data
print("\nMissing values (NaN):")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# In this dataset some categories are the string "unknown" instead of NaN.
for col in df.select_dtypes("object").columns:
    n_unknown = (df[col] == "unknown").sum()
    if n_unknown > 0:
        print(f"  {col}: {n_unknown} 'unknown' values")

# Build a data-quality table for Excel: per column, how many NaN / "unknown".
quality = pd.DataFrame({"column": df.columns,
                        "missing_nan": df.isna().sum().to_numpy()})
# Comparing a numeric column to "unknown" just gives 0, so no type check needed.
quality["unknown"] = [int((df[c] == "unknown").sum()) for c in df.columns]


# 3. Target: how many people subscribed?
rate = df["subscribed"].mean()
print(f"\nSubscription rate: {rate:.2%}")

# Small overview table for Excel (the headline numbers in one place).
overview = pd.DataFrame({
    "metric": ["rows", "columns", "duplicate_rows", "subscription_rate_%"],
    "value": [len(df), df.shape[1], int(df.duplicated().sum()), round(rate * 100, 2)],
})

counts = df["y"].value_counts()
plt.figure(figsize=(5, 4))
plt.bar(counts.index, counts.values, color=["gray", "green"])
plt.title(f"Subscription (rate = {rate:.1%})")
plt.ylabel("clients")
plt.tight_layout()
plt.savefig("figures/01_target.png")
plt.close()


# 4. Numeric features
numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# Mean of each numeric column for people who said no vs yes.
# Build a tidy table: one row per feature, columns "no" / "yes" / "diff".
num_means = df.groupby("subscribed")[numeric_cols].mean().T   # rows = features
num_means.columns = ["no", "yes"]
num_means["diff"] = (num_means["yes"] - num_means["no"]).round(2)
num_means = num_means.round(2).reset_index().rename(columns={"index": "feature"})
print("\nNumeric means by class (0 = no, 1 = yes):")
print(num_means)

# Correlation between numeric features.
# Same numbers as the heatmap below, but kept as a table for the Excel report.
corr_table = df[numeric_cols].corr().round(2).reset_index()
corr_table = corr_table.rename(columns={"index": "feature"})

# Correlation heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between numeric features")
plt.tight_layout()
plt.savefig("figures/02_correlation.png")
plt.close()


# 5. Categorical features: subscription rate per group
categorical_cols = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]

cat_tables = {}  # keep each table so we can save them to Excel later
for col in categorical_cols:
    # rate = доля подписок в группе, n = размер группы,
    # lift = во сколько раз группа лучше/хуже среднего (1.0 = как среднее).
    group = df.groupby(col)["subscribed"].agg(["mean", "size"])
    group.columns = ["rate", "n"]
    group["rate_%"] = (group["rate"] * 100).round(1)
    group["lift"] = (group["rate"] / rate).round(2)
    group = group.sort_values("rate", ascending=False)
    print(f"\nSubscription rate by {col}:")
    print(group[["rate_%", "n", "lift"]])

    # reset_index() turns the category from the row label into a real column,
    # so in Excel we get proper columns (category | rate_% | n | lift),
    # not everything squished into one column.
    table = group[["rate_%", "n", "lift"]].reset_index()
    cat_tables[col] = table

# Plot one example: rate by previous-campaign outcome.
group = df.groupby("poutcome")["subscribed"].mean().sort_values(ascending=False)
plt.figure(figsize=(6, 4))
plt.bar(group.index, group.values * 100, color="green")
plt.title("Subscription rate by poutcome")
plt.ylabel("rate, %")
plt.tight_layout()
plt.savefig("figures/03_poutcome.png")
plt.close()


# 6. Simple model: logistic regression
# We turn text columns into numbers with get_dummies (one-hot encoding).
# Note: we drop "duration" because it is only known after the call, so using
# it would be cheating (the model would already "know" the answer).
X = df.drop(columns=["y", "subscribed", "duration"])
y = df["subscribed"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\nROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, pred, target_names=["no", "yes"]))

# Same report as a table for Excel (precision / recall / f1 per class).
report_dict = classification_report(y_test, pred, target_names=["no", "yes"],
                                    output_dict=True)
report_table = pd.DataFrame(report_dict).T.round(3).reset_index()
report_table = report_table.rename(columns={"index": "metric"})

# Which features push the prediction up or down the most?
coefs = pd.Series(model.coef_[0], index=X.columns)
print("\nTop positive features:")
print(coefs.sort_values(ascending=False).head(10))
print("\nTop negative features:")
print(coefs.sort_values().head(10))

# Tidy table for Excel: feature | coef (10 strongest up + 10 strongest down).
top_pos = coefs.sort_values(ascending=False).head(10)
top_neg = coefs.sort_values().head(10)
drivers = pd.concat([top_pos, top_neg]).round(3)
drivers = drivers.reset_index()
drivers.columns = ["feature", "coef"]


# 7. Is the model actually good? Cross-validation + compare with other models
# One train/test split can get lucky. Cross-validation splits the data 5 times
# and averages the score, so we trust the number more.
# We also compare with a "dummy" model (always guesses the majority) and a
# random forest, to see if logistic regression was a reasonable choice.
print("\n=== Cross-validation (ROC-AUC, 5 folds) ===")

# IMPORTANT: the rows in the file are ordered by date (month after month).
# We must shuffle before splitting, otherwise each fold is a different time
# period and the scores come out wrong (even below 0.5). shuffle=True fixes it.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "dummy (baseline)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
}

cv_results = []  # collect for the Excel report
for name in models:
    scores = cross_val_score(models[name], X, y, cv=folds, scoring="roc_auc")
    print(f"{name:22} AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")
    cv_results.append({"model": name,
                       "AUC_mean": round(scores.mean(), 3),
                       "AUC_std": round(scores.std(), 3)})

cv_results = pd.DataFrame(cv_results)


# 8. Business view: who should we call first? (gains curve)
# If we sort clients by predicted probability and call the most promising first,
# how many of the real subscribers do we catch after calling only X% of people?
# This turns the model into something the marketing team can actually use.
order = np.argsort(proba)[::-1]              # clients sorted: most likely first
y_sorted = y_test.to_numpy()[order]
caught = np.cumsum(y_sorted) / y_sorted.sum()  # share of subscribers caught
share_called = np.arange(1, len(y_sorted) + 1) / len(y_sorted)  # share of clients called

# Example numbers: what we catch by calling the top 20% and top 50%.
for top in [0.2, 0.5]:
    idx = int(top * len(y_sorted)) - 1
    print(f"\nCalling the top {top:.0%} of clients catches {caught[idx]:.0%} of all subscribers")

# Tidy gains table for Excel: for each 10% step, how many subscribers we catch.
gains_rows = []
for top in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    idx = int(top * len(y_sorted)) - 1
    gains_rows.append({"clients_called_%": int(top * 100),
                       "subscribers_caught_%": round(caught[idx] * 100, 1)})
gains_table = pd.DataFrame(gains_rows)

plt.figure(figsize=(6, 5))
plt.plot(share_called * 100, caught * 100, color="green", label="model")
plt.plot([0, 100], [0, 100], "--", color="gray", label="random calling")
plt.xlabel("clients called, %")
plt.ylabel("subscribers caught, %")
plt.title("Gains curve — call the most promising first")
plt.legend()
plt.tight_layout()
plt.savefig("figures/04_gains_curve.png")
plt.close()


# 9. Save all results to one Excel file, each table on its own sheet (page).
# Every table has proper columns (not squished into one column).
# Needs the openpyxl library (it is in requirements.txt).
with pd.ExcelWriter("results.xlsx") as writer:
    # Headline numbers and data quality first.
    overview.to_excel(writer, sheet_name="overview", index=False)
    quality.to_excel(writer, sheet_name="data_quality", index=False)

    # Numeric features: means by class and the correlation matrix (the heatmap).
    num_means.to_excel(writer, sheet_name="numeric_means", index=False)
    corr_table.to_excel(writer, sheet_name="correlation", index=False)

    # Model results: comparison, the classification report, and top drivers.
    cv_results.to_excel(writer, sheet_name="model_comparison", index=False)
    report_table.to_excel(writer, sheet_name="model_report", index=False)
    drivers.to_excel(writer, sheet_name="top_features", index=False)

    # Business view.
    gains_table.to_excel(writer, sheet_name="gains", index=False)

    # One sheet per categorical feature (job, month, poutcome, ...).
    for col in cat_tables:
        cat_tables[col].to_excel(writer, sheet_name=f"by_{col}"[:31], index=False)

print("\nResults saved to results.xlsx (one sheet per table).")
print("Done. Figures saved in the figures/ folder.")
