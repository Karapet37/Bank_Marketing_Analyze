# -*- coding: utf-8 -*-
# Bank Marketing - simple data analysis
# Dataset: UCI Bank Marketing (bank/bank-full.csv)
# Goal: see which clients say "yes" to a term deposit.

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


# 3. Target: how many people subscribed?
rate = df["subscribed"].mean()
print(f"\nSubscription rate: {rate:.2%}")

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
print("\nNumeric means by class (0 = no, 1 = yes):")
print(df.groupby("subscribed")[numeric_cols].mean())

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

for col in categorical_cols:
    group = df.groupby(col)["subscribed"].mean().sort_values(ascending=False)
    print(f"\nSubscription rate by {col}:")
    print((group * 100).round(1))

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

# Which features push the prediction up or down the most?
coefs = pd.Series(model.coef_[0], index=X.columns)
print("\nTop positive features:")
print(coefs.sort_values(ascending=False).head(10))
print("\nTop negative features:")
print(coefs.sort_values().head(10))

print("\nDone. Figures saved in the figures/ folder.")
