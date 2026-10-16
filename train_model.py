import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score
)

print("==========================================================")
print("   STAGE 0: BUILDING MASTER DATASET & DOMAIN FEATURES     ")
print("==========================================================")

# 1. Load and combine both datasets (319 historical + 38 recent 2020-2026 records)
dfs = []
if os.path.exists("SCRAPED_RAW_IPO.csv"):
    df_scraped = pd.read_csv("SCRAPED_RAW_IPO.csv")
    dfs.append(df_scraped)

if os.path.exists("RAW IPO.csv"):
    df_recent = pd.read_csv("RAW IPO.csv")
    dfs.append(df_recent)

df_master = pd.concat(dfs, ignore_index=True)

# Deduplicate by company name (keeping the most recent entry)
df_master["Company_Clean"] = df_master["Company_Name"].astype(str).str.strip().str.lower()
df_master = df_master.drop_duplicates(subset=["Company_Clean"], keep="last").drop(columns=["Company_Clean"])
df_master.to_csv("MASTER_IPO_DATASET.csv", index=False)
print(f"[INFO] Combined Master Dataset created: {len(df_master)} unique IPOs.")

# 2. Domain Feature Engineering Function
def engineer_features(data):
    d = data.copy()
    num_cols = ["Issue_Size_Cr", "Issue_Price_INR", "Total_Sub_x", "QIB_Sub_x", "HNI_Sub_x", "Retail_Sub_x"]
    for col in num_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    # A. Log-transforms to normalize skewed subscription & size distributions
    d["Log_Issue_Size"] = np.log1p(d["Issue_Size_Cr"].clip(lower=0))
    d["Log_Total_Sub"] = np.log1p(d["Total_Sub_x"].clip(lower=0))
    d["Log_QIB_Sub"] = np.log1p(d["QIB_Sub_x"].clip(lower=0))
    d["Log_HNI_Sub"] = np.log1p(d["HNI_Sub_x"].clip(lower=0))
    d["Log_Retail_Sub"] = np.log1p(d["Retail_Sub_x"].clip(lower=0))

    # B. Institutional Conviction vs. Retail FOMO Ratios
    d["QIB_to_Retail_Ratio"] = (d["QIB_Sub_x"] + 1.0) / (d["Retail_Sub_x"] + 1.0)
    d["HNI_to_Retail_Ratio"] = (d["HNI_Sub_x"] + 1.0) / (d["Retail_Sub_x"] + 1.0)
    d["Institutional_Dominance"] = (d["QIB_Sub_x"] + d["HNI_Sub_x"]) / (d["Total_Sub_x"] + 1.0)

    # C. Liquidity & Issue Scarcity Metrics
    d["Total_Capital_Demanded_Cr"] = d["Issue_Size_Cr"] * d["Total_Sub_x"]
    d["Log_Capital_Demanded"] = np.log1p(d["Total_Capital_Demanded_Cr"].clip(lower=0))
    d["Scarcity_Index"] = d["Total_Sub_x"] / ((d["Issue_Size_Cr"] / 100.0) + 1.0)

    return d

df_feat = engineer_features(df_master)

FEATURE_COLS = [
    "Issue_Size_Cr", "Issue_Price_INR", "Total_Sub_x", "QIB_Sub_x", "HNI_Sub_x", "Retail_Sub_x",
    "Log_Issue_Size", "Log_Total_Sub", "Log_QIB_Sub", "Log_HNI_Sub", "Log_Retail_Sub",
    "QIB_to_Retail_Ratio", "HNI_to_Retail_Ratio", "Institutional_Dominance",
    "Log_Capital_Demanded", "Scarcity_Index"
]

# Separate listed IPOs (for training) from unlisted 2026 IPOs
df_train_full = df_feat.dropna(subset=["Listing_Gain_Pct"]).copy()
X = df_train_full[FEATURE_COLS]

# Targets:
# 1. Classification Target: 1 if Listing Gain > 5% (Profitable Listing), else 0
y_clf = (df_train_full["Listing_Gain_Pct"] > 5.0).astype(int)

# 2. Regression Target: Winsorize extreme outliers at [-40%, +150%] so outliers don't distort fit
y_reg = df_train_full["Listing_Gain_Pct"].clip(lower=-40.0, upper=150.0)

X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf
)

print("\n==========================================================")
print("   STAGE 1: CLASSIFICATION ENGINE (WIN PROBABILITY)       ")
print("==========================================================")

clf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=6,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)
clf_model.fit(X_train, y_clf_train)

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_scores = cross_val_score(clf_model, X, y_clf, cv=cv, scoring="roc_auc")

y_clf_pred = clf_model.predict(X_test)
y_clf_prob = clf_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_clf_test, y_clf_pred)
test_auc = roc_auc_score(y_clf_test, y_clf_prob)

print(f"5-Fold CV ROC-AUC Score : {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std():.4f})")
print(f"Holdout Test Accuracy   : {test_acc * 100:.2f}%")
print(f"Holdout Test ROC-AUC    : {test_auc:.4f}")

print("\n==========================================================")
print("   STAGE 2: HUBER & QUANTILE REGRESSION (GAIN RANGE)      ")
print("==========================================================")

# Huber Loss Regressor (robust to outliers) for Expected Gain
reg_expected = GradientBoostingRegressor(
    loss="huber", alpha=0.85, n_estimators=200, max_depth=4, learning_rate=0.04, random_state=42
)
# 25th Percentile Regressor (Conservative Floor)
reg_low = GradientBoostingRegressor(
    loss="quantile", alpha=0.25, n_estimators=150, max_depth=3, learning_rate=0.04, random_state=42
)
# 75th Percentile Regressor (Optimistic Ceiling)
reg_high = GradientBoostingRegressor(
    loss="quantile", alpha=0.75, n_estimators=150, max_depth=3, learning_rate=0.04, random_state=42
)

reg_expected.fit(X_train, y_reg_train)
reg_low.fit(X_train, y_reg_train)
reg_high.fit(X_train, y_reg_train)

y_reg_pred = reg_expected.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Outlier-Adjusted R2 Score : {r2:.4f}")
print(f"Mean Absolute Error (MAE) : {mae:.2f}%")

print("\n==========================================================")
print("   STAGE 3: EXPORTING MODELS & DASHBOARD DATASETS         ")
print("==========================================================")

# 1. Save Feature Importance Report
fi_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Classifier_Importance_Pct": (clf_model.feature_importances_ * 100).round(2),
    "Regressor_Importance_Pct": (reg_expected.feature_importances_ * 100).round(2)
}).sort_values(by="Classifier_Importance_Pct", ascending=False)

fi_df.to_csv("Feature_Importance_Report.csv", index=False)
print("\nTop 5 Most Influential Features:")
print(fi_df.head(5).to_string(index=False))

# 2. Generate Predictions across all IPOs (including 2026 unlisted IPOs) for Dashboard
X_all = df_feat[FEATURE_COLS]
df_master["Win_Probability_Pct"] = (clf_model.predict_proba(X_all)[:, 1] * 100).round(2)
df_master["Expected_Gain_Pct"] = reg_expected.predict(X_all).round(2)
df_master["Conservative_25th_Pct"] = reg_low.predict(X_all).round(2)
df_master["Optimistic_75th_Pct"] = reg_high.predict(X_all).round(2)

def assign_signal(row):
    prob = row["Win_Probability_Pct"]
    gain = row["Expected_Gain_Pct"]
    if prob >= 70 and gain >= 20:
        return "STRONG APPLY (Bumper Potential)"
    elif prob >= 55 and gain >= 8:
        return "MODERATE APPLY (Listing Gain)"
    else:
        return "AVOID / HIGH RISK"

df_master["Model_Signal"] = df_master.apply(assign_signal, axis=1)
df_master.to_csv("IPO_Predictions_Dashboard.csv", index=False)

# 3. Save bundled model artifact
model_bundle = {
    "features": FEATURE_COLS,
    "classifier": clf_model,
    "reg_expected": reg_expected,
    "reg_low": reg_low,
    "reg_high": reg_high
}
joblib.dump(model_bundle, "ipo_prediction_model.pkl")

print("\n[SUCCESS] Saved Hybrid Model Bundle to 'ipo_prediction_model.pkl'")
print("[SUCCESS] Saved Feature Importances to 'Feature_Importance_Report.csv'")
print("[SUCCESS] Saved Full Dashboard Data to 'IPO_Predictions_Dashboard.csv'")