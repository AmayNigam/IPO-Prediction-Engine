import joblib
import numpy as np
import pandas as pd

# 1. Load the Hybrid Model Bundle
bundle = joblib.load("ipo_prediction_model.pkl")
feature_cols = bundle["features"]
clf_model = bundle["classifier"]
reg_expected = bundle["reg_expected"]
reg_low = bundle["reg_low"]
reg_high = bundle["reg_high"]

def engineer_single_ipo(ipo_dict):
    d = pd.DataFrame([ipo_dict])
    d["Log_Issue_Size"] = np.log1p(d["Issue_Size_Cr"].clip(lower=0))
    d["Log_Total_Sub"] = np.log1p(d["Total_Sub_x"].clip(lower=0))
    d["Log_QIB_Sub"] = np.log1p(d["QIB_Sub_x"].clip(lower=0))
    d["Log_HNI_Sub"] = np.log1p(d["HNI_Sub_x"].clip(lower=0))
    d["Log_Retail_Sub"] = np.log1p(d["Retail_Sub_x"].clip(lower=0))

    d["QIB_to_Retail_Ratio"] = (d["QIB_Sub_x"] + 1.0) / (d["Retail_Sub_x"] + 1.0)
    d["HNI_to_Retail_Ratio"] = (d["HNI_Sub_x"] + 1.0) / (d["Retail_Sub_x"] + 1.0)
    d["Institutional_Dominance"] = (d["QIB_Sub_x"] + d["HNI_Sub_x"]) / (d["Total_Sub_x"] + 1.0)

    d["Total_Capital_Demanded_Cr"] = d["Issue_Size_Cr"] * d["Total_Sub_x"]
    d["Log_Capital_Demanded"] = np.log1p(d["Total_Capital_Demanded_Cr"].clip(lower=0))
    d["Scarcity_Index"] = d["Total_Sub_x"] / ((d["Issue_Size_Cr"] / 100.0) + 1.0)
    return d[feature_cols]

# 2. Test on an upcoming / live IPO
new_ipo = {
    "Company_Name": "Sample Tech IPO Ltd",
    "Issue_Size_Cr": 3200.0,
    "Issue_Price_INR": 450.0,
    "Total_Sub_x": 48.0,
    "QIB_Sub_x": 82.0,
    "HNI_Sub_x": 36.0,
    "Retail_Sub_x": 9.5
}

X_new = engineer_single_ipo(new_ipo)

win_prob = clf_model.predict_proba(X_new)[0, 1] * 100
exp_gain = reg_expected.predict(X_new)[0]
low_gain = min(reg_low.predict(X_new)[0], exp_gain)
high_gain = max(reg_high.predict(X_new)[0], exp_gain)

issue_price = new_ipo["Issue_Price_INR"]
exp_price = issue_price * (1 + exp_gain / 100)
low_price = issue_price * (1 + low_gain / 100)
high_price = issue_price * (1 + high_gain / 100)

if win_prob >= 70 and exp_gain >= 20:
    verdict = "STRONG APPLY (High Institutional Conviction)"
elif win_prob >= 55 and exp_gain >= 8:
    verdict = "MODERATE APPLY (Subscribe for Listing Gain)"
else:
    verdict = "AVOID / HIGH RISK (Weak Demand Dynamics)"

print("\n========================================================")
print(f" IPO EVALUATION REPORT: {new_ipo['Company_Name']}")
print("========================================================")
print(f" Issue Size         : INR {new_ipo['Issue_Size_Cr']} Cr | Issue Price: INR {issue_price}")
print(f" Subscription       : Total {new_ipo['Total_Sub_x']}x (QIB: {new_ipo['QIB_Sub_x']}x, HNI: {new_ipo['HNI_Sub_x']}x, Retail: {new_ipo['Retail_Sub_x']}x)")
print("--------------------------------------------------------")
print(f" Profitable Listing Probability : {win_prob:.1f}%")
print(f" Expected Listing Gain          : {exp_gain:+.2f}%  (Est. Price: INR {exp_price:.2f})")
print(f" 25th-75th Percentile Range     : {low_gain:+.2f}% to {high_gain:+.2f}% (INR {low_price:.2f} - INR {high_price:.2f})")
print("--------------------------------------------------------")
print(f" FINAL MODEL SIGNAL : {verdict}")
print("========================================================\n")