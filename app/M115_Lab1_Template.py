#!/usr/bin/env python
# coding: utf-8

# # M115 Lab 1: Phishing URL Detection — EDA, Parameterized Logistic Regression & SHAP Explainability
# 
# ---
# 
# ## 🏛️ ENSIASD Taroudant — Ibn Zohr University
# **Master in Intelligent Systems and Cyber Defense (Module M115: AI & Intelligent Systems)**  
# **Instructor:** Prof. Sara EL-ATEIF | **Academic Year:** 2026–2027
# 
# ---
# 
# ## 🧭 Lab Overview & Solution Key
# 
# This notebook contains the assignment of **Lab 1** on the **[PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/kaggleprollc/phishing-url-websites-dataset-phiusiil/data)** (*235,795 URLs and 56 lexical, network, and HTML features*, Prasad & Chandra, *Computers & Security*, 2024).
# 
# ### 📋 Table of Contents
# * **1 - Importing Packages & Loading Data**
# * **2 - Part 1: Security-Oriented Exploratory Data Analysis (EDA)**
#   * 2.1 Data Quality Inspection & SOC Target Alignment (Exercise 1)
#   * 2.2 Class Balance, Numerical Distributions & Binary Security Indicators (Exercise 2)
#   * 2.3 Identifier Cleanup & Feature–Target Correlations (Exercise 3)
#   * *Part 1 Reflection Questions*
# * **3 - Part 2: Zero-Data Leakage Preprocessing & Standardization**
#   * 3.1 Stratified 75/25 Split & Training-Only StandardScaler (Exercise 4)
#   * *Part 2 Reflection Questions*
# * **4 - Part 3: Parameterized Logistic Regression & Interpretability**
#   * 4.1 Comparing 5 Logistic Regression Configurations (Exercise 5)
#   * 4.2 L1 Regularization Path & Feature Selection (Exercise 6)
#   * 4.3 Interpreting Coefficients ($w_j$) and Odds Ratios ($e^{w_j}$) (Exercise 7)
#   * *Part 3 Reflection Questions*
# * **5 - Part 4: Explainable AI (SHAP) & SOC Threshold Analysis**
#   * 5.1 Global Explanations with `shap.LinearExplainer` (Exercise 8)
#   * 5.2 Local SOC Alert Triage: TP, TN, and FP Waterfall Plots (Exercise 9)
#   * 5.3 Operational Decision Threshold Tuning ($\tau \in \{0.20, 0.50, 0.85\}$) (Exercise 10)
#   * *Part 4 Reflection Questions*
# * **6 - Conclusion & SOC Engineering Synthesis**

# ---
# ## 1 - Importing Packages & Loading Data
# 
# Run the cell below to import the required libraries and configure reproducibility (`SEED = 42`):

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score
)     
import shap                                                    

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 4.5)
SEED = 42
np.random.seed(SEED)


# ---
# ## 2 - Part 1: Security-Oriented Exploratory Data Analysis (EDA)
# 
# ### 2.1 Data Quality Inspection & SOC Target Alignment (Exercise 1)
# 
# First, we load `PhiUSIIL_Phishing_URL_Dataset.csv` and inspect its shape, data types, missing values, duplicate rows, and summary statistics across key numerical features (`URLLength`, `LineOfCode`, `CharContinuationRate`, `NoOfSelfRef`).

# In[5]:


# Load dataset and inspect data quality
file_path = 'Lab1/PhiUSIIL_Phishing_URL_Dataset.csv'

df = pd.read_csv(file_path)
print("Dataset Shape:", df.shape)
print("Total Missing Values:", df.isna().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("\nDataset Info:")
df.info()

display(df[['FILENAME', 'URL', 'URLLength', 'LineOfCode', 'IsHTTPS', 'HasPasswordField', 'NoOfSelfRef', 'label']].head(5))
display(df[['URLLength', 'LineOfCode', 'CharContinuationRate', 'NoOfSelfRef']].describe().round(2))


# > **⚠️ Critical Cyber Label Warning in PhiUSIIL:**  
# > In the raw CSV file, **`label = 1` denotes Legitimate (Benign)** and **`label = 0` denotes Phishing (Malicious)**.  
# > In a SOC, the positive class (`1`) **must always represent the attack** (`Phishing = 1`, `Benign = 0`) so that True Positives (TP), Precision, and Recall measure our threat detection capability.
# 
# ### Exercise 1

# In[ ]:


# Data Quality & SOC Target Alignment (1 = Phishing, 0 = Benign)

df['IsPhishing'] = 1-df['label']
phishing_ratio = df['IsPhishing'].mean()

print(f"Phishing URLs (1): {(df['IsPhishing'] == 1).sum():,}")
print(f"Legitimate URLs (0): {(df['IsPhishing'] == 0).sum():,}")
print(f"Phishing Proportion: {phishing_ratio * 100:.2f}%")
print("\nNormalized Class Proportions:")
print(df['IsPhishing'].value_counts(normalize=True).round(4))


# ---
# ### 2.2 Class Balance, Numerical Distributions & Binary Security Indicators (Exercise 2)
# 
# We compare legitimate (`0`) and phishing (`1`) URLs across class balance, continuous lexical/HTML features (`URLLength`, `CharContinuationRate`, `NoOfSelfRef`), and binary security indicators (`IsHTTPS`, `HasPasswordField`, `HasCopyrightInfo`, `HasSocialNet`).
# 
# ### Exercise 2

# In[ ]:


# Visualizing Distributions and Binary Security Indicators
binary_cols = ['IsHTTPS', 'HasPasswordField', 'HasCopyrightInfo', 'HasSocialNet']
num_cols = ['URLLength', 'CharContinuationRate', 'NoOfSelfRef']

binary_stats = df[binary_cols].mean()

print("Percentage (%) of URLs presenting each binary indicator per class (0=Benign, 1=Phishing):")
display(binary_stats.round(2))

# 1. Class Balance & Binary Indicators Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.countplot(data=df, x='IsPhishing', hue='IsPhishing', ax=axes[0], palette=['#16a34a', '#dc2626'], legend=False)
axes[0].set_title("Class Balance (0 = Legitimate, 1 = Phishing)")
axes[0].set_xlabel("IsPhishing (0 = Benign, 1 = Phishing)")
axes[0].set_ylabel("URL Count")

binary_stats.T.plot(kind='bar', ax=axes[1], color=['#16a34a', '#dc2626'])
axes[1].set_title("Prevalence (%) of Binary Security Indicators by Class")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_xticklabels(binary_cols, rotation=15)
axes[1].legend(["Legitimate (0)", "Phishing (1)"])
plt.tight_layout()
plt.show()

# 2. Numerical Distributions (KDE Plots clipped at 99th percentile for readability)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num_cols):
    upper_clip = df[col].quantile(0.99)
    sns.kdeplot(
        data=df[df[col] <= upper_clip],
        x=col,
        hue='IsPhishing',
        common_norm=False,
        fill=True,
        palette=['#16a34a', '#dc2626'],
        ax=axes[i]
    )
    axes[i].set_title(f"Distribution of {col} (<=99th pct)")
plt.tight_layout()
plt.show()


# ---
# ### 2.3 Identifier Cleanup & Feature–Target Correlations (Exercise 3)
# 
# We drop non-generalizable raw text strings (`FILENAME`, `URL`, `Domain`, `TLD`, `Title`), the old `label`, and the pre-computed `URLSimilarityIndex` score so our model learns from genuine lexical and HTML indicators. Then we compute the Pearson correlation of each numerical feature with `IsPhishing`.
# 
# ### Exercise 3

# In[ ]:


# Constructing X, y and Analyzing Target Correlations
cols_to_drop = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label', 'IsPhishing', 'URLSimilarityIndex']

X = df.drop(columns=cols_to_drop)
y = df['IsPhishing']
correlations = X.corrwith(y).sort_values(ascending=False)

print("Shape of X:", X.shape, "| Shape of y:", y.shape)

# Plot Top 15 Negatively and Top 15 Positively Correlated Features
top_corr = pd.concat([correlations.head(15), correlations.tail(15)])
plt.figure(figsize=(10, 7.5))
colors = ['#16a34a' if c < 0 else '#dc2626' for c in top_corr.values]
top_corr.plot(kind='barh', color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Top 15 Negatively (Benign) and Positively (Phishing) Correlated Features with IsPhishing")
plt.xlabel("Pearson Correlation Coefficient (r)")
plt.tight_layout()
plt.show()


# ---
# ### 💡 Part 1 Reflection Questions
# 
# 1. **Looking at your plot for `IsHTTPS`, what percentage of phishing sites currently use an HTTPS certificate? Why is a static rule *"If HTTPS then Legitimate"* dangerous?**
#    * **Answer:** Although HTTPS has negative correlation with phishing sites approximately -0.60, this does not mean that all HTTPS websites are legitimate. A sighificant proportion of websites can also use HTTPS certifcates, including free certificates from services such as Let's encrypt.
# 
# 2. **Why do phishing websites typically have very few internal links (`NoOfSelfRef`) but very frequently contain a password field (`HasPasswordField`)?**
#    * **Answer:** Phishing websites typically contain very few internal links because they are designed for a specific purpose, such as collecting user's credential, rather than providing a complet website fonctionality. However they contain frequently password fields because attakers aime to steal sensitive information.
# 
# 3. **What would happen in production if we left the `FILENAME`, `URL`, or `Domain` column in our training matrix?**
#    * **Answer:** Kepping the filename, URL or Domain columns could introduce data leakage or overfitting, the model might memorise filenames, specific domains, instead of learning mningful characteristics of phishing websites. Consecently it acheive very high accuracy during training and testing but perform poorly in production when encountering previously unseen URLs and Domains.

# ---
# ## 3 - Part 2: Zero-Data Leakage Preprocessing & Standardization
# 
# ### 3.1 Stratified 75/25 Split & Training-Only StandardScaler (Exercise 4)
# 
# We apply the **SecMLOps Golden Rule**: split the dataset into training (75%) and test (25%) subsets *before* computing Z-score normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$).
# 
# ### Exercise 4

# In[ ]:


# Stratified Split and Leakage-Free Standardization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retain column names and indices for weight and SHAP interpretability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Train shape:", X_train_scaled.shape, "| Test shape:", X_test_scaled.shape)
print("Mean of X_train_scaled (URLLength):", round(X_train_scaled['URLLength'].mean(), 4))
print("Std of X_train_scaled (URLLength):", round(X_train_scaled['URLLength'].std(), 4))


# ---
# ### 💡 Part 2 Reflection Questions
# 
# 1. **Compare the scale of `LineOfCode` (values up to 10,000+) and `IsHTTPS` (0 or 1) before standardization. Why would L1/L2 regularization be skewed without `StandardScaler`?**
#    * **Answer:** Features have different scales. This affects coefficient sizes, causing L1/L2 regularization to penalize them unevenly. StandardScaler puts features on comparable scales.
# 
# 2. **Precisely explain why calling `scaler.fit_transform(X)` before `train_test_split` constitutes Preprocessing Data Leakage.**
#    * **Answer:** Because the scaler uses information from the entire dataset, including the test set, to calculate the mean and standard deviation. The correct approach is to split first, fit the scaler on training data, then transform both sets.

# ---
# ## 4 - Part 3: Parameterized Logistic Regression & Interpretability
# 
# ### 4.1 Comparing 5 Logistic Regression Configurations (Exercise 5)
# 
# We train and evaluate the five specified Logistic Regression configurations on `X_train_scaled` and `X_test_scaled`:
# * **M1: Unregularized (`penalty=None`)**
# * **M2: Strong L2 Ridge (`penalty='l2', C=0.001`)**
# * **M3: Standard L2 Ridge (`penalty='l2', C=1.0`)**
# * **M4: Sparse L1 Lasso (`penalty='l1', C=0.01`)**
# * **M5: SOC Weighted (`penalty='l2', C=1.0, class_weight='balanced'`)**
# 
# ### Exercise 5

# In[ ]:


# Training and Comparing 5 Logistic Regression Configurations
configurations = {
    "M1: Unregularized (None)": LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=SEED),
    "M2: Strong L2 Ridge (C=0.001)": LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', max_iter=500, random_state=SEED),
    "M3: Standard L2 Ridge (C=1.0)": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=500, random_state=SEED),
    "M4: Sparse L1 Lasso (C=0.01)": LogisticRegression(penalty='l1', C=0.01, solver='liblinear', max_iter=500, random_state=SEED),
    "M5: SOC Weighted (balanced)": LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=500, random_state=SEED),
}

results = []
for name, model in configurations.items():
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    w = model.coef_.ravel()
    n_zero_weights = np.sum(np.isclose(w, 0, atol=1e-6))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({
        "Model": name,
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall (TPR)": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
        "False Alarms (FP)": int(fp),
        "Missed Phish (FN)": int(fn),
        "Zeroed Weights (Sparsity)": f"{n_zero_weights} / {len(w)}"
    })

results_df = pd.DataFrame(results)
display(results_df)


# ---
# ### 4.2 L1 Regularization Path & Feature Selection (Exercise 6)
# 
# We vary `C` across `[0.002, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]` with `penalty='l1'` (`solver='liblinear'`) and plot the number of retained features ($|w_j| \ge 10^{-4}$) as a function of `C` on a logarithmic scale.
# 
# ### Exercise 6

# In[ ]:


# L1 Regularization Path (Number of Active Features vs. C)
C_values = [0.002, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
active_features_count = []

for c_val in C_values:
    m_l1 = LogisticRegression(penalty='l1', C=c_val, solver='liblinear', max_iter=500, random_state=SEED)
    m_l1_fit = m_l1.fit(X_train_scaled, y_train)
    n_active = np.sum(np.abs(m_l1_fit.coef_.ravel()) >= 1e-4)
    active_features_count.append(n_active)

path_df = pd.DataFrame({"C": C_values, "Active_Features": active_features_count, "Zeroed_Features": [X.shape[1] - k for k in active_features_count]})
display(path_df)

plt.figure(figsize=(8, 4))
plt.plot(C_values, active_features_count, marker='o', linewidth=2, color='#0284c7')
plt.xscale('log')
plt.xlabel("Inverse Regularization Strength C (log scale)")
plt.ylabel("Number of Retained Features (|w_j| >= 1e-4)")
plt.title(f"L1 Lasso Regularization Path (Total Features = {X.shape[1]})")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# ---
# ### 4.3 Interpreting Coefficients ($w_j$) and Odds Ratios ($e^{w_j}$) (Exercise 7)
# 
# We extract the learned log-odds weights $w_j$ from `M3: Standard L2 Ridge (C=1.0)`, compute the Odds Ratio $e^{w_j}$, compare against `M4: Sparse L1 Lasso (C=0.01)`, and display the **Top 8 Phishing** ($w_j > 0$) and **Top 8 Benign** ($w_j < 0$) features.
# 
# ### Exercise 7

# In[ ]:


# Inspecting Log-Odds (w_j) and Odds Ratios (exp(w_j))
model_l2 = configurations["M3: Standard L2 Ridge (C=1.0)"]
model_l1 = configurations["M4: Sparse L1 Lasso (C=0.01)"]

weights_df = weights_df = pd.DataFrame({
    "Feature": X.columns,
    "L2_Weight": model_l2.coef_.ravel(),
    "L2_Odds_Ratio": np.exp(model_l2.coef_.ravel()),
    "L1_Weight": model_l1.coef_.ravel(),
    "L1_Odds_Ratio": np.exp(model_l1.coef_.ravel())
}).set_index("Feature").sort_values("L2_Weight", ascending=False)

print("🔴 TOP 8 FEATURES PUSHING TOWARD PHISHING (w_j > 0):")
display(weights_df.head(8).round(4))

print("\n🟢 TOP 8 FEATURES PUSHING TOWARD BENIGN (w_j < 0):")
display(weights_df.tail(8).round(4))


# In[13]:


print("M4 zero weights:", np.sum(np.isclose(model_l1.coef_, 0, atol=1e-6)))
print("M3 F1:", f1_score(y_test, configurations["M3: Standard L2 Ridge (C=1.0)"].predict(X_test_scaled)))
print("M4 F1:", f1_score(y_test, model_l1.predict(X_test_scaled)))


# ---
# ### 💡 Part 3 Reflection Questions —
# 
# 1. **How many variables did `M4: Sparse L1 Lasso (C=0.01)` eliminate (weights set to zero)? Did the F1-Score drop significantly? What is the advantage for a real-time firewall?**
#    * **Answer:** M4 eliminated 24 features by setting their weights to zero using L1 regularization.
# The F1-Score decreased slightly from 0.99925 (M3) to 0.99821 (M4), showing that feature selection had very little impact on model performance.
# For a real-time firewall, this is beneficial because a simpler model requires fewer computations and can make faster predictions while maintaining excellent detection performance.
# 
# 2. **What is the impact of `class_weight='balanced'` (M5) on False Positives and False Negatives compared to M3?**
#    * **Answer:** class_weight='balanced' gives more importance to the minority class. It often reduces False Negatives (missed phishing attacks), but may increase False Positives (legitimate websites incorrectly flagged as phishing).
# 
# 3. **How do you interpret a positive weight $w_j > 0$ versus a negative weight $w_j < 0$ in the formula $\sigma(\mathbf{w}^\top\mathbf{x} + b)$?**
#    * **Answer:** Positive weight (\(w_j > 0\)): Increases the phishing probability.
#                  
#                  Negative weight (\(w_j < 0\)): Decreases the phishing probability, pushing the prediction toward benign.

# ---
# ## 5 - Part 4: Explainable AI (SHAP) & SOC Threshold Analysis
# 
# ### 5.1 Global Explanations with `shap.LinearExplainer` (Exercise 8)
# 
# We initialize `shap.LinearExplainer` on `model_l2` (`M3`) using `X_train_scaled` as the background reference distribution, compute exact Shapley values on the first `1,000` test URLs, and plot the global Bar and Beeswarm charts (`max_display=15`).
# 
# ### Exercise 8

# In[ ]:


# Computing SHAP Values and Global Importance Plots (1,000 Test URLs)
X_test_sample = X_test_scaled.iloc[:1000]

explainer = shap.LinearExplainer(model_l2, X_train_scaled)
shap_values = explainer(X_test_sample)

# 1. Global Mean Absolute SHAP Importance (Bar Plot)
shap.plots.bar(shap_values, max_display=15)

# 2. Global Directional Impact (Beeswarm Plot)
shap.plots.beeswarm(shap_values, max_display=15)


# ---
# ### 5.2 Local SOC Alert Triage: TP, TN, and FP Waterfall Plots (Exercise 9)
# 
# We locate the first **True Positive (TP)**, **True Negative (TN)**, and **False Positive (FP — False Alarm)** in our 1,000-URL test sample and inspect their local SHAP Waterfall plots (`max_display=10`), passing unscaled `X_test` values in `data=` so SOC analysts can read actual URL metrics.
# 
# ### Exercise 9

# In[ ]:


# Local SOC Triage of TP, TN, and FP Cases with SHAP Waterfall
y_test_sample = y_test.iloc[:1000].values
y_prob_sample = model_l2.predict_proba(X_test_sample)[:, 1]
y_pred_sample = (y_prob_sample >= 0.50).astype(int)

tp_idx = int(np.where((y_test_sample == 1) & (y_pred_sample == 1))[0][0])
tn_idx = int(np.where((y_test_sample == 0) & (y_pred_sample == 0))[0][0])
fp_candidates = np.where((y_test_sample == 0) & (y_pred_sample == 1))[0]

fp_idx = int(fp_candidates[0]) if len(fp_candidates) > 0 else int(np.argmax(y_prob_sample * (y_test_sample == 0)))

def plot_soc_waterfall(idx, case_title):
    print(f"=== {case_title} (Sample Index={idx}) | True Label={y_test_sample[idx]} | P(Phishing)={y_prob_sample[idx]*100:.2f}% ===")
    local_exp = shap.Explanation(
        values=shap_values.values[idx],
        base_values=float(np.ravel(shap_values.base_values)[idx]),
        data=X_test.iloc[idx].values, # Unscaled values for human analyst readability
        feature_names=X.columns
    )
    shap.plots.waterfall(local_exp, max_display=10)

plot_soc_waterfall(tp_idx, "CASE 1: TRUE POSITIVE (Confirmed Phishing Attack Caught)")
plot_soc_waterfall(tn_idx, "CASE 2: TRUE NEGATIVE (Legitimate Traffic Correctly Cleared)")
plot_soc_waterfall(fp_idx, "CASE 3: FALSE POSITIVE (Legitimate Site Wrongly Flagged — False Alarm)")


# ---
# ### 5.3 Operational Decision Threshold Tuning ($\tau \in \{0.20, 0.50, 0.85\}$) (Exercise 10)
# 
# We evaluate `model_l2` (`M3`) on the full test set across three operational decision thresholds:
# * **$\tau = 0.20$ (Threat Hunting Mode):** Maximizes Recall to catch stealthy phishing campaigns.
# * **$\tau = 0.50$ (Default Mode):** Balanced threshold.
# * **$\tau = 0.85$ (Automated Firewall Blocking Mode):** Maximizes Precision and minimizes False Positives to prevent blocking legitimate business traffic.
# 
# ### Exercise 10

# In[ ]:


# Comparing Operational Decision Thresholds (0.20, 0.50, 0.85)
y_prob_full = model_l2.predict_proba(X_test_scaled)[:, 1]
thresholds = [0.20, 0.50, 0.85]
threshold_records = []

for tau in thresholds:
    y_pred_tau = (y_prob_full >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tau).ravel()

    threshold_records.append({
        "Threshold (tau)": tau,
        "SOC Operational Mode": "Threat Hunting" if tau == 0.20 else ("Default" if tau == 0.50 else "Automated Blocking"),
        "Precision": round(precision_score(y_test, y_pred_tau), 4),
        "Recall (TPR)": round(recall_score(y_test, y_pred_tau), 4),
        "F1-Score": round(f1_score(y_test, y_pred_tau), 4),
        "True Positives (TP)": int(tp),
        "True Negatives (TN)": int(tn),
        "False Positives (FP)": int(fp),
        "False Negatives (FN)": int(fn)
    })

threshold_df = pd.DataFrame(threshold_records)
display(threshold_df)


# ---
# ### 💡 Part 4 Reflection Questions —
# 
# 1. **What is the mathematical difference between inspecting a global weight $w_j$ and looking at a local SHAP value $\phi_j(x)$ for a specific URL?**
#    * **Answer:** A global weight \(w_j\) represents the overall influence of a feature in the Logistic Regression model. A local SHAP value \(\phi_j(x)\) explains how much that feature contributes to the prediction of one specific URL.
# 
# 2. **On your False Positive (FP) Waterfall plot, which 2 or 3 features misled the model and caused this false alarm?**
#    * **Answer:** Identify the features with the largest positive SHAP values in your FP waterfall plot. These features increased the predicted phishing probability and caused the legitimate URL to be incorrectly classified as phishing.
# 
# 3. **Why would a SOC deploy $\tau = 0.85$ for automated DNS/firewall blocking while routing URLs with $0.20 \le P(\text{Phishing}) < 0.85$ to a human analyst queue?**
#    * **Answer:** A high threshold (0.85) reduces false positives, preventing legitimate websites from being blocked automatically. URLs with intermediate probabilities are sent to human analysts for verification, balancing security and usability.

# ---
# ## 6 - Conclusion & SOC Engineering Synthesis (Deliverable Solution)
# 
# > ### 🛡️ SOC Engineering Synthesis Report
# > 1. **Hyperparameter & Sparsity Trade-Off:**
# 
# > 2. **Global Interpretability (Odds Ratios & SHAP):**
# 
# > 3. **SOC Alert Triage & Dual-Threshold Policy:**
#   * *False Positive Triage Note (3 sentences):*
# 
#   * *Recommended Threshold Policy:*

#  ### 🛡️ SOC Engineering Synthesis Report
#  
#  1. **Hyperparameter & Sparsity Trade-Off:**
# 
# L1 regularization eliminated 24 features while the F1-Score decreased only slightly, from 0.99925 (M3) to 0.99821 (M4). This produces a simpler and more efficient model with almost the same detection performance, making it suitable for real-time security applications.
# 
# 2. **Global Interpretability (Odds Ratios & SHAP):**
# 
# Logistic Regression weights and odds ratios show how each feature influences phishing predictions. SHAP values provide additional explanations by showing each feature's contribution to individual predictions. Together, they improve model transparency and help security analysts understand suspicious URLs.
# 
# 3. **SOC Alert Triage & Dual-Threshold Policy:**
# 
# False Positive Triage Note (3 sentences):
# 
# The False Positive case represents a legitimate website incorrectly classified as phishing. The SHAP waterfall plot helps identify which features contributed most to this incorrect prediction. A SOC analyst should investigate the URL and verify its legitimacy before taking action.
# 
# Recommended Threshold Policy:
# - P(Phishing) below 0.20: Treat as low risk and allow access.
# - 0.20 ≤ P(Phishing) < 0.85: Send an alert to the SOC analyst for manual investigation.
# - P(Phishing) ≥ 0.85: Automatically block the URL, subject to security policy
