# Phishing URL Detection Using Machine Learning

A machine-learning cybersecurity project that detects phishing URLs using Logistic Regression and explains model predictions using SHAP. The project also explores how a Security Operations Center (SOC) can use classification thresholds to prioritize and respond to suspicious URLs.

## Project Overview

Phishing attacks use deceptive URLs and websites to steal sensitive information, such as passwords and banking credentials.

The objective of this project is to build a model that classifies URLs into two categories:

- **Benign (0):** Legitimate URLs.
- **Phishing (1):** Malicious or suspicious URLs.

The project uses the **PhiUSIIL Phishing URL Dataset** and focuses on model training, evaluation, interpretability, and practical SOC decision-making.

## Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SHAP
- Git and GitHub

## Project Workflow

### 1. Data Preparation
- Load and inspect the dataset.
- Explore phishing and benign class distributions.
- Analyze correlations between numerical features and the target.
- Remove selected identifier and raw-text columns from the model input.
- Prepare the target variable `IsPhishing`.

### 2. Data Preprocessing
- Separate features (`X`) and target (`y`).
- Split the dataset into training (75%) and testing (25%) sets.
- Use stratified splitting to preserve class proportions.
- Apply `StandardScaler` to numerical features.
- Fit the scaler only on the training set to prevent data leakage.

### 3. Logistic Regression Experiments

Five configurations are compared:

| Model | Configuration |
|---|---|
| M1 | Unregularized Logistic Regression |
| M2 | L2 regularization, C = 0.001 |
| M3 | L2 regularization, C = 1.0 |
| M4 | L1 regularization, C = 0.01 |
| M5 | L2 regularization, C = 1.0, balanced class weights |

Evaluation metrics include:
- Precision
- Recall
- F1-Score
- Precision-Recall AUC
- False Positives (FP)
- False Negatives (FN)

### 4. L1 Regularization and Feature Selection

Different values of `C` are tested to study the effect of L1 regularization on feature selection.

L1 regularization can force some feature coefficients to zero, producing a simpler and potentially faster model.

### 5. Model Interpretability

Two interpretability approaches are explored:

**Logistic Regression coefficients and Odds Ratios**
- Positive coefficients indicate an increase in phishing odds.
- Negative coefficients indicate a decrease in phishing odds.
- Odds ratios help interpret the direction and magnitude of feature effects.

**SHAP (SHapley Additive exPlanations)**
- Global feature importance using bar plots.
- Global feature impact using beeswarm plots.
- Local explanations using waterfall plots.
- Investigation of True Positive, True Negative, and False Positive cases.

### 6. SOC Decision Thresholds

The model is evaluated using three classification thresholds:

| Threshold | SOC Operational Mode |
|---|---|
| 0.20 | Threat Hunting |
| 0.50 | Default Classification |
| 0.85 | Automated Blocking |

A dual-threshold policy is proposed:

- **Probability below 0.20:** Low-risk classification.
- **Probability from 0.20 to below 0.85:** Send for human investigation.
- **Probability of 0.85 or higher:** Candidate for automated blocking, subject to security policy.

## Experimental Results

The following results were obtained from the test set:

| Model | F1-Score | Zeroed Features |
|---|---:|---:|
| M3 – Standard L2 | 0.99925 | — |
| M4 – Sparse L1 | 0.99821 | 24 |

The L1 model eliminated 24 features while maintaining a very similar F1-Score to the standard L2 model.

This demonstrates a promising trade-off between model simplicity and predictive performance.

*Note: These are the observed results from the current experiment. Results may vary depending on preprocessing, dataset splits, and model settings.*

## Repository Structure

```text
phishing-url-detection/
│
├── notebooks/
│   └── phishing_url_detection.ipynb
│
├── app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```


## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/phishing-url-detection.git
cd phishing-url-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook using Jupyter or upload it to Google Colab.

## Dataset

Dataset: `PhiUSIIL_Phishing_URL_Dataset.csv`

Place the dataset in the appropriate local data directory or load it through Google Drive in Colab.

The dataset is not necessarily included in this repository. Check its license and distribution conditions before redistributing it.

## Limitations

- High test-set performance does not guarantee the same performance on new, unseen phishing campaigns.
- Random train/test splitting may allow URLs from the same domains to appear in both sets.
- A domain-separated evaluation would provide a stronger test of generalization to unseen domains.
- SHAP explanations describe model behavior; they do not prove that a feature causes phishing.
- Automated blocking should include safeguards against false positives.

## Future Improvements

- Test additional classification algorithms such as Random Forest, XGBoost, and SVM.
- Evaluate generalization using domain-based train/test splitting.
- Build an interactive Streamlit application for URL analysis.
- Add URL feature extraction for new URLs.
- Integrate explainable predictions into a SOC-style dashboard.
- Evaluate the model on more recent phishing datasets.

## Disclaimer

This project is developed for educational and cybersecurity research purposes. Model predictions should not be considered definitive proof that a URL is malicious or safe. Any automated blocking system should include appropriate validation and safeguards.

## Author

**IMOURIG Moustapha**

GitHub: [musta-007](https://github.com/musta-007)

