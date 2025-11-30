# PowerCo SME Churn Prediction

End-to-end data science project modelling customer churn for **PowerCo**, an energy provider serving small and medium enterprises (SMEs).  
Built as part of the **BCG X Data Science** job simulation and expanded into a production-style GitHub project.

The goal is to **predict which SME customers are at risk of churning** and to translate model outputs into **actionable retention strategies**.

---

## 1. Business context

PowerCo has observed rising churn in its SME portfolio.  
Management believes that *price sensitivity* is the main driver, but there is limited analytical evidence.

Key business questions:

1. Is price really the dominant driver of churn, or are other factors more important?
2. Can we build a model that predicts which SME customers are likely to leave?
3. How should the business use these predictions to prioritise retention actions?

The dataset contains:

- **14,606 SME customers** with historical churn labels (0 = retained, 1 = churned).
- **193,002 price time-series records** with variable and fixed tariffs by date.
- Rich customer-level information: yearly and monthly consumption, margins, forecasted consumption and prices, contract dates and channels, gas/electricity flags, etc.

---

## 2. Project pipeline

This repository implements a full CRISP-DM-style pipeline aligned to the six simulation tasks.

### 2.1 Business understanding (Task 1–2)

- Summarises the client context, SME segment and commercial risk of churn.
- Frames churn prediction as a **binary classification** problem.
- Drafts a structured email to the senior data scientist specifying:
  - Required data (billing, consumption, contract, service, price history, market benchmarks).
  - Potential churn drivers beyond price (usage level, contract type, tenure, service issues, etc.).
  - Planned analytics steps: EDA, feature engineering, modelling, evaluation, and insight translation.

### 2.2 Exploratory data analysis & cleaning (Task 3)

Notebook: `03_eda_and_data_cleaning.ipynb`  
Key steps:

- Robust CSV loading with fallback paths and explicit schema checks.
- Date parsing for `date_activ`, `date_end`, `date_modif_prod`, `date_renewal`, `price_date`.
- Binary encoding of `churn` and audit of missingness (0% in all modelling fields).
- Descriptive statistics for 18 numeric and 8 categorical client features.
- Base churn rate: **9.7%** of customers churn.
- Visual exploration:
  - Histograms for consumption, margins, net margin, and tenure.
  - Boxplots for `cons_12m`, `forecast_cons_12m`, `pow_max` to highlight skew and extreme outliers.
  - Churn rate by `channel_sales`, `origin_up`, and `has_gas`.
- Construction of **customer-level price features** from the time series (per `id`):
  - Last, median, standard deviation and slope for variable and fixed price levels.
- Initial hypothesis tests:
  - Point-biserial correlations between engineered price features and churn.
  - Churn rate by quantile of key price variables.

Findings:

- Price-related features show **only weak positive correlation** with churn (|r| = 0.04–0.05).
- Margins and some forecast variables are more strongly associated with churn.
- Dataset is clean, with no duplicate `id` values and complete coverage of price features.

### 2.3 Feature engineering (Task 4)

Notebook: `04_feature_engineering.ipynb`

Engineered features include:

- **Estelle’s Dec→Jan price deltas** (per customer, latest year with both months):
  - `offpeak_diff_dec_january_energy` (variable price).
  - `offpeak_diff_dec_january_power` (fixed price).
- **Mean and max monthly price differences** across off-peak, mid-peak and peak tariffs for both variable and fixed prices.
- **Temporal features** derived from contract dates:
  - `tenure` (years between activation and end).
  - `months_activ`, `months_to_end`, `months_modif_prod`, `months_renewal` relative to 2016-01-01.
- **Categoricals**:
  - `has_gas` converted to 0/1.
  - One-hot encoding for `channel_sales` and `origin_up`, with rare dummies dropped (<100 observations).
- **Skew handling**:
  - Log10(x+1) transform for heavily skewed variables such as `cons_12m`, `cons_last_month`, `forecast_cons_12m`, and forecasted price and discount fields.
  - Before/after histograms saved to `figures/features/`.

Correlation heatmaps (saved to `figures/features/`) are used to drop a small number of highly collinear fields (`num_years_antig`, `forecast_cons_year`).

Output:

- Final modelling table: **14,606 rows × 63 columns**, stored as `outputs/clean_data_with_features.csv`.
- Compact feature dictionary: `outputs/feature_dictionary.csv`.

### 2.4 Modelling & evaluation (Task 5)

Notebook: `05_model_training_and_evaluation_rf.ipynb`

Model:

- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`).
- 80/20 **stratified train–test split** to preserve churn rate.
- Key hyperparameters:
  - `n_estimators=500`
  - `class_weight="balanced"`
  - `random_state=42`

Test-set metrics:

| Metric     | Score  | Interpretation |
|-----------|--------|----------------|
| Accuracy  | **0.911** | Excellent overall hit-rate, driven by majority (non-churn) class |
| Precision (churn=1) | **0.897** | When the model predicts churn, it is usually correct |
| Recall (churn=1)    | **0.092** | Only 9% of actual churners are detected |
| F1 (churn=1)        | **0.166** | Low overall effectiveness in catching churners |

Key visualisations (saved under `figures/models/`):

- **Confusion matrix** heatmap highlighting:
  - 2,635 true negatives,
  - 3 false positives,
  - 258 false negatives,
  - 26 true positives.
- **Feature importance bar chart** (top 30 drivers).

### Model Performance

#### 🔹 Confusion Matrix – Random Forest
![Confusion Matrix](figures/Confusion%20Matrix-Random%20Forest.png)

#### 🔹 Feature Importance – Random Forest
![Feature Importance](figures/Feature%20Importance-Random%20Forest.png)

Top churn drivers according to the Random Forest:

1. `cons_12m` – 12-month electricity consumption.
2. `margin_gross_pow_ele` and `margin_net_pow_ele` – profitability of the power contract.
3. `forecast_meter_rent_12m` and `net_margin`.
4. `forecast_cons_12m` and `cons_last_month`.
5. Temporal features such as `months_activ` and `months_modif_prod`.
6. Selected price-sensitivity variables and off-peak/peak differentials.

Insight:

> **Usage patterns and profitability signals matter more than raw price levels**.  
> Churn is not purely a “cheap competitor” problem; it is tied to how valuable and engaged the customer is.

### 2.5 Executive summary (Task 6)

File: `reports/executive_summary_powerco.pdf`

One-slide executive summary for senior stakeholders:

- **Recommended solution**: deploy a churn-prediction system and prioritise retention offers to high-risk SME customers.
- **Situation**: 10% churn across 14.6k SME records; Random Forest model built to estimate churn probability.
- **Complication**: strong overall accuracy but **very low recall**, meaning most at-risk customers are currently missed.
- **Key drivers**: 12-month consumption, contract margins and relationship length.
- **Key question**: how to use these insights to improve recall and design targeted retention campaigns.

---

## 3. How to run this project

### 3.1 Setup

```bash
git clone https://github.com/<your-username>/powerco-sme-churn-prediction.git
cd powerco-sme-churn-prediction

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

