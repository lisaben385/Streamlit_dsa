# Data Analysis Project — ENSTA Lab
**Student:** Lisa Bentaleb | **Branch:** bentaleb-lisa | **Year:** 2026

---

# Part 1 — Shopping Behavior Dataset

## Dataset Overview

**Source:** Kaggle — Customer Shopping Behavior  
**File:** `shopping_behavior_updated.csv`  
**Size:** 3,900 records × 16 columns  
**Target variable:** `PurchaseAmount (USD)`

---

## Lab 0 — Dataset Selection & Description

The shopping behavior dataset is a structured, tabular dataset where each row represents one customer purchase transaction. It falls into the medium-sized category — large enough to draw statistically meaningful conclusions, but small enough to process quickly.

**Column types:**

| Column | Type | Description |
|---|---|---|
| `Age` | Numerical (discrete) | Customer age (18–70) |
| `PurchaseAmount` | Numerical (continuous) | Amount spent in USD — TARGET |
| `ReviewRating` | Numerical (continuous) | Rating from 2.5 to 5.0 |
| `PreviousPurchases` | Numerical (discrete) | Count of past purchases |
| `Gender` | Categorical (nominal) | Male / Female |
| `Item Purchased` | Categorical (nominal) | Product name |
| `Category` | Categorical (nominal) | Clothing / Footwear / Outerwear / Accessories |
| `Location` | Categorical (nominal) | US state |
| `Size` | Categorical (ordinal) | S / M / L / XL |
| `Color` | Categorical (nominal) | Item color |
| `Season` | Categorical (nominal) | Spring / Summer / Fall / Winter |
| `Subscription Status` | Categorical (binary) | Yes / No |
| `Discount Applied` | Categorical (binary) | Yes / No |
| `Payment Method` | Categorical (nominal) | Credit Card, PayPal, Venmo, etc. |
| `Frequency of Purchases` | Categorical (ordinal) | Weekly → Annually |

---

## Lab 1 — Data Story & Descriptive Statistics

Built a narrative around the data using descriptive statistics and visualizations.

**Who are the shoppers?**
- Average age: **44 years old** (range: 18–70)
- **Male customers dominate**: 2,652 vs 1,248 female (roughly 2:1 ratio)

**How much do they spend?**
- Average purchase: **$59.76**
- Standard deviation: $23.69
- Range: $20 to $100 — no extreme outliers

**What do they buy?**

| Category | Avg Spend | Count |
|---|---|---|
| Clothing | $60.03 | 1,737 |
| Accessories | $59.84 | 1,240 |
| Footwear | $60.26 | 599 |
| Outerwear | $57.17 | 324 |

All categories cluster around the same $60 average — no category is dramatically more expensive.

**Other findings:**
- Women spend slightly more per transaction ($60.25 vs $59.54) — negligible difference
- Subscribers and non-subscribers spend almost identically ($59.49 vs $59.87)

**Early signal:** The near-identical averages across all groups hinted early on that no single feature meaningfully separates spending behavior.

---

## Lab 2 — Inferential Statistics (Shopping)

Applied t-tests, chi-square tests, and correlation analysis to the shopping dataset.

**Results:**
- No statistically significant difference in purchase amounts between genders
- Product category preference is independent of gender (chi-square test)
- All variables show weak correlation with `PurchaseAmount`
- The uniform distribution of values limits the depth of statistical insight

**Conclusion:** The statistical procedures were correctly applied, but the dataset structure itself limits the findings. Variables are relatively balanced and weakly correlated, with no strong demographic effect on spending amount or product category selection.

---

## Lab 3 — Preprocessing

- Verified **zero missing values** across all 16 columns
- Removed `Customer ID` — non-informative identifier
- Removed `Item Purchased` — too granular (hundreds of unique values)
- Applied **Label Encoding** to binary categorical columns
- Applied **One-Hot Encoding** to multi-class categorical columns
- Applied `StandardScaler` to numerical features (`Age`, `PreviousPurchases`)
- Train/Test split: **80% training / 20% test**, random_state=42

---

## Lab 5 — Exploratory Data Analysis (Shopping)

**Key findings:**

- `PurchaseAmount` is **nearly uniformly distributed** between $20 and $100 — mean and median are almost identical (~$60), confirming no skew
- All groups (gender, category, season, subscription) show differences under $3 in average spend
- No feature shows a visible linear pattern when plotted against `PurchaseAmount`

| Feature | Correlation with PurchaseAmount |
|---|---|
| ReviewRating | 0.031 |
| Age | -0.010 |
| PreviousPurchases | 0.008 |

All correlations are effectively zero. The EDA confirmed what the descriptive statistics already suggested — the dataset is extremely balanced and does not contain the structured relationships that regression modeling requires.

---

## Lab 7 — Linear Regression (Shopping)

Applied Linear Regression to predict `PurchaseAmount` from all available features.

**Results:**

| Metric | Value |
|---|---|
| R² Score | -0.028 |
| RMSE | $23.98 |
| MAE | $20.88 |

**Interpretation:**

A negative R² means the model performs worse than simply predicting the mean for every customer. With a MAE of ~$21 on items priced between $20 and $100, the error rate is approximately 20–25%.

This is not a modeling failure — the pipeline was correctly implemented with proper encoding, scaling, and train/test splitting. The problem is the data itself: `PurchaseAmount` is uniformly distributed with near-zero correlation to every feature. There is no linear signal for the model to learn.

**Why the model could not work:**
- The target variable has no skew, no outliers, no natural groupings — it looks like a flat, random distribution
- No demographic variable has any meaningful influence on spend
- Shopping behavior in this dataset is driven by individual factors not captured in the columns

**Conclusion:** Linear regression is the wrong tool for this dataset. It is better suited for customer segmentation (clustering), classification of behavioral patterns, or association rule mining — approaches that do not require a continuous numeric relationship between features and a target.

---
---

# Part 2 — Insurance Charges Dataset

## Why the Dataset Was Switched

After completing all labs on the shopping dataset, the results from the EDA and modeling phases were not demonstrating the course concepts in a meaningful way:

- The EDA produced flat, uninformative charts — no visible patterns, no separating groups, no trends
- The linear regression model produced R² = -0.028, meaning it learned nothing from the data
- The coefficient table showed values near zero for every feature — nothing interpretable

To properly demonstrate linear regression — including how to read coefficients, evaluate model fit, and connect statistical findings to real-world meaning — the **Medical Insurance Charges** dataset was introduced. This dataset has real structure: smoking, age, and BMI all have strong, measurable linear relationships with charges, making it the right case for the modeling lab.

The switch was not to avoid difficulty, but to produce results that are actually meaningful and worth interpreting.

---

## Dataset Overview

**Source:** Kaggle — Medical Cost Personal Datasets  
**File:** `insurance.csv`  
**Size:** 1,338 records × 7 columns  
**Target variable:** `charges` (annual medical costs in USD)

| Column | Type | Description |
|---|---|---|
| `age` | Numerical | Age of beneficiary (18–64) |
| `sex` | Categorical | male / female |
| `bmi` | Numerical | Body Mass Index (15.96–53.13) |
| `children` | Numerical | Number of covered dependents (0–5) |
| `smoker` | Categorical | yes / no |
| `region` | Categorical | northeast / northwest / southeast / southwest |
| `charges` | Numerical | Annual medical costs — TARGET |

---

## Lab 2 — Inferential Statistics (Insurance)

The same inferential techniques were applied to the insurance dataset alongside the shopping analysis. The contrast was immediate and clear.

| Test | Result | Finding |
|---|---|---|
| 95% Confidence Interval | ($10,786 — $14,143) | True mean ($13,270) sits inside — reliable estimation |
| T-test: Smokers vs Non-smokers | t=46.66, p≈8.27×10⁻²⁸³ | Smokers pay $32,050 vs $8,434 — difference of $23,616/year |
| ANOVA: Regional differences | F=2.97, p=0.031 | At least one region differs significantly |
| Chi-square: Sex vs Smoking | χ²=7.39, p=0.0065 | Men smoke at higher rates than women |

Smoking status is the dominant cost driver — the t-test result (difference of $23,616) foreshadows exactly what the regression model finds later.

---

## EDA — Insurance Dataset

- Charges are **right-skewed**: mean ($13,270) is well above median ($9,382), driven by high-cost smokers
- Clear bimodal distribution — one cluster for non-smokers, one for smokers
- Scatter plots show visible linear trends between `age` and `charges`, and `bmi` and `charges`

| Feature | Correlation with charges |
|---|---|
| smoker (encoded) | 0.787 |
| age | 0.299 |
| bmi | 0.198 |
| children | 0.068 |
| sex | 0.057 |
| region | < 0.075 |

---

## Linear Regression — Insurance Dataset

**Preprocessing:**
- Label Encoding: `sex`, `smoker`
- One-Hot Encoding: `region` (drop_first=True to avoid multicollinearity)
- Train/Test split: 80/20, random_state=42

**Results:**

| Metric | Value |
|---|---|
| R² Score | 0.7836 |
| RMSE | $5,796 |
| MAE | $4,181 |

**Coefficients:**

| Feature | Coefficient | Interpretation |
|---|---|---|
| smoker | +$23,651 | Dominant predictor — smoking adds ~$23k to annual charges |
| age | +$257/year | Each additional year adds ~$257 |
| bmi | +$337/unit | Each BMI unit adds ~$337 |
| children | +$425 | Each dependent adds ~$425 |
| sex | -$19 | Negligible effect |
| region | -$370 to -$810 | Modest regional variation |

**Key insight:** The smoker coefficient (+$23,651) matches almost exactly the group difference found by the t-test ($23,616). The regression and the inferential statistics tell the same story — this consistency confirms the analysis is correct end to end.

**Model limitation:** R²=0.78 is strong but not perfect. The remaining 22% comes from the non-linear smoker × BMI interaction — obese smokers cost exponentially more than a linear model can capture. A tree-based model would improve this further.

---

## Dashboard Deployment — Streamlit

Both datasets were deployed as interactive Streamlit dashboards covering all pipeline stages: Collection, Preprocessing, EDA, and Modeling.

**Installation:**
```powershell
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

**Run shopping dashboard:**
```powershell
cd "path\to\Streamlit\shopping"
streamlit run app.py
```

**Run insurance dashboard:**
```powershell
cd "path\to\Streamlit\insurance"
streamlit run app.py
```

Both apps open automatically in the browser at `localhost:8501`. To run both simultaneously, use two separate terminal windows — the second app will open at `localhost:8502`.

---

## Repository Structure

```
bentaleb-lisa/
├── Lab0/
│   └── README.md                   - Dataset selection and type classification
├── Lab1/
│   └── README.md                   - Data story and descriptive statistics
├── Lab2/
│   └── README.md                   - Inferential statistics (shopping dataset)
├── Lab2 (new dataset)/
│   └── README.md                   - Inferential statistics (insurance dataset)
├── Lab3/
│   └── preprocessing.ipynb         - Encoding, scaling, train/test split
├── Lab5/
│   └── eda.ipynb                   - EDA visualizations and correlation analysis
├── Lab7/
│   ├── linear_regression_lab.ipynb
│   ├── Regression_shopping.py
│   └── READme.md                   - Regression results and interpretation
├── Streamlit/
│   ├── shopping/
│   │   ├── app.py
│   │   └── shopping_behavior_updated.csv
│   └── insurance/
│       ├── app.py
│       └── insurance.csv
└── README.md                       
```

---

*ENSTA Data Analysis Course — Lisa Bentaleb — 2026*