import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS -----------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem; font-weight: 800;
        color: #064e3b; text-align: center; padding: 0.4rem 0;
    }
    .sub-title {
        font-size: 1rem; color: #666;
        text-align: center; margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 12px; padding: 1.1rem 1.4rem;
        color: white; text-align: center;
    }
    .kpi-label { font-size: 0.8rem; opacity: 0.85; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; }
    .section-hdr {
        font-size: 1.2rem; font-weight: 700; color: #064e3b;
        border-left: 4px solid #059669;
        padding-left: 0.7rem; margin: 1.1rem 0 0.7rem 0;
    }
    .insight {
        background: #ecfdf5; border-left: 4px solid #059669;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.93rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 42px; background-color: #f1f3f5;
        border-radius: 8px 8px 0 0; padding: 0 18px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #059669 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# -- Data loading & modelling --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

@st.cache_data
def run_model(df):
    df_enc = df.copy()
    df_enc['sex']    = LabelEncoder().fit_transform(df_enc['sex'])
    df_enc['smoker'] = LabelEncoder().fit_transform(df_enc['smoker'])
    df_enc = pd.get_dummies(df_enc, columns=['region'], drop_first=True)

    X = df_enc.drop('charges', axis=1)
    y = df_enc['charges']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df = coef_df.reindex(
        coef_df['Coefficient'].abs().sort_values(ascending=False).index)

    return model, X, X_train, X_test, y_train, y_test, y_pred, r2, rmse, mae, coef_df, df_enc

df = load_data()
model, X, X_train, X_test, y_train, y_test, y_pred, r2, rmse, mae, coef_df, df_enc = run_model(df)

# -- Header --------------------------------------------------------------------
st.markdown('<div class="main-title">Insurance Charges Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-end pipeline: Collection - Preprocessing - EDA - Linear Regression</div>', unsafe_allow_html=True)
st.markdown("---")

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("Filters")
    sel_smoker = st.multiselect("Smoker", df['smoker'].unique(), default=list(df['smoker'].unique()))
    sel_sex    = st.multiselect("Sex",    df['sex'].unique(),    default=list(df['sex'].unique()))
    sel_region = st.multiselect("Region", df['region'].unique(), default=list(df['region'].unique()))
    age_range  = st.slider("Age range", int(df['age'].min()), int(df['age'].max()),
                           (int(df['age'].min()), int(df['age'].max())))
    st.markdown("---")
    st.caption("ENSTA Data Analysis Lab - Lisa Bentaleb - 2026")

df_f = df[
    df['smoker'].isin(sel_smoker) &
    df['sex'].isin(sel_sex) &
    df['region'].isin(sel_region) &
    df['age'].between(age_range[0], age_range[1])
]


if df_f.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()
# -- Tabs ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1 - Collection",
    "2 - Preprocessing",
    "3 - EDA",
    "4 - Modeling",
])

GREEN  = '#059669'
TEAL   = '#10b981'
COLORS = [GREEN, TEAL, '#34d399', '#6ee7b7', '#a7f3d0', '#f59e0b', '#f87171']

# ==============================================================================
# TAB 1 - COLLECTION
# ==============================================================================
with tab1:
    st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Total Records", "Features", "Missing Values", "Duplicates"],
        [f"{len(df):,}", "7", "0", str(df.duplicated().sum())]
    ):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-hdr">Column Descriptions</div>', unsafe_allow_html=True)
    col_desc = {
        "age":      "Age of the primary beneficiary (18-64)",
        "sex":      "Gender of the beneficiary: male / female",
        "bmi":      "Body Mass Index (15.96 - 53.13)",
        "children": "Number of covered dependents (0-5)",
        "smoker":   "Smoking status: yes / no",
        "region":   "US residential area: northeast / northwest / southeast / southwest",
        "charges":  "Medical costs billed by insurance - TARGET variable",
    }
    desc_df = pd.DataFrame(list(col_desc.items()), columns=["Column", "Description"])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-hdr">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=280)

    st.markdown('<div class="section-hdr">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

# ==============================================================================
# TAB 2 - PREPROCESSING
# ==============================================================================
with tab2:
    st.markdown('<div class="section-hdr">Preprocessing Steps</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 1. Missing Values")
        mv = df.isnull().sum().reset_index()
        mv.columns = ['Feature', 'Missing']
        mv['Status'] = mv['Missing'].apply(lambda x: 'Clean' if x == 0 else 'Has nulls')
        st.dataframe(mv, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### 2. Encoding Strategy")
        st.markdown("""
        | Column | Method | Result |
        |---|---|---|
        | `sex` | Label Encoding | female=0, male=1 |
        | `smoker` | Label Encoding | no=0, yes=1 |
        | `region` | One-Hot Encoding | 3 dummy columns |
        """)
        st.markdown("#### 3. No columns dropped")
        st.markdown("All 6 features were kept for modeling. `charges` is the target.")

    st.markdown("#### 4. Class Balance - Categorical Features")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ['sex', 'smoker', 'region']):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor='white')
        ax.set_title(col.capitalize(), fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 5. Train / Test Split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training samples", f"{len(X_train):,} (80%)")
    c2.metric("Test samples",     f"{len(X_test):,} (20%)")
    c3.metric("Features used",    str(len(X.columns)))

    st.markdown('<div class="insight"><b>Summary:</b> No imputation needed - zero missing values. Binary categoricals were label-encoded; region used one-hot encoding with drop_first=True to avoid multicollinearity. Dataset split 80/20 with random_state=42.</div>',
                unsafe_allow_html=True)

# ==============================================================================
# TAB 3 - EDA
# ==============================================================================
with tab3:
    st.markdown(f'<div class="section-hdr">EDA - {len(df_f):,} records (filtered)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Charges",    f"${df_f['charges'].mean():,.0f}")
    c2.metric("Median Charges", f"${df_f['charges'].median():,.0f}")
    c3.metric("Max Charges",    f"${df_f['charges'].max():,.0f}")
    c4.metric("Smoker %",       f"{(df_f['smoker']=='yes').mean()*100:.1f}%")

    # Charges distribution + smoker boxplot
    st.markdown("#### Charges Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(df_f['charges'], bins=40, color=GREEN, edgecolor='white', alpha=0.85)
    axes[0].axvline(df_f['charges'].mean(),   color='red',     linestyle='--',
                    label=f"Mean: ${df_f['charges'].mean():,.0f}")
    axes[0].axvline(df_f['charges'].median(), color='#064e3b', linestyle='--',
                    label=f"Median: ${df_f['charges'].median():,.0f}")
    axes[0].set_title("Distribution of Insurance Charges", fontweight='bold')
    axes[0].set_xlabel("Charges ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    groups = [df_f[df_f['smoker'] == s]['charges'] for s in ['no', 'yes']]
    bp = axes[1].boxplot(groups, tick_labels=['Non-Smoker', 'Smoker'], patch_artist=True)
    bp['boxes'][0].set_facecolor(TEAL)
    bp['boxes'][1].set_facecolor(GREEN)
    axes[1].set_title("Charges by Smoking Status", fontweight='bold')
    axes[1].set_ylabel("Charges ($)")
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">Charges are <b>right-skewed</b>: most people pay under $20k, but a minority (mostly smokers) push the mean well above the median. Smokers occupy an almost entirely separate cost distribution.</div>',
                unsafe_allow_html=True)

    # Age and BMI scatter
    st.markdown("#### Key Feature Relationships")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    colors_s = df_f['smoker'].map({'yes': GREEN, 'no': '#ccc'})
    axes[0].scatter(df_f['age'], df_f['charges'], c=colors_s, alpha=0.5, s=18)
    axes[0].set_title("Age vs Charges", fontweight='bold')
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Charges ($)")
    import matplotlib.patches as mpatches
    axes[0].legend(handles=[
        mpatches.Patch(color=GREEN, label='Smoker'),
        mpatches.Patch(color='#ccc',  label='Non-Smoker'),
    ])
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].scatter(df_f['bmi'], df_f['charges'], c=colors_s, alpha=0.5, s=18)
    axes[1].set_title("BMI vs Charges", fontweight='bold')
    axes[1].set_xlabel("BMI")
    axes[1].set_ylabel("Charges ($)")
    axes[1].legend(handles=[
        mpatches.Patch(color=GREEN, label='Smoker'),
        mpatches.Patch(color='#ccc',  label='Non-Smoker'),
    ])
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">Smokers (green) cluster into a high-cost band across all ages and BMI values. Among non-smokers, a second high-cost cluster emerges at high BMI, likely obese patients with chronic conditions.</div>',
                unsafe_allow_html=True)

    # Region and children + correlation heatmap
    st.markdown("#### Region, Children and Correlations")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    region_means = df_f.groupby('region')['charges'].mean().sort_values(ascending=False)
    axes[0].bar(region_means.index, region_means.values,
                color=COLORS[:len(region_means)], edgecolor='white')
    axes[0].set_title("Avg Charges by Region", fontweight='bold')
    axes[0].set_ylabel("Avg Charges ($)")
    axes[0].set_ylim(0, region_means.max() * 1.2)
    for i, v in enumerate(region_means.values):
        axes[0].text(i, v + 50, f"${v:,.0f}", ha='center', fontsize=9)
    axes[0].spines[['top', 'right']].set_visible(False)

    children_means = df_f.groupby('children')['charges'].mean()
    axes[1].bar(children_means.index.astype(str), children_means.values,
                color=GREEN, edgecolor='white')
    axes[1].set_title("Avg Charges by Number of Children", fontweight='bold')
    axes[1].set_xlabel("Number of Children")
    axes[1].set_ylabel("Avg Charges ($)")
    for i, v in enumerate(children_means.values):
        axes[1].text(i, v + 50, f"${v:,.0f}", ha='center', fontsize=9)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df_enc.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='Greens',
                ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap (encoded)", fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight"><b>Smoker</b> has a correlation of 0.79 with charges - nearly as predictive as the entire rest of the feature set combined. Age (0.30) and BMI (0.20) follow as secondary predictors.</div>',
                unsafe_allow_html=True)

# ==============================================================================
# TAB 4 - MODELING
# ==============================================================================
with tab4:
    st.markdown('<div class="section-hdr">Linear Regression - Predicting Insurance Charges</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2 Score",  f"{r2:.4f}", delta="vs mean baseline")
    c2.metric("RMSE",      f"${rmse:,.2f}")
    c3.metric("MAE",       f"${mae:,.2f}")
    c4.metric("Test size", f"{len(X_test):,} records")

    st.markdown("")

    st.markdown("#### Model Fit: Actual vs Predicted")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.35, color=GREEN, s=15)
    lim = (min(y_test.min(), y_pred.min()) - 500,
           max(y_test.max(), y_pred.max()) + 500)
    axes[0].plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_title("Actual vs Predicted Charges", fontweight='bold')
    axes[0].set_xlabel("Actual ($)")
    axes[0].set_ylabel("Predicted ($)")
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.35, color=TEAL, s=15)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted", fontweight='bold')
    axes[1].set_xlabel("Predicted ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">The model fits well for most patients but <b>under-predicts high-cost smokers</b> (upper-right cluster sits above the red diagonal). The residual plot shows two fans, confirming the model cannot fully capture the non-linear smoker x BMI interaction.</div>',
                unsafe_allow_html=True)

    st.markdown("#### Feature Coefficients")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors_c = ['#059669' if v >= 0 else '#f87171' for v in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title("Linear Regression Coefficients", fontweight='bold')
        ax.set_xlabel("Coefficient Value ($)")
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col_b:
        st.markdown("**Coefficient Table**")
        display = coef_df.copy()
        display['Coefficient'] = display['Coefficient'].apply(lambda x: f"${x:+,.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown(f"**Intercept:** ${model.intercept_:,.2f}")

    st.markdown("#### Interpretation and Key Findings")
    findings = [
        ("Smoking adds +$23,651 to annual charges",
         "The smoker coefficient is far larger than any other feature, effectively splitting the population into two cost regimes: smokers and non-smokers."),
        ("Age adds ~$257 per year",
         "Each additional year of age consistently adds about $257 to annual insurance charges, reflecting increasing healthcare needs with age."),
        ("BMI adds ~$337 per unit",
         "Higher BMI increases costs, especially when combined with smoking. This non-linear interaction is not fully captured by the linear model."),
        ("Each child adds ~$425",
         "Dependents add a modest but consistent cost increase to the policyholder's charges."),
        ("Region has minimal impact",
         "Regional coefficients are all under $810, indicating that geography is not a major pricing factor in this dataset."),
        ("R2 = 0.7836: solid but improvable",
         "The model explains ~78% of variance. The remaining ~22% is mainly due to the non-linear smoker x BMI interaction. A polynomial or tree-based model would improve RMSE significantly."),
    ]
    for title, body in findings:
        st.markdown(f'<div class="insight"><b>{title}</b><br>{body}</div>',
                    unsafe_allow_html=True)

    st.markdown("#### Try a Prediction")
    with st.expander("Predict insurance charges for a custom profile", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            age_in    = st.slider("Age", 18, 64, 35)
            sex_in    = st.selectbox("Sex", ["male", "female"])
            bmi_in    = st.slider("BMI", 15.0, 55.0, 28.0, step=0.5)
        with c2:
            children_in = st.slider("Children", 0, 5, 1)
            smoker_in   = st.radio("Smoker", ["no", "yes"])
            region_in   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        sex_enc    = 1 if sex_in == "male" else 0
        smoker_enc = 1 if smoker_in == "yes" else 0
        region_nw  = 1 if region_in == "northwest"  else 0
        region_se  = 1 if region_in == "southeast"  else 0
        region_sw  = 1 if region_in == "southwest"  else 0

        X_new = np.array([[age_in, sex_enc, bmi_in, children_in, smoker_enc,
                           region_nw, region_se, region_sw]])
        pred_val = model.predict(X_new)[0]
        pred_val = max(0, pred_val)

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Charges", f"${pred_val:,.2f}")
        c2.metric("vs Dataset Mean",   f"${pred_val - df['charges'].mean():+,.2f}")
        risk = "High Risk" if pred_val > 20000 else "Medium Risk" if pred_val > 10000 else "Low Risk"
        c3.metric("Risk Category", risk)