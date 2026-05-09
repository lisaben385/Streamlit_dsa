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
    page_title="Shopping Behavior Analysis",
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
    return pd.read_csv("Streamlit/Shopping/shopping_behavior_updated.csv")

@st.cache_data
def run_model(df):
    cat_cols = ['Gender', 'Category', 'Location', 'Size', 'Color', 'Season',
                'SubscriptionStatus', 'DiscountApplied', 'PaymentMethod',
                'FrequencyofPurchases']
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

    X = df_enc.drop(['Customer ID', 'PurchaseAmount', 'ItemPurchased'], axis=1)
    y = df_enc['PurchaseAmount']
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
st.markdown('<div class="main-title">Shopping Behavior Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-end pipeline: Collection - Preprocessing - EDA - Linear Regression</div>', unsafe_allow_html=True)
st.markdown("---")

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("Filters")
    sel_gender   = st.multiselect("Gender",   df['Gender'].unique(),   default=list(df['Gender'].unique()))
    sel_category = st.multiselect("Category", df['Category'].unique(), default=list(df['Category'].unique()))
    sel_season   = st.multiselect("Season",   df['Season'].unique(),   default=list(df['Season'].unique()))
    sel_sub      = st.multiselect("Subscription", df['SubscriptionStatus'].unique(),
                                  default=list(df['SubscriptionStatus'].unique()))
    st.markdown("---")
    st.caption("ENSTA Data Analysis Lab - Lisa Bentaleb - 2026")

df_f = df[
    df['Gender'].isin(sel_gender) &
    df['Category'].isin(sel_category) &
    df['Season'].isin(sel_season) &
    df['SubscriptionStatus'].isin(sel_sub)
]

# GUARD
if df_f.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()

# -- Tabs ----
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
        [f"{len(df):,}", "16", "0", str(df.duplicated().sum())]
    ):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-hdr">Column Descriptions</div>', unsafe_allow_html=True)
    col_desc = {
        "Customer ID":            "Unique identifier - not used in modeling",
        "Age":                    "Customer age (18-70)",
        "Gender":                 "Male / Female",
        "Item Purchased":         "Name of the purchased item",
        "Category":               "Product category: Clothing, Accessories, Footwear, Outerwear",
        "Purchase Amount (USD)":  "Transaction value - TARGET variable",
        "Location":               "US state of purchase",
        "Size":                   "Item size: S / M / L / XL",
        "Color":                  "Item color",
        "Season":                 "Season of purchase: Spring / Summer / Fall / Winter",
        "Review Rating":          "Customer rating (2.5-5.0)",
        "Subscription Status":    "Whether customer has a subscription (Yes/No)",
        "Discount Applied":       "Whether a discount was used (Yes/No)",
        "Previous Purchases":     "Number of prior purchases by this customer",
        "Payment Method":         "Payment method used",
        "Frequency of Purchases": "How often the customer shops",
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
        st.markdown("#### 2. Columns Dropped Before Modeling")
        st.markdown("""
        | Column | Reason |
        |---|---|
        | `Customer ID` | Non-informative identifier |
        | `Item Purchased` | Too granular (hundreds of unique items) |
        """)
        st.markdown("#### 3. Encoding Strategy")
        st.markdown("""
        All remaining categorical columns were encoded with **Label Encoding**:
        `Gender`, `Category`, `Location`, `Size`, `Color`, `Season`,
        `SubscriptionStatus`, `DiscountApplied`, `PaymentMethod`,
        `FrequencyofPurchases`
        """)

    st.markdown("#### 4. Class Balance - Categorical Features")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    cat_plot_cols = ['Gender', 'Category', 'Season', 'SubscriptionStatus',
                     'DiscountApplied', 'FrequencyofPurchases']
    for ax, col in zip(axes, cat_plot_cols):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor='white')
        ax.set_title(col, fontweight='bold', fontsize=10)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=30)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 10, str(v), ha='center', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 5. Train / Test Split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training samples", f"{len(X_train):,} (80%)")
    c2.metric("Test samples",     f"{len(X_test):,} (20%)")
    c3.metric("Features used",    str(len(X.columns)))

    st.markdown('<div class="insight"><b>Summary:</b> No imputation needed - zero missing values. All categorical columns were label-encoded. Customer ID and Item Purchased were dropped. Dataset split 80/20 with random_state=42.</div>',
                unsafe_allow_html=True)

# ==============================================================================
# TAB 3 - EDA
# ==============================================================================
with tab3:
    st.markdown(f'<div class="section-hdr">EDA - {len(df_f):,} records (filtered)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Purchase",    f"${df_f['PurchaseAmount'].mean():,.2f}")
    c2.metric("Median Purchase", f"${df_f['PurchaseAmount'].median():,.0f}")
    c3.metric("Max Purchase",    f"${df_f['PurchaseAmount'].max():,.0f}")
    c4.metric("Avg Rating",      f"{df_f['ReviewRating'].mean():.2f} stars")

    st.markdown("#### Purchase Amount Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(df_f['PurchaseAmount'], bins=30, color=GREEN, edgecolor='white', alpha=0.85)
    axes[0].axvline(df_f['PurchaseAmount'].mean(),   color='red',     linestyle='--',
                    label=f"Mean: ${df_f['PurchaseAmount'].mean():.1f}")
    axes[0].axvline(df_f['PurchaseAmount'].median(), color='#064e3b', linestyle='--',
                    label=f"Median: ${df_f['PurchaseAmount'].median():.1f}")
    axes[0].set_title("Distribution of Purchase Amount", fontweight='bold')
    axes[0].set_xlabel("Purchase Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].boxplot(
        [df_f[df_f['Gender'] == g]['PurchaseAmount'] for g in ['Male', 'Female']],
        tick_labels=['Male', 'Female'], patch_artist=True,
        boxprops=dict(facecolor=TEAL, alpha=0.6))
    axes[1].set_title("Purchase Amount by Gender", fontweight='bold')
    axes[1].set_ylabel("Purchase Amount ($)")
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">Purchase amounts are <b>nearly uniformly distributed</b> between $20-$100. The mean and median are almost identical (~$60), confirming no skew. Gender has essentially <b>no effect</b> on spend ($60.2 vs $59.5).</div>',
                unsafe_allow_html=True)

    st.markdown("#### Purchase Amount by Category and Season")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    cat_means = df_f.groupby('Category')['PurchaseAmount'].mean().sort_values(ascending=False)
    axes[0].bar(cat_means.index, cat_means.values, color=COLORS[:len(cat_means)], edgecolor='white')
    axes[0].set_title("Avg Purchase by Category", fontweight='bold')
    axes[0].set_ylabel("Avg Purchase ($)")
    axes[0].set_ylim(0, cat_means.max() * 1.2)
    for i, v in enumerate(cat_means.values):
        axes[0].text(i, v + 0.3, f"${v:.1f}", ha='center', fontsize=9)
    axes[0].spines[['top', 'right']].set_visible(False)

    season_means = df_f.groupby('Season')['PurchaseAmount'].mean().sort_values(ascending=False)
    axes[1].bar(season_means.index, season_means.values, color=COLORS[:len(season_means)], edgecolor='white')
    axes[1].set_title("Avg Purchase by Season", fontweight='bold')
    axes[1].set_ylabel("Avg Purchase ($)")
    axes[1].set_ylim(0, season_means.max() * 1.2)
    for i, v in enumerate(season_means.values):
        axes[1].text(i, v + 0.3, f"${v:.1f}", ha='center', fontsize=9)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">Category and season differences are <b>trivially small</b> (under $3 spread). All categories and seasons serve roughly the same average spend.</div>',
                unsafe_allow_html=True)

    st.markdown("#### Top Items and Payment Methods")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    top_items = df_f['ItemPurchased'].value_counts().head(10)
    axes[0].barh(top_items.index[::-1], top_items.values[::-1], color=GREEN)
    axes[0].set_title("Top 10 Items Purchased", fontweight='bold')
    axes[0].set_xlabel("Count")
    axes[0].spines[['top', 'right']].set_visible(False)

    pay_counts = df_f['PaymentMethod'].value_counts()
    axes[1].pie(pay_counts.values, labels=pay_counts.index, autopct='%1.1f%%',
                colors=COLORS[:len(pay_counts)], startangle=140)
    axes[1].set_title("Payment Method Distribution", fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Feature Correlations with Purchase Amount")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].scatter(df_f['Age'], df_f['PurchaseAmount'], alpha=0.2, color=GREEN, s=12)
    axes[0].set_title("Age vs Purchase Amount", fontweight='bold')
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Purchase Amount ($)")
    axes[0].spines[['top', 'right']].set_visible(False)

    num_corr = df_enc[['Age', 'ReviewRating', 'PreviousPurchases', 'PurchaseAmount']]\
        .corr()['PurchaseAmount'].drop('PurchaseAmount').sort_values()
    colors_bar = ['#059669' if v >= 0 else '#f87171' for v in num_corr.values]
    axes[1].barh(num_corr.index, num_corr.values, color=colors_bar)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title("Numeric Feature Correlations", fontweight='bold')
    axes[1].set_xlabel("Pearson Correlation")
    for i, v in enumerate(num_corr.values):
        axes[1].text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=9)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">All numeric correlations with PurchaseAmount are <b>below 0.03</b> - effectively zero. No feature has a linear relationship with spend. This is the key insight that explains why the linear model performs poorly.</div>',
                unsafe_allow_html=True)

# ==============================================================================
# TAB 4 - MODELING
# ==============================================================================
with tab4:
    st.markdown('<div class="section-hdr">Linear Regression - Predicting Purchase Amount</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2 Score",  f"{r2:.4f}", delta="vs mean baseline")
    c2.metric("RMSE",      f"${rmse:.2f}")
    c3.metric("MAE",       f"${mae:.2f}")
    c4.metric("Test size", f"{len(X_test):,} records")

    st.markdown("")

    st.markdown("#### Model Fit: Actual vs Predicted")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, color=GREEN, s=15)
    lim = (min(y_test.min(), y_pred.min()) - 2,
           max(y_test.max(), y_pred.max()) + 2)
    axes[0].plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_title("Actual vs Predicted Purchase Amount", fontweight='bold')
    axes[0].set_xlabel("Actual ($)")
    axes[0].set_ylabel("Predicted ($)")
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, color=TEAL, s=15)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted", fontweight='bold')
    axes[1].set_xlabel("Predicted ($)")
    axes[1].set_ylabel("Residual ($)")
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="insight">The scatter plot shows a <b>horizontal cloud</b> with no diagonal trend - the model predictions are essentially random. The residuals are large and structureless, confirming that linear regression cannot capture the data pattern.</div>',
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
        display['Coefficient'] = display['Coefficient'].apply(lambda x: f"{x:+.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown(f"**Intercept:** ${model.intercept_:.2f}")

    st.markdown("#### Interpretation and Key Findings")
    findings = [
        ("R2 = -0.013: the model has no predictive power",
         "A negative R2 means the model is worse than predicting the mean every time. This is the most important result of the entire analysis."),
        ("All feature correlations are below 0.03",
         "No feature - age, gender, category, season, subscription status - has any meaningful linear relationship with purchase amount."),
        ("Purchase amount is uniformly distributed ($20-$100)",
         "The target variable itself is nearly flat across its range. A linear model cannot learn from a uniform distribution."),
        ("Better approaches for this dataset",
         "This data is better suited for clustering (customer segmentation), classification (predicting subscription status or frequency), or association rules (item co-purchase patterns)."),
        ("What the data does tell us",
         "The customer base is diverse and balanced. Clothing and Accessories dominate sales. Male customers are 68% of transactions. Spending is consistent across all seasons and demographics."),
    ]
    for title, body in findings:
        st.markdown(f'<div class="insight"><b>{title}</b><br>{body}</div>',
                    unsafe_allow_html=True)

   
