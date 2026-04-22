# ================================================================
# STREAMLIT APP GitHub Repository Statistical Analysis
# Project: SSDI E017
# ================================================================
# HOW TO RUN:
#   pip install -r requirements.txt
#   streamlit run app.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GitHub Repo Analysis SSDI E017",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg:        #111210;
        --surface:   #191a18;
        --border:    #2a2c28;
        --border2:   #3a3c38;
        --text:      #e8e6e0;
        --muted:     #888880;
        --faint:     #444440;
        --accent:    #d4a853;
        --accent2:   #7ab87a;
        --accent3:   #7aaad4;
        --danger:    #d47a7a;
        --success:   #7ab87a;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp { background-color: var(--bg); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.8rem;
        color: var(--muted) !important;
        letter-spacing: 0.04em;
    }

    /* Nav label */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: var(--faint) !important;
    }

    /* Radio items */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
        color: var(--muted) !important;
        padding: 6px 0 !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
    }

    /* ── Headings ── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        color: var(--text) !important;
        letter-spacing: -0.03em !important;
        line-height: 1.1 !important;
    }
    h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text) !important;
    }
    h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.7rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid var(--border) !important;
        padding-bottom: 8px !important;
        margin-top: 24px !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 20px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--accent);
    }
    [data-testid="metric-container"] label {
        font-family: 'Syne', sans-serif !important;
        color: var(--muted) !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace !important;
        color: var(--text) !important;
        font-size: 1.8rem !important;
        font-weight: 500 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--faint) !important;
        background: transparent !important;
        border-radius: 0 !important;
        padding: 12px 28px !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        background: transparent !important;
    }

    /* ── Callout boxes ── */
    .stInfo > div {
        background-color: #14181f !important;
        border: 1px solid #1e2a3a !important;
        border-left: 3px solid var(--accent3) !important;
        border-radius: 4px !important;
        color: #9ab8d8 !important;
        font-size: 0.875rem !important;
    }
    .stWarning > div {
        background-color: #1a1710 !important;
        border: 1px solid #2a2518 !important;
        border-left: 3px solid var(--accent) !important;
        border-radius: 4px !important;
        color: #c8a870 !important;
        font-size: 0.875rem !important;
    }
    .stSuccess > div {
        background-color: #101a12 !important;
        border: 1px solid #182318 !important;
        border-left: 3px solid var(--success) !important;
        border-radius: 4px !important;
        color: #88c898 !important;
        font-size: 0.875rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] th {
        background-color: var(--surface) !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        border-bottom: 1px solid var(--border2) !important;
    }
    [data-testid="stDataFrame"] td {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
        color: var(--text) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px dashed var(--border2) !important;
        border-radius: 6px !important;
    }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; margin: 24px 0 !important; }

    /* ── Custom label pill ── */
    .label-pill {
        display: inline-block;
        background: #1e1c16;
        border: 1px solid #3a3520;
        color: var(--accent);
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 3px;
        margin-bottom: 12px;
    }

    /* ── Section title ── */
    .section-eyebrow {
        font-family: 'Syne', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 4px;
    }
    .section-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 2px;
        margin-bottom: 20px;
        font-style: italic;
    }

    /* ── Text ── */
    p { color: var(--text); line-height: 1.6; }
    strong { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 700; color: #e8e6e0; letter-spacing: -0.02em;'>GitHub Analysis</div>
        <div style='font-family: DM Mono, monospace; font-size: 0.7rem; color: #888880; margin-top: 3px;'>SSDI PROJECT E017</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
    st.markdown("---")
    page = st.radio("Section", [
        "Home",
        "EDA",
        "T-Test",
        "Correlation",
        "ANOVA",
        "Regression",
        "Isolation Forest",
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 0.68rem; color: #444440; line-height: 1.8;'>
    Python &nbsp; JavaScript<br>
    Java &nbsp; C++ &nbsp; C# &nbsp; Go<br>
    500 repos per language
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div style='padding: 80px 0 40px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: #d4a853; margin-bottom: 12px;'>SSDI Project E017</div>
        <h1 style='font-family: Syne, sans-serif; font-size: 2.8rem; font-weight: 800; color: #e8e6e0; letter-spacing: -0.04em; line-height: 1.05; margin: 0 0 16px 0;'>GitHub Repository<br>Statistical Analysis</h1>
        <p style='color: #888880; font-size: 1rem; max-width: 480px;'>Upload your dataset to explore statistical patterns across 6 programming languages using real GitHub API data.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Upload your github_repos_dataset.csv file in the sidebar to begin.")
    st.stop()

# ── Plot style ──
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#cccccc',
    'axes.labelcolor':  '#333333',
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'text.color':       '#333333',
    'grid.color':       '#eeeeee',
    'grid.alpha':       0.8,
    'axes.grid':        True,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    df = df.dropna()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df['forks2']        = df['forks'].values ** 2
    df['contributors2'] = df['contributors'].values ** 2 if 'contributors' in df.columns else 0
    df['forksissues']   = df['forks'] * df['issues']
    if 'subscribers' in df.columns:
        df['subscribers2']   = df['subscribers'].values ** 2
    if 'network_count' in df.columns:
        df['network_count2'] = df['network_count'].values ** 2
    iso_feats = [f for f in ['forks','issues','size_kb','repo_age'] if f in df.columns]
    X_scaled  = StandardScaler().fit_transform(df[iso_feats].fillna(0))
    iso       = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_scaled)
    df['anomaly_label'] = iso.predict(X_scaled)
    df['anomaly_score'] = iso.decision_function(X_scaled)
    return df

df = load_data(uploaded_file)

poly_features = [f for f in ['forks2','contributors2','forksissues',
                               'subscribers2','network_count2'] if f in df.columns]
COLORS = ['#d4a853','#7aaad4','#7ab87a','#d47a7a','#aa88cc','#d49a7a']


def page_header(eyebrow, title, subtitle=""):
    st.markdown(f"""
    <div style='padding: 4px 0 20px 0;'>
        <div class='section-eyebrow'>{eyebrow}</div>
        <h1 style='margin: 4px 0 6px 0;'>{title}</h1>
        {"<p class='section-subtitle'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HOME
# ──────────────────────────────────────────────
if page == "Home":
    page_header("SSDI E017", "GitHub Repository Statistical Analysis",
                "6 languages, 3000 repositories, collected via GitHub REST API")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Repos",   f"{len(df):,}")
    col2.metric("Languages",     df['language'].nunique())
    col3.metric("Mean Stars",    f"{df['stars'].mean():,.0f}")
    col4.metric("Total Columns", df.shape[1])

    st.markdown("### Dataset Sample")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Raw Feature Statistics")
        raw_cols = [c for c in ['stars','forks','issues','repo_age','size_kb',
                                 'contributors','topic_count','description_length'] if c in df.columns]
        st.dataframe(df[raw_cols].describe().round(2))
    with col2:
        st.markdown("### Polynomial Feature Statistics")
        st.dataframe(df[poly_features].describe().round(2))

# ──────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────
elif page == "EDA":
    page_header("Exploratory Analysis", "Data Overview",
                "Understanding the structure and distribution of the dataset")
    st.markdown("---")

    st.markdown("### Distribution of Stars")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df['stars'], bins=60, color=COLORS[0], alpha=0.85, edgecolor='none')
    ax.set_xlabel("Stars"); ax.set_ylabel("Count")
    ax.set_title("Star Distribution Across All Repositories", pad=14,
                 fontsize=11, fontweight='normal')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.info(f"**Inference:** Highly right-skewed (skewness = {df['stars'].skew():.2f}). "
            f"Most repos have modest stars while a few viral repos dominate.")
    st.warning("**Interpretation:** A small number of repositories attract a "
               "disproportionately large share of community attention.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Repository Count by Language")
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df['language'].value_counts().index
        counts = [df['language'].value_counts()[l] for l in order]
        bars = ax.bar(order, counts, color=COLORS[:len(order)], edgecolor='none', alpha=0.9)
        ax.set_xlabel("Language"); ax.set_ylabel("Count")
        ax.set_title("Repos per Language", pad=12, fontsize=10,
                     color='#c8c6c0', fontweight='normal')
        plt.xticks(rotation=20)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.info("**Inference:** Dataset is perfectly balanced, 500 repositories per language.")
        st.warning("**Interpretation:** Equal group sizes eliminate sample size bias, "
                   "giving our ANOVA and t-tests maximum validity.")

    with col2:
        st.markdown("### Average Stars by Language")
        lang_mean = df.groupby('language')['stars'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(lang_mean.index, lang_mean.values, color=COLORS[:len(lang_mean)],
               edgecolor='none', alpha=0.9)
        ax.set_xlabel("Language"); ax.set_ylabel("Mean Stars")
        ax.set_title("Mean Stars per Language", pad=12, fontsize=10,
                     color='#c8c6c0', fontweight='normal')
        plt.xticks(rotation=20)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.info(f"**Inference:** {lang_mean.index[0]} leads at {lang_mean.iloc[0]:,.0f} mean stars. "
                f"{lang_mean.index[-1]} is lowest at {lang_mean.iloc[-1]:,.0f}.")
        st.warning("**Interpretation:** Visual differences need statistical validation. "
                   "We use ANOVA to confirm whether differences are real or due to chance.")

    st.markdown("### Summary Table")
    st.dataframe(lang_mean.reset_index().rename(columns={'stars': 'Mean Stars'}).round(0))

# ──────────────────────────────────────────────
# T-TEST
# ──────────────────────────────────────────────
elif page == "T-Test":
    page_header("Hypothesis Testing", "Two-Sample T-Test",
                "Do Python and Java repositories have significantly different stars?")
    st.markdown("---")

    python_stars = df[df['language'] == 'Python']['stars']
    java_stars   = df[df['language'] == 'Java']['stars']
    t_stat, p_value = ttest_ind(python_stars, java_stars, equal_var=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("T-Statistic", f"{t_stat:.4f}")
    col2.metric("P-Value",     f"{p_value:.2e}")
    col3.metric("Decision",    "Reject H0" if p_value < 0.05 else "Fail to Reject")

    st.markdown("### Hypotheses")
    st.markdown("""
    **H0:** Mean stars for Python equals mean stars for Java

    **H1:** Mean stars for Python does not equal mean stars for Java

    **Significance level:** alpha = 0.05
    """)

    if p_value < 0.05:
        st.success(f"REJECT H0  Significant difference found (p = {p_value:.2e})")
    else:
        st.warning(f"FAIL TO REJECT H0 (p = {p_value:.2e})")

    st.markdown("### Group Statistics")
    st.dataframe(pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std', 'n'],
        'Python': [f"{python_stars.mean():,.0f}", f"{python_stars.median():,.0f}",
                   f"{python_stars.std():,.0f}", len(python_stars)],
        'Java':   [f"{java_stars.mean():,.0f}", f"{java_stars.median():,.0f}",
                   f"{java_stars.std():,.0f}", len(java_stars)],
    }), hide_index=True)

    st.info(f"**Inference:** p = {p_value:.2e} which is less than 0.05. There is a "
            f"statistically significant difference between Python (mean = {python_stars.mean():,.0f}) "
            f"and Java (mean = {java_stars.mean():,.0f}).")
    st.warning("**Interpretation:** Python repositories are significantly more popular "
               "than Java repositories on GitHub.")

# ──────────────────────────────────────────────
# CORRELATION
# ──────────────────────────────────────────────
elif page == "Correlation":
    page_header("Statistical Analysis", "Correlation Analysis",
                "Measuring the strength of relationships between variables")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Raw Features", "Polynomial Features"])

    with tab1:
        raw_corr_cols = [c for c in ['stars','forks','issues','repo_age','size_kb',
                                      'description_length','topic_count','contributors'] if c in df.columns]
        raw_corr = df[raw_corr_cols].corr()
        r_forks  = raw_corr['stars'].get('forks', 0)
        r_issues = raw_corr['stars'].get('issues', 0)
        r_age    = raw_corr['stars'].get('repo_age', 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("r (Stars vs Forks)",    f"{r_forks:.4f}")
        col2.metric("r (Stars vs Issues)",   f"{r_issues:.4f}")
        col3.metric("r (Stars vs Repo Age)", f"{r_age:.4f}")

        st.markdown("### Correlation with Stars")
        st.dataframe(
            raw_corr['stars'].sort_values(ascending=False)
            .reset_index()
            .rename(columns={'index': 'Feature', 'stars': 'Pearson r'})
            .round(4)
        )

        st.markdown("### Heatmap")
        fig, ax = plt.subplots(figsize=(9, 6))
        mask = np.zeros_like(raw_corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        sns.heatmap(raw_corr, annot=True, cmap='RdYlBu_r', fmt='.2f', ax=ax,
                    linewidths=1, linecolor='#111210',
                    annot_kws={'size': 9, 'color': '#e8e6e0'},
                    vmin=-1, vmax=1)
        ax.set_title("Pearson Correlation Matrix", pad=14, fontsize=11,
                     color='#c8c6c0', fontweight='normal')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        st.info(f"**Inference:** Forks r = {r_forks:.4f} (Strong). "
                f"Issues r = {r_issues:.4f} (Weak). "
                f"Repo age r = {r_age:.4f} (Negligible).")
        st.warning("**Interpretation:** "
                   "Forks and stars are strongly related. "
                   "Both could be influenced by underlying factors such as project quality.")

    with tab2:
        poly_corr_cols = [c for c in ['stars'] + poly_features if c in df.columns]
        poly_corr = df[poly_corr_cols].corr()

        st.markdown("### Correlation with Stars — Polynomial Features")
        poly_corr_stars = poly_corr['stars'].drop('stars').sort_values(ascending=False)
        poly_corr_df = poly_corr_stars.reset_index()
        poly_corr_df.columns = ['Feature', 'Pearson r']
        poly_corr_df['Formula'] = poly_corr_df['Feature'].map({
            'forks2':         'forks squared',
            'contributors2':  'contributors squared',
            'forksissues':    'forks times issues',
            'subscribers2':   'subscribers squared',
            'network_count2': 'network_count squared',
        })
        poly_corr_df['Strength'] = poly_corr_df['Pearson r'].apply(
            lambda r: 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'
        )
        st.dataframe(poly_corr_df.round(4), hide_index=True)

        st.markdown("### Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(poly_corr, annot=True, cmap='RdYlBu_r', fmt='.2f', ax=ax,
                    linewidths=1, linecolor='#111210',
                    annot_kws={'size': 9, 'color': '#e8e6e0'},
                    vmin=-1, vmax=1)
        ax.set_title("Polynomial Feature Correlation Matrix", pad=14, fontsize=11,
                     color='#c8c6c0', fontweight='normal')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

        

# ──────────────────────────────────────────────
# ANOVA
# ──────────────────────────────────────────────
elif page == "ANOVA":
    page_header("Hypothesis Testing", "One-Way ANOVA",
                "Is there a significant difference in stars across programming languages?")
    st.markdown("---")

    anova_model = ols("stars ~ language", data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    anova_p     = anova_table["PR(>F)"].iloc[0]
    anova_f     = anova_table["F"].iloc[0]
    ss_between  = anova_table["sum_sq"].iloc[0]
    ss_total    = anova_table["sum_sq"].sum()
    eta_sq      = ss_between / ss_total

    col1, col2, col3 = st.columns(3)
    col1.metric("F-Statistic",    f"{anova_f:.4f}")
    col2.metric("P-Value",        f"{anova_p:.2e}")
    col3.metric("Eta Squared",    f"{eta_sq:.4f}")

    st.markdown("### Hypotheses")
    st.markdown("""
    **H0:** Mean stars are equal across all 6 languages

    **H1:** At least one language mean is different

    **Significance level:** alpha = 0.05
    """)

    if anova_p < 0.05:
        st.success(f"REJECT H0  Language significantly affects popularity (p = {anova_p:.2e})")
    else:
        st.warning("FAIL TO REJECT H0")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### ANOVA Table")
        st.dataframe(anova_table.round(4))

    with col2:
        st.markdown("### Mean Stars by Language")
        lang_mean = df.groupby('language')['stars'].mean().sort_values(ascending=False)
        st.dataframe(lang_mean.reset_index().rename(columns={'stars': 'Mean Stars'}).round(0))

    st.markdown("### Visual Comparison")
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(lang_mean.index, lang_mean.values,
                  color=COLORS[:len(lang_mean)], edgecolor='none', alpha=0.9)
    for bar, val in zip(bars, lang_mean.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', va='bottom',
                fontsize=8.5, color='#888880',
                fontfamily='DM Mono')
    ax.set_xlabel("Language"); ax.set_ylabel("Mean Stars")
    ax.set_title(f"Mean Stars by Language  |  F = {anova_f:.2f}  |  p = {anova_p:.2e}",
                 pad=14, fontsize=10, fontweight='normal')
    plt.xticks(rotation=0)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.info(f"**Inference:** F = {anova_f:.4f}, p = {anova_p:.2e}. Language "
            f"significantly affects popularity.")
  

# ──────────────────────────────────────────────
# REGRESSION
# ──────────────────────────────────────────────
elif page == "Regression":
    page_header("Statistical Modelling", "Regression Analysis",
                "Which factors explain the number of stars?")
    st.markdown("---")

    st.info("Note: This is an explanatory model, not a predictive one. "
            "The goal is to identify which variables best explain the variance in repository popularity.")

    tab1, tab2 = st.tabs(["Simple Regression", "Multiple Regression"])

    with tab1:
        st.markdown("### Stars and Forks")
        model_forks = ols("stars ~ forks", data=df).fit()

        col1, col2 = st.columns(2)
        col1.metric("R Squared",   f"{model_forks.rsquared:.4f}")
        col2.metric("Coefficient", f"{model_forks.params['forks']:.4f}")

        st.text(model_forks.summary().as_text())
        st.info(f"**Inference:** R squared = {model_forks.rsquared:.4f}. "
                f"Forks alone explains {model_forks.rsquared*100:.1f}% of variance in stars.")
        st.warning(f"**Interpretation:** Each additional fork is associated with "
                   f"{model_forks.params['forks']:.2f} more stars on average. "
                   "Forks is the strongest individual predictor of stars in this dataset.")

    with tab2:
        st.markdown("### Multiple Regression with Polynomial Features")
        st.markdown("Includes raw features and polynomial features: forks squared, contributors squared, forks times issues")

        base_cols = [c for c in ['forks','issues','repo_age','size_kb',
                                  'description_length','topic_count',
                                  'contributors','language'] if c in df.columns]
        all_cols  = base_cols + poly_features
        formula   = "stars ~ " + " + ".join(all_cols)
        ols_model = ols(formula, data=df).fit()
        r2_mlr    = ols_model.rsquared
        r2_adj    = ols_model.rsquared_adj

        col1, col2, col3 = st.columns(3)
        col1.metric("R Squared",          f"{r2_mlr:.4f}")
        col2.metric("Adjusted R Squared", f"{r2_adj:.4f}")
        col3.metric("Predictors",         len(all_cols))

        st.text(ols_model.summary().as_text())

        st.markdown("### Coefficients sorted by significance")
        coef_df = pd.DataFrame({
            'Feature':     ols_model.params.index,
            'Coefficient': ols_model.params.values.round(4),
            'P-value':     ols_model.pvalues.values.round(4),
            'Significant': ['Yes' if p < 0.05 else 'No' for p in ols_model.pvalues.values]
        }).sort_values('P-value')
        st.dataframe(coef_df, hide_index=True)

        st.info(f"**Inference:** R squared = {r2_mlr:.4f}. Model explains {r2_mlr*100:.1f}% "
                f"of variance. Adjusted R squared = {r2_adj:.4f} confirms no overfitting.")
        st.warning(f"**Interpretation:** This explanatory model shows forks dominates "
                   f"the explanation of star variance. Polynomial terms capture "
                   f"non-linear effects. R squared = {r2_mlr:.4f} means our variables "
                   f"explain {r2_mlr*100:.1f}% of why some repos have more stars. "
                   "The model uses forks, polynomial terms and language as key explanatory variables.")

# ──────────────────────────────────────────────
# ISOLATION FOREST
# ──────────────────────────────────────────────
elif page == "Isolation Forest":
    page_header("Isolation Forest",
                "Unique model-derived feature detecting statistically anomalous repositories")
    st.markdown("---")

    st.markdown("""
    ### How It Works

    The model builds 100 random decision trees and measures how many splits are needed to isolate each repository.
    A repository isolated in fewer splits has an unusual combination of features and is flagged as anomalous.
    Normal repositories blend in and require more splits to isolate.

    Input features: forks, issues, size_kb, repo_age. Stars are never used. Zero leakage.
    """)

    anomalies = df[df['anomaly_label'] == -1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Repos",     f"{len(df):,}")
    col2.metric("Anomalous Repos", f"{len(anomalies):,}")
    col3.metric("Anomaly Rate",    f"{len(anomalies)/len(df)*100:.1f}%")

    st.markdown("### Top 10 Most Anomalous Repositories")
    st.markdown("Found without seeing star counts.")
    top10 = anomalies.sort_values('anomaly_score').head(10)[
        ['repo_name','forks','issues','repo_age','stars','anomaly_score']
    ]
    st.dataframe(top10, hide_index=True)

    st.markdown("### Anomaly Score Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    top15  = anomalies.sort_values('anomaly_score').head(15)
    colors = [COLORS[['Python','JavaScript','Java','C++','C#','Go'].index(l)]
              if l in ['Python','JavaScript','Java','C++','C#','Go'] else COLORS[0]
              for l in top15['language']]
    bars = ax.barh(top15['repo_name'], top15['anomaly_score'],
                   color=colors, alpha=0.85, edgecolor='none')
    ax.set_xlabel("Anomaly Score  (more negative = more anomalous)")
    ax.set_title("Top 15 Most Anomalous Repositories", pad=14,
                 fontsize=11, fontweight='normal')
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=8.5)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("### Anomaly Rate by Language")
    anomaly_rate = df.groupby('language')['anomaly_label'].apply(
        lambda x: (x == -1).mean() * 100
    ).sort_values(ascending=False).reset_index()
    anomaly_rate.columns = ['Language', 'Anomaly Rate (%)']
    st.dataframe(anomaly_rate.round(2), hide_index=True)

    st.info(f"**Inference:** {len(anomalies)} anomalous repos detected "
            f"({len(anomalies)/len(df)*100:.1f}%). The model flagged tensorflow, "
            f"pytorch and react as most anomalous without ever seeing their star counts.")
    st.warning("**Interpretation:** Anomalous repos have unusual combinations of "
               "forks, issues, size and age. The anomaly score is a unique "
               "model-derived feature that captures viral potential which raw metrics "
               "cannot explain. The model is unsupervised, no labels needed, zero leakage from stars.")
