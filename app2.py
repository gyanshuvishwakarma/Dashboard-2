"""
Advanced Business Analytics Dashboard
--------------------------------------
A KPI-first, visualization-heavy Streamlit dashboard built for sales / business
CSV data (works generically on any tabular CSV, with extra intelligence for
common business columns like Sales, Profit, Discount, Category, Region, Date).

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ======================================================================
# PAGE CONFIG & STYLE
# ======================================================================
st.set_page_config(page_title="Advanced Business Dashboard", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(28,131,225,0.10), rgba(28,131,225,0.03));
        border: 1px solid rgba(28,131,225,0.25);
        padding: 14px 16px;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; }
    .block-container { padding-top: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Advanced Business Analytics Dashboard")
st.caption("Upload a CSV to get executive KPIs, trend analysis, category performance, "
           "profitability breakdowns, correlation analysis, ML-driven clustering & anomaly "
           "detection, and AI-generated insights.")

# ======================================================================
# FILE UPLOAD
# ======================================================================
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if not file:
    st.info("👆 Upload a CSV file to begin.")
    st.caption(
        "Works best with business/sales data containing columns like Sales, Profit, "
        "Quantity, Discount, Category, Region, and an Order/Date column — "
        "but adapts automatically to any tabular CSV."
    )
    st.stop()

# ======================================================================
# LOAD & CLEAN DATA
# ======================================================================
try:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    # Strip whitespace from all string/object columns (common in messy exports)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    st.success(f"✅ Loaded `{file.name}` — {df.shape[0]:,} rows × {df.shape[1]} columns")

    if df.empty:
        st.warning("⚠️ The uploaded file is empty.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error reading file: {e}")
    st.stop()

# ======================================================================
# SMART COLUMN DETECTION
# ======================================================================
num_cols = df.select_dtypes(include="number").columns.tolist()
obj_cols = df.select_dtypes(include="object").columns.tolist()


def find_col(keywords, candidates, exclude=None):
    exclude = exclude or []
    for kw in keywords:
        for c in candidates:
            if c in exclude:
                continue
            if kw == c.lower().replace("_", "").replace(" ", ""):
                return c
    for kw in keywords:
        for c in candidates:
            if c in exclude:
                continue
            if kw in c.lower():
                return c
    return None


date_cols = []
for col in obj_cols:
    if any(k in col.lower() for k in ["date", "time"]):
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0.5 * len(df):
            date_cols.append(col)

sales_col = find_col(["sales", "revenue", "amount", "totalsales"], num_cols)
profit_col = find_col(["profit"], num_cols)
quantity_col = find_col(["quantity", "qty", "units"], num_cols)
discount_col = find_col(["discount"], num_cols)
margin_col = find_col(["margin"], num_cols)
primary_date_col = find_col(["orderdate", "date"], date_cols) or (date_cols[0] if date_cols else None)

category_col = find_col(["category"], obj_cols, exclude=["Sub_Category", "Sales_Category"])
subcategory_col = find_col(["subcategory"], obj_cols)
region_col = find_col(["region"], obj_cols)
state_col = find_col(["state", "country", "city"], obj_cols)
product_col = find_col(["productname", "product"], obj_cols)

cat_cols_all = [c for c in obj_cols if c not in date_cols]
remaining_num = [c for c in num_cols if c not in [sales_col, profit_col, quantity_col, discount_col, margin_col]]

primary_metric = sales_col or (num_cols[0] if num_cols else None)

# ======================================================================
# SIDEBAR FILTERS
# ======================================================================
st.sidebar.header("🎯 Filters")
filtered_df = df.copy()

if primary_date_col:
    d = pd.to_datetime(df[primary_date_col], errors="coerce")
    valid_dates = d.dropna()
    if not valid_dates.empty:
        min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
        if min_d < max_d:
            date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                parsed_dates = pd.to_datetime(filtered_df[primary_date_col], errors="coerce")
                mask = (parsed_dates.dt.date >= date_range[0]) & (parsed_dates.dt.date <= date_range[1])
                filtered_df = filtered_df[mask | parsed_dates.isna()]

for col in [c for c in [category_col, region_col, subcategory_col, state_col] if c]:
    unique_vals = df[col].dropna().unique()
    if 1 < len(unique_vals) < 60:
        selected = st.sidebar.multiselect(col, sorted(unique_vals), default=sorted(unique_vals))
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

other_cat_filters = [c for c in cat_cols_all if c not in [category_col, region_col, subcategory_col, state_col, product_col]]
with st.sidebar.expander("More filters"):
    for col in other_cat_filters:
        unique_vals = df[col].dropna().unique()
        if 1 < len(unique_vals) < 40:
            selected = st.multiselect(col, sorted(unique_vals), default=sorted(unique_vals), key=f"f_{col}")
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

if filtered_df.empty:
    st.warning("⚠️ No rows match the current filters. Adjust filters in the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** rows")

missing_total = int(df.isnull().sum().sum())
dup_count = int(df.duplicated().sum())

# ======================================================================
# TABS
# ======================================================================
tab_kpi, tab_trend, tab_perf, tab_profit, tab_explore, tab_ml, tab_insights, tab_data = st.tabs(
    ["🎯 Executive KPIs", "📈 Trends", "🏆 Category & Region", "💰 Profitability",
     "🔎 Visual Explorer", "🧠 ML Analysis", "🤖 AI Insights", "🧾 Raw Data"]
)

# ======================================================================
# TAB 1 — EXECUTIVE KPIs
# ======================================================================
with tab_kpi:
    st.subheader("🎯 Executive Summary")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    total_sales = filtered_df[sales_col].sum() if sales_col else None
    total_profit = filtered_df[profit_col].sum() if profit_col else None
    total_orders = len(filtered_df)
    avg_discount = filtered_df[discount_col].mean() if discount_col else None
    total_qty = filtered_df[quantity_col].sum() if quantity_col else None
    profit_margin = (total_profit / total_sales * 100) if (total_sales and total_profit is not None and total_sales != 0) else None

    if sales_col:
        k1.metric("💵 Total Sales", f"${total_sales:,.0f}")
    else:
        k1.metric("Total Rows", f"{total_orders:,}")

    if profit_col:
        k2.metric("📈 Total Profit", f"${total_profit:,.0f}")
    if profit_margin is not None:
        k3.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
    k4.metric("🧾 Total Orders", f"{total_orders:,}")
    if quantity_col:
        k5.metric("📦 Units Sold", f"{total_qty:,.0f}")
    if discount_col:
        k6.metric("🏷️ Avg Discount", f"{avg_discount*100:.1f}%" if avg_discount <= 1 else f"{avg_discount:.1f}%")

    st.markdown("#### Additional metrics")
    extra_cols = st.columns(4)
    if sales_col:
        aov = filtered_df[sales_col].mean()
        extra_cols[0].metric("Avg Order Value", f"${aov:,.2f}")
    if category_col:
        top_cat = filtered_df.groupby(category_col)[sales_col].sum().idxmax() if sales_col else filtered_df[category_col].value_counts().idxmax()
        extra_cols[1].metric("Top Category", top_cat)
    if region_col:
        top_region = filtered_df.groupby(region_col)[sales_col].sum().idxmax() if sales_col else filtered_df[region_col].value_counts().idxmax()
        extra_cols[2].metric("Top Region", top_region)
    extra_cols[3].metric("Data Quality", f"{100 - (missing_total / (df.size or 1) * 100):.1f}% complete")

    st.markdown("---")
    st.markdown("#### 📐 Detailed statistics")
    if num_cols:
        stats_df = filtered_df[num_cols].agg(["count", "mean", "median", "std", "min", "max"]).T
        stats_df.columns = ["Count", "Mean", "Median", "Std Dev", "Min", "Max"]
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

    if missing_total > 0:
        with st.expander("🕳️ Missing values by column"):
            miss_series = df.isnull().sum()
            miss_series = miss_series[miss_series > 0].sort_values(ascending=False)
            fig_missing = px.bar(x=miss_series.values, y=miss_series.index, orientation="h",
                                  labels={"x": "Missing Count", "y": "Column"},
                                  color=miss_series.values, color_continuous_scale="Oranges")
            st.plotly_chart(fig_missing, use_container_width=True)

# ======================================================================
# TAB 2 — TRENDS
# ======================================================================
with tab_trend:
    st.subheader("📈 Trend Analysis")

    if not primary_date_col:
        st.info("No date column detected — trend analysis needs a date/order-date column.")
    else:
        trend_df = filtered_df.copy()
        trend_df[primary_date_col] = pd.to_datetime(trend_df[primary_date_col], errors="coerce")
        trend_df = trend_df.dropna(subset=[primary_date_col]).sort_values(primary_date_col)

        if trend_df.empty:
            st.info("No valid dates found after filtering.")
        else:
            metric_choices = [c for c in [sales_col, profit_col, quantity_col] if c] or num_cols
            tcol1, tcol2 = st.columns([2, 1])
            metric = tcol1.selectbox("Metric", metric_choices)
            freq = tcol2.selectbox("Granularity", ["Daily", "Weekly", "Monthly", "Quarterly"], index=2)
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}

            agg = trend_df.groupby(pd.Grouper(key=primary_date_col, freq=freq_map[freq]))[metric].sum().reset_index()
            fig_trend = px.area(agg, x=primary_date_col, y=metric, title=f"{metric} Over Time ({freq})",
                                 color_discrete_sequence=["#1C83E1"])
            fig_trend.update_traces(line=dict(width=2))
            st.plotly_chart(fig_trend, use_container_width=True)

            # Trend split by category/region if available
            split_col = category_col or region_col
            if split_col:
                agg2 = trend_df.groupby([pd.Grouper(key=primary_date_col, freq=freq_map[freq]), split_col])[metric].sum().reset_index()
                fig_split = px.line(agg2, x=primary_date_col, y=metric, color=split_col, markers=True,
                                     title=f"{metric} Over Time by {split_col}")
                st.plotly_chart(fig_split, use_container_width=True)

            # Period-over-period % change
            if len(agg) >= 2:
                agg["pct_change"] = agg[metric].pct_change() * 100
                agg["pct_change"] = agg["pct_change"].replace([np.inf, -np.inf], np.nan)
                last_change = agg["pct_change"].iloc[-1]
                if pd.notna(last_change):
                    st.metric(f"Latest {freq} change", f"{last_change:+.1f}%")
                fig_change = px.bar(agg.dropna(subset=["pct_change"]), x=primary_date_col, y="pct_change",
                                     title=f"Period-over-Period % Change in {metric}",
                                     color="pct_change", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig_change, use_container_width=True)

# ======================================================================
# TAB 3 — CATEGORY & REGION PERFORMANCE
# ======================================================================
with tab_perf:
    st.subheader("🏆 Category & Region Performance")

    metric = st.selectbox("Metric to analyze", [c for c in [sales_col, profit_col, quantity_col] if c] or num_cols, key="perf_metric")

    perf_cols = st.columns(2)
    if category_col:
        cat_agg = filtered_df.groupby(category_col)[metric].sum().sort_values(ascending=False).reset_index()
        fig_cat = px.bar(cat_agg, x=category_col, y=metric, title=f"{metric} by {category_col}",
                          color=metric, color_continuous_scale="Blues", text_auto=".2s")
        perf_cols[0].plotly_chart(fig_cat, use_container_width=True)

    if region_col:
        reg_agg = filtered_df.groupby(region_col)[metric].sum().sort_values(ascending=False).reset_index()
        fig_reg = px.bar(reg_agg, x=region_col, y=metric, title=f"{metric} by {region_col}",
                          color=metric, color_continuous_scale="Teal", text_auto=".2s")
        perf_cols[1].plotly_chart(fig_reg, use_container_width=True)

    if category_col and subcategory_col:
        st.markdown("#### 🌳 Category → Sub-Category Breakdown")
        tree_df = filtered_df.groupby([category_col, subcategory_col])[metric].sum().reset_index()
        tree_df = tree_df[tree_df[metric] > 0]
        fig_tree = px.treemap(tree_df, path=[category_col, subcategory_col], values=metric,
                               color=metric, color_continuous_scale="Sunsetdark",
                               title=f"{metric} — Category → Sub-Category")
        st.plotly_chart(fig_tree, use_container_width=True)

    if category_col and region_col:
        st.markdown("#### 🔥 Category × Region Heatmap")
        pivot = filtered_df.pivot_table(index=category_col, columns=region_col, values=metric, aggfunc="sum", fill_value=0)
        fig_heat = px.imshow(pivot, text_auto=".2s", color_continuous_scale="Viridis", aspect="auto",
                              title=f"{metric}: {category_col} × {region_col}")
        st.plotly_chart(fig_heat, use_container_width=True)

    if product_col:
        st.markdown("#### 🥇 Top 10 Products")
        top_products = filtered_df.groupby(product_col)[metric].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top = px.bar(top_products, x=metric, y=product_col, orientation="h",
                          color=metric, color_continuous_scale="Purples",
                          title=f"Top 10 Products by {metric}")
        fig_top.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_top, use_container_width=True)

    if not any([category_col, region_col, product_col]):
        st.info("No category/region/product-like column detected in this dataset.")

# ======================================================================
# TAB 4 — PROFITABILITY
# ======================================================================
with tab_profit:
    st.subheader("💰 Profitability Analysis")

    if not profit_col:
        st.info("No profit column detected — this section needs a numeric 'Profit' column.")
    else:
        pc1, pc2 = st.columns(2)

        if discount_col:
            fig_scatter = px.scatter(filtered_df, x=discount_col, y=profit_col,
                                      color=category_col if category_col else None,
                                      size=sales_col if sales_col else None,
                                      title="Discount vs Profit", opacity=0.75,
                                      trendline="ols" if len(filtered_df) > 5 else None)
            pc1.plotly_chart(fig_scatter, use_container_width=True)

        fig_profit_dist = px.histogram(filtered_df, x=profit_col, nbins=25, marginal="box",
                                        title="Profit Distribution", color_discrete_sequence=["#2E7D32"])
        pc2.plotly_chart(fig_profit_dist, use_container_width=True)

        if margin_col:
            fig_margin = px.histogram(filtered_df, x=margin_col, nbins=25, marginal="violin",
                                       title="Profit Margin % Distribution", color_discrete_sequence=["#00897B"])
            st.plotly_chart(fig_margin, use_container_width=True)

        if category_col:
            st.markdown("#### 📦 Profit by Category (Box Plot)")
            fig_box = px.box(filtered_df, x=category_col, y=profit_col, color=category_col,
                              title="Profit Spread by Category")
            st.plotly_chart(fig_box, use_container_width=True)

        # High discount impact
        if discount_col:
            st.markdown("#### 🏷️ Discount Impact")
            disc_series = filtered_df[discount_col]
            threshold = disc_series.median()
            high_disc = filtered_df[disc_series > threshold]
            low_disc = filtered_df[disc_series <= threshold]
            impact_cols = st.columns(2)
            impact_cols[0].metric(f"Avg Profit — Discount > {threshold:.2f}", f"${high_disc[profit_col].mean():,.2f}")
            impact_cols[1].metric(f"Avg Profit — Discount ≤ {threshold:.2f}", f"${low_disc[profit_col].mean():,.2f}")

# ======================================================================
# TAB 5 — VISUAL EXPLORER
# ======================================================================
with tab_explore:
    st.subheader("🔎 Custom Chart Builder")

    cc1, cc2, cc3 = st.columns(3)
    chart_type = cc1.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box", "Histogram", "Pie"])
    x_col = cc2.selectbox("X-axis", df.columns)
    y_col = cc3.selectbox("Y-axis", num_cols) if num_cols and chart_type not in ["Histogram", "Pie"] else None
    color_by = st.selectbox("Color by (optional)", ["(none)"] + cat_cols_all)
    color_arg = color_by if color_by != "(none)" else None

    try:
        if chart_type == "Bar":
            fig = px.bar(filtered_df, x=x_col, y=y_col, color=color_arg, barmode="group")
        elif chart_type == "Line":
            plot_df = filtered_df.sort_values(x_col) if x_col in filtered_df.columns else filtered_df
            fig = px.line(plot_df, x=x_col, y=y_col, color=color_arg)
        elif chart_type == "Scatter":
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_arg)
        elif chart_type == "Box":
            fig = px.box(filtered_df, x=x_col if x_col in cat_cols_all else None, y=y_col, color=color_arg)
        elif chart_type == "Pie":
            pie_val = y_col or (num_cols[0] if num_cols else None)
            pie_data = filtered_df.groupby(x_col)[pie_val].sum().reset_index() if pie_val else filtered_df[x_col].value_counts().reset_index()
            fig = px.pie(pie_data, names=x_col, values=pie_val if pie_val else "count", hole=0.4,
                         title=f"Share by {x_col}")
        else:
            fig = px.histogram(filtered_df, x=x_col, color=color_arg, marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Chart error: {e}")

    st.markdown("---")

    if len(num_cols) >= 2:
        st.markdown("#### 🔗 Correlation Heatmap")
        corr = filtered_df[num_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              aspect="auto", title="Correlation Between Numeric Columns")
        st.plotly_chart(fig_corr, use_container_width=True)

    if num_cols:
        st.markdown("#### 📊 Distribution Overview")
        show_cols = num_cols[:6]
        n = len(show_cols)
        rows = (n + 2) // 3
        fig_dist = make_subplots(rows=rows, cols=min(3, n), subplot_titles=show_cols)
        for i, col in enumerate(show_cols):
            r, c = i // 3 + 1, i % 3 + 1
            fig_dist.add_trace(go.Histogram(x=filtered_df[col], name=col, showlegend=False), row=r, col=c)
        fig_dist.update_layout(height=280 * rows, title_text="Distributions of Numeric Columns")
        st.plotly_chart(fig_dist, use_container_width=True)

# ======================================================================
# TAB 6 — ML ANALYSIS
# ======================================================================
with tab_ml:
    st.subheader("🧠 Machine Learning Analysis")

    ml_ready_cols = [c for c in num_cols if filtered_df[c].notna().sum() > 5]

    if len(ml_ready_cols) < 2:
        st.info("Need at least 2 numeric columns with sufficient data to run clustering / anomaly detection.")
    else:
        ml_df = filtered_df[ml_ready_cols].dropna()

        if len(ml_df) < 8:
            st.info("Not enough complete rows (need 8+) to run ML analysis.")
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(ml_df)

            st.markdown("#### 🧩 K-Means Clustering")
            k = st.slider("Number of clusters (k)", 2, min(8, len(ml_df) - 1), 3)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            coords = None
            if X.shape[1] >= 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X)
                plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=ml_df.index)
                plot_df["Cluster"] = cluster_labels.astype(str)
                explained = pca.explained_variance_ratio_.sum() * 100
                fig_cluster = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                                          title=f"Cluster Map (PCA, {explained:.0f}% variance explained)", opacity=0.8)
                st.plotly_chart(fig_cluster, use_container_width=True)

            cluster_summary = ml_df.copy()
            cluster_summary["Cluster"] = cluster_labels
            st.markdown("**Cluster profiles (mean values):**")
            st.dataframe(cluster_summary.groupby("Cluster")[ml_ready_cols].mean().style.format("{:.2f}"),
                         use_container_width=True)
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            st.caption("Cluster sizes: " + ", ".join(f"Cluster {i}: {n} rows" for i, n in cluster_sizes.items()))

            st.markdown("---")
            st.markdown("#### 🚨 Anomaly Detection (Isolation Forest)")
            contamination = st.slider("Expected anomaly rate", 0.01, 0.25, 0.05, 0.01)
            iso = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso.fit_predict(X)
            n_anomalies = int((anomaly_labels == -1).sum())
            st.metric("Anomalies Detected", f"{n_anomalies:,}", f"{n_anomalies / len(ml_df) * 100:.1f}% of rows")

            if coords is not None:
                anomaly_plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=ml_df.index)
                anomaly_plot_df["Status"] = np.where(anomaly_labels == -1, "Anomaly", "Normal")
                fig_anom = px.scatter(anomaly_plot_df, x="PC1", y="PC2", color="Status",
                                       color_discrete_map={"Anomaly": "#D9534F", "Normal": "#4FA3D1"},
                                       title="Anomaly Map (PCA projection)", opacity=0.8)
                st.plotly_chart(fig_anom, use_container_width=True)

            if n_anomalies > 0:
                with st.expander(f"🔍 View {n_anomalies} anomalous rows"):
                    st.dataframe(filtered_df.loc[ml_df.index[anomaly_labels == -1]], use_container_width=True)

# ======================================================================
# TAB 7 — AI INSIGHTS
# ======================================================================
with tab_insights:
    st.subheader("🤖 AI-Generated Insights")
    insights = []

    try:
        if sales_col and profit_col:
            margin = filtered_df[profit_col].sum() / filtered_df[sales_col].sum() * 100 if filtered_df[sales_col].sum() else 0
            insights.append(f"💰 **Overall profit margin:** {margin:.1f}% (${filtered_df[profit_col].sum():,.0f} profit on ${filtered_df[sales_col].sum():,.0f} sales)")

        if num_cols:
            means = filtered_df[num_cols].mean()
            best, worst = means.idxmax(), means.idxmin()
            insights.append(f"📈 **Highest average metric:** `{best}` ({means[best]:,.2f})")
            insights.append(f"📉 **Lowest average metric:** `{worst}` ({means[worst]:,.2f})")

            stds = filtered_df[num_cols].std()
            if not stds.empty:
                most_var = stds.idxmax()
                insights.append(f"🌪️ **Most variable metric:** `{most_var}` (std dev {stds[most_var]:,.2f})")

            outlier_notes = []
            for col in num_cols:
                q1, q3 = filtered_df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_out = ((filtered_df[col] < lower) | (filtered_df[col] > upper)).sum()
                    if n_out > 0:
                        outlier_notes.append(f"`{col}` has {n_out} potential outlier(s)")
            if outlier_notes:
                insights.append("🚨 **Outliers detected:** " + "; ".join(outlier_notes[:5]))

            if len(num_cols) >= 2:
                corr = filtered_df[num_cols].corr(numeric_only=True).abs()
                corr_arr = np.array(corr.values, copy=True)
                np.fill_diagonal(corr_arr, 0)
                corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
                if corr.max().max() >= 0.7:
                    max_pair = corr.stack().idxmax()
                    strength = corr.loc[max_pair]
                    insights.append(f"🔗 **Strong relationship:** `{max_pair[0]}` and `{max_pair[1]}` are highly correlated (r={strength:.2f})")

        if primary_date_col and (sales_col or num_cols):
            try:
                ycol = sales_col or num_cols[0]
                tdf = filtered_df[[primary_date_col, ycol]].copy()
                tdf[primary_date_col] = pd.to_datetime(tdf[primary_date_col], errors="coerce")
                tdf = tdf.dropna().sort_values(primary_date_col)
                if len(tdf) >= 5:
                    x_numeric = (tdf[primary_date_col] - tdf[primary_date_col].min()).dt.days.values.astype(float)
                    y_vals = tdf[ycol].values.astype(float)
                    slope = np.polyfit(x_numeric, y_vals, 1)[0]
                    direction = "increasing 📈" if slope > 0 else "decreasing 📉" if slope < 0 else "flat ➡️"
                    insights.append(f"⏱️ **Trend:** `{ycol}` over `{primary_date_col}` is {direction} on average over the period")
            except Exception:
                pass

        if category_col:
            vc = filtered_df.groupby(category_col)[sales_col].sum() if sales_col else filtered_df[category_col].value_counts()
            top_val = vc.idxmax()
            top_pct = vc.max() / vc.sum() * 100
            insights.append(f"🏆 **Top `{category_col}`:** {top_val} ({top_pct:.1f}% of {'sales' if sales_col else 'rows'})")
            if top_pct > 60:
                insights.append(f"⚖️ **Concentration risk:** `{category_col}` is dominated by {top_val} ({top_pct:.1f}%)")

        if discount_col and profit_col:
            corr_dp = filtered_df[[discount_col, profit_col]].corr().iloc[0, 1]
            if pd.notna(corr_dp) and abs(corr_dp) >= 0.3:
                relation = "reduces" if corr_dp < 0 else "increases"
                insights.append(f"🏷️ **Discount effect:** higher `{discount_col}` tends to {relation} `{profit_col}` (r={corr_dp:.2f})")

        if missing_total > 0:
            worst_missing_col = df.isnull().sum().idxmax()
            worst_missing_pct = df[worst_missing_col].isnull().mean() * 100
            insights.append(f"🕳️ **Data quality:** `{worst_missing_col}` has the most missing values ({worst_missing_pct:.1f}%)")

        if dup_count > 0:
            insights.append(f"♻️ **{dup_count} duplicate row(s)** detected in the dataset")

        if insights:
            for ins in insights:
                st.markdown(f"- {ins}")
            st.caption("👉 See the **ML Analysis** tab for clustering and anomaly detection.")
        else:
            st.info("Not enough data to generate insights.")

    except Exception as e:
        st.warning(f"⚠️ Could not generate insights: {e}")

# ======================================================================
# TAB 8 — RAW DATA
# ======================================================================
with tab_data:
    st.subheader("🧾 Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("📥 Download Filtered Data as CSV", filtered_df.to_csv(index=False),
                        file_name="filtered_data.csv", mime="text/csv")