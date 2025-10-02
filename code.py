import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Meaningful EDA", layout="wide")

st.title("🛒 Interactive EDA Dashboard")
st.write("Upload your dataset and explore meaningful insights with cool-colored interactive graphs.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("🔍 Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Columns:", list(df.columns))

    # ---------- Numerical Column Analysis ----------
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if num_cols:
        st.subheader("💰 Price / Numeric Distributions")
        for col in num_cols[:3]:  # Show first 3 numeric cols meaningfully
            fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#6EC5E9"])
            fig.update_layout(title=f"Distribution of {col}", bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"ℹ️ This graph shows how **{col}** values are distributed in the dataset. "
                    "It helps identify common ranges, outliers, or skewed data.")

    # ---------- Top Categories ----------
    if cat_cols:
        st.subheader("🏆 Top 10 Categories (Most Frequent)")
        for col in cat_cols[:2]:  # Show top 2 categorical cols
            top10 = df[col].value_counts().nlargest(10)
            fig = px.bar(top10, x=top10.index, y=top10.values,
                         color=top10.values, color_continuous_scale="Blues")
            fig.update_layout(title=f"Top 10 {col}", xaxis_title=col, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"ℹ️ This graph shows the **Top 10 most common values in {col}**. "
                    "It highlights which categories dominate your dataset.")

    # ---------- Revenue Analysis ----------
    if set(["Quantity", "Price"]).issubset(df.columns):
        st.subheader("💸 Revenue Analysis")
        df["Revenue"] = df["Quantity"] * df["Price"]
        top_products = df.groupby("Product")["Revenue"].sum().nlargest(10)
        fig = px.bar(top_products, x=top_products.index, y=top_products.values,
                     color=top_products.values, color_continuous_scale="Tealgrn")
        fig.update_layout(title="Top 10 Products by Revenue", xaxis_title="Product", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This chart highlights which products generated the most revenue. "
                "It helps identify best-selling items driving business growth.")

    # ---------- Correlation Heatmap ----------
    if len(num_cols) > 1:
        st.subheader("📊 Correlation Heatmap")
        corr = df[num_cols].corr()
        plt.figure(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
        st.info("ℹ️ The heatmap shows correlations between numeric columns. "
                "A high correlation means the variables move together, useful for feature selection.")

    # ---------- Missing Values ----------
    st.subheader("❌ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig = px.bar(missing, x=missing.index, y=missing.values,
                     color=missing.values, color_continuous_scale="Purples")
        fig.update_layout(title="Missing Values per Column", xaxis_title="Column", yaxis_title="Missing Count")
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This chart highlights columns with missing values. "
                "Understanding missing data is crucial before modeling.")
    else:
        st.success("✅ No missing values found in this dataset!")

else:
    st.warning("📂 Please upload a CSV file to begin EDA.")
