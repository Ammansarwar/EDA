import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Comprehensive EDA Dashboard", layout="wide")
st.title("📊 Automated EDA Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📌 Raw Data Preview")
    st.dataframe(df.head())

    # ----------------- Data Cleaning -----------------
    st.subheader("🧹 Data Cleaning & Info")

    st.write("**Shape of dataset:**", df.shape)

    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

    st.write("**Missing values per column:**")
    st.write(df.isnull().sum())

    st.write("**Duplicate rows:**", df.duplicated().sum())

    # Drop duplicates for analysis
    df = df.drop_duplicates()

    # ----------------- Descriptive Statistics -----------------
    st.subheader("📈 Descriptive Statistics")
    st.write(df.describe(include="all").T)

    # Standard deviation summary
    st.write("**Standard Deviations of numeric columns:**")
    st.write(df.std(numeric_only=True))

    # ----------------- Visualizations -----------------
    st.subheader("📊 General Visualizations")

    # Missing value heatmap
    st.write("🔍 Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # Histograms for numeric columns
    st.write("📉 Histograms")
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

    # Boxplots
    st.write("📦 Boxplots (Outlier Detection)")
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        plt.title(f"Boxplot of {col}")
        st.pyplot(fig)

    # Correlation heatmap
    if len(num_cols) > 1:
        st.write("🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ----------------- Group-wise Analysis -----------------
    st.subheader("📂 Group-wise Mean & Std")

    if "Product" in df.columns:
        group_stats = df.groupby("Product").agg(["mean", "std"])
        st.write(group_stats)

    # ----------------- Top 10 Products Analysis -----------------
    if "Product" in df.columns and "OrderQuantity" in df.columns:
        st.subheader("🏆 Top 10 Most In-Demand Products")
        top_products = df.groupby("Product")["OrderQuantity"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)

        if "OrderDate" in df.columns:
            st.subheader("📈 Trends of Top 10 Products")
            df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
            top_df = df[df["Product"].isin(top_products.index)]
            trend = top_df.groupby(["OrderDate", "Product"])["OrderQuantity"].sum().reset_index()

            fig, ax = plt.subplots(figsize=(12,6))
            sns.lineplot(data=trend, x="OrderDate", y="OrderQuantity", hue="Product", ax=ax)
            plt.title("Trends of Top 10 Products")
            st.pyplot(fig)

        if "Price" in df.columns:
            st.subheader("💰 Price Distribution vs Order Counts")
            price_order = top_df.groupby("Product").agg(
                avg_price=("Price", "mean"),
                total_orders=("OrderQuantity", "sum")
            ).reset_index()

            fig, ax1 = plt.subplots(figsize=(10,6))
            sns.barplot(data=price_order, x="Product", y="avg_price", color="skyblue", ax=ax1)
            ax2 = ax1.twinx()
            sns.lineplot(data=price_order, x="Product", y="total_orders", marker="o", color="red", ax=ax2)
            ax1.set_ylabel("Average Price")
            ax2.set_ylabel("Total Orders")
            plt.title("Price vs Order Count for Top 10 Products")
            st.pyplot(fig)

        # ----------------- Conclusion -----------------
        st.subheader("📝 Conclusion")
        most_demanded = top_products.index[0]
        st.write(f"- The most demanded product is **{most_demanded}** with {top_products.iloc[0]} orders.")
        st.write("- Products with high order counts don’t always have the highest price.")
        st.write("- Seasonal or time trends can be observed if `OrderDate` is available.")
