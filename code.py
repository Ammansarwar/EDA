import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Streamlit page setup
st.set_page_config(page_title="📊 Automated EDA Dashboard", layout="wide")

st.title("📊 Automated Exploratory Data Analysis (EDA)")
st.write("Upload a CSV file to automatically generate insights, trends, and visualizations.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.header("🔎 Dataset Overview")
    st.write("Here’s the preview of your data:")
    st.dataframe(df.head())

    # Dataset info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.subheader("📋 Data Info")
    st.text(info_str)

    # Shape of dataset
    st.subheader("📐 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Missing values
    st.subheader("❓ Missing Values")
    st.write(df.isnull().sum())

    # Descriptive stats
    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe(include="all"))

    # Correlation heatmap (numerical only)
    st.subheader("🔥 Correlation Heatmap")
    num_cols = df.select_dtypes(include="number")
    if not num_cols.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)
    else:
        st.write("No numeric columns for correlation heatmap.")

    # Top 10 products by order count (if Product column exists)
    if "Product" in df.columns:
        st.header("🏆 Top 10 Products in Demand")

        top_products = df["Product"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_products.values, y=top_products.index, palette="viridis", ax=ax)
        ax.set_title("Top 10 Products by Order Count", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Orders")
        ax.set_ylabel("Product")
        st.pyplot(fig)

        # Trends over time (if OrderDate exists)
        if "OrderDate" in df.columns:
            st.subheader("📈 Demand Trends Over Time")
            df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
            trend = df.groupby([df["OrderDate"].dt.to_period("M"), "Product"]).size().unstack().fillna(0)
            top_trend = trend[top_products.index]

            fig, ax = plt.subplots(figsize=(12, 6))
            for product in top_trend.columns:
                ax.plot(top_trend.index.astype(str), top_trend[product], label=product)

            ax.set_title("Monthly Demand Trends of Top 10 Products", fontsize=14, fontweight="bold")
            ax.set_xlabel("Month")
            ax.set_ylabel("Order Count")
            ax.legend(title="Products")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Price distribution (if Price column exists)
    if "Price" in df.columns:
        st.header("💰 Price Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Price"], kde=True, bins=30, color="skyblue", ax=ax)
        ax.set_title("Price Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Price")
        st.pyplot(fig)

    # Group-wise statistics (if categorical + numerical exist)
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0 and len(num_cols.columns) > 0:
        st.header("📊 Group-wise Mean & Std")

        selected_cat = st.selectbox("Choose a categorical column:", cat_cols)
        selected_num = st.selectbox("Choose a numerical column:", num_cols.columns)

        group_stats = df.groupby(selected_cat)[selected_num].agg(["mean", "std"]).reset_index()
        st.write(group_stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=selected_cat, y="mean", data=group_stats, palette="magma", ax=ax)
        ax.set_title(f"Mean {selected_num} by {selected_cat}", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"Mean of {selected_num}")
        ax.set_xlabel(selected_cat)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.success("✅ EDA Completed! Check insights and visualizations above.")
