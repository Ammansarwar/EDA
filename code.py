import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Streamlit page setup
st.set_page_config(page_title="📊 Pro EDA Dashboard", layout="wide")
sns.set_style("whitegrid")

st.title("📊 Automated EDA for Your Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Overview
    st.header("🔎 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("❓ Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe(include="all"))

    # Correlation
    num_cols = df.select_dtypes(include="number")
    if not num_cols.empty:
        st.header("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Top 10 products
    if "Product" in df.columns:
        st.header("🏆 Top 10 Products by Demand")
        top_products = df["Product"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(y=top_products.index, x=top_products.values, palette="viridis", ax=ax)
        ax.set_title("Top 10 Products", fontsize=14)
        ax.set_xlabel("Order Count")
        ax.set_ylabel("Product")
        st.pyplot(fig)

        # Trends
        if "OrderDate" in df.columns:
            st.subheader("📈 Demand Trends Over Time")
            df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
            trend = df.groupby([df["OrderDate"].dt.to_period("M"), "Product"]).size().unstack().fillna(0)
            trend = trend[top_products.index]

            fig, ax = plt.subplots(figsize=(12, 6))
            for product in trend.columns:
                ax.plot(trend.index.astype(str), trend[product], label=product)
            ax.set_title("Monthly Trends of Top 10 Products", fontsize=14)
            ax.set_xlabel("Month")
            ax.set_ylabel("Orders")
            ax.legend(title="Products", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Price distribution
    if "Price" in df.columns:
        st.header("💰 Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["Price"], bins=30, kde=True, color="skyblue", ax=ax)
        ax.set_title("Price Distribution", fontsize=14)
        st.pyplot(fig)

    # Group-wise
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0 and len(num_cols.columns) > 0:
        st.header("📊 Group-wise Mean & Std")
        selected_cat = st.selectbox("Choose categorical column:", cat_cols)
        selected_num = st.selectbox("Choose numerical column:", num_cols.columns)

        group_stats = df.groupby(selected_cat)[selected_num].agg(["mean", "std"]).reset_index()
        st.write(group_stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=group_stats, x=selected_cat, y="mean", palette="magma", ax=ax)
        ax.set_title(f"Mean {selected_num} by {selected_cat}", fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Conclusion
    st.header("📝 Conclusion")
    conclusions = []
    if "Product" in df.columns:
        top_item = df["Product"].value_counts().idxmax()
        conclusions.append(f"➡️ **{top_item}** is the most demanded product.")
    if "Price" in df.columns:
        mean_price = df["Price"].mean()
        conclusions.append(f"➡️ Average price is around **{mean_price:.2f}**.")
    if not conclusions:
        conclusions.append("➡️ Dataset explored successfully, but no specific insights could be highlighted.")

    for c in conclusions:
        st.write(c)

    st.success("✅ EDA Completed!")
