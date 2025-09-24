import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ----------------- Streamlit Setup -----------------
st.set_page_config(page_title="Stylish EDA Dashboard", layout="wide")
st.title("🌈 Automated Exploratory Data Analysis (EDA)")

# Use nice seaborn theme
sns.set_theme(style="whitegrid", palette="Set2")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📂 Upload a dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # ----------------- Preview -----------------
    st.subheader("👀 Dataset Preview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))

    # ----------------- Info -----------------
    st.subheader("📑 Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("**Missing values per column:**")
    st.write(df.isnull().sum())

    st.write("**Duplicate rows:**", df.duplicated().sum())

    # ----------------- Descriptive Stats -----------------
    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe(include="all").T)

    # ----------------- Univariate Analysis -----------------
    st.subheader("🎯 Univariate Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        st.markdown("### 🔢 Numeric Features")
        for col in num_cols:
            c1, c2 = st.columns(2)

            with c1:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.histplot(df[col], kde=True, color="teal", ax=ax)
                ax.set_title(f"Distribution of {col}", fontsize=12, color="darkblue")
                st.pyplot(fig)

            with c2:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.boxplot(x=df[col], color="orange", ax=ax)
                ax.set_title(f"Boxplot of {col}", fontsize=12, color="darkred")
                st.pyplot(fig)

    if cat_cols:
        st.markdown("### 🏷️ Categorical Features")
        for col in cat_cols[:3]:  # only first 3 to avoid clutter
            fig, ax = plt.subplots(figsize=(7,4))
            sns.barplot(
                x=df[col].value_counts().head(10).index,
                y=df[col].value_counts().head(10).values,
                palette="viridis",
                ax=ax
            )
            ax.set_title(f"Top Categories in {col}", fontsize=12, color="purple")
            ax.set_ylabel("Count")
            ax.set_xlabel(col)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ----------------- Bivariate Analysis -----------------
    st.subheader("🔗 Bivariate Analysis")

    if len(num_cols) > 1:
        st.markdown("### 📌 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=14, color="darkgreen")
        st.pyplot(fig)

    # ----------------- Domain-Specific Analysis -----------------
    if "Product" in df.columns and "OrderQuantity" in df.columns:
        st.subheader("🏆 Top 10 Most In-Demand Products")
        top_products = df.groupby("Product")["OrderQuantity"].sum().sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=top_products.index, y=top_products.values, palette="magma", ax=ax)
        ax.set_title("Top 10 Products by Demand", fontsize=14, color="navy")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        if "OrderDate" in df.columns:
            st.markdown("### 📈 Trends of Top 10 Products")
            df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")
            top_df = df[df["Product"].isin(top_products.index)]
            trend = top_df.groupby(["OrderDate", "Product"])["OrderQuantity"].sum().reset_index()

            fig, ax = plt.subplots(figsize=(12,6))
            sns.lineplot(data=trend, x="OrderDate", y="OrderQuantity", hue="Product", palette="tab10", ax=ax)
            ax.set_title("Demand Trends of Top 10 Products", fontsize=14, color="darkred")
            st.pyplot(fig)

        if "Price" in df.columns:
            st.markdown("### 💰 Price vs Orders (Top 10 Products)")
            price_order = df[df["Product"].isin(top_products.index)].groupby("Product").agg(
                avg_price=("Price", "mean"),
                total_orders=("OrderQuantity", "sum")
            ).reset_index()

            fig, ax1 = plt.subplots(figsize=(10,6))
            sns.barplot(data=price_order, x="Product", y="avg_price", palette="Blues", ax=ax1)
            ax2 = ax1.twinx()
            sns.lineplot(data=price_order, x="Product", y="total_orders", marker="o", color="crimson", ax=ax2)
            ax1.set_ylabel("Average Price", color="blue")
            ax2.set_ylabel("Total Orders", color="red")
            ax1.set_title("Price vs Orders for Top 10 Products", fontsize=14, color="black")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ----------------- Conclusion -----------------
    st.subheader("📝 Automated Conclusion")
    st.success(f"✔ Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    if num_cols:
        st.info("✔ Numeric features show varied distributions; spread highlighted by standard deviations.")
    if cat_cols:
        st.warning("✔ Categorical features show imbalances (some dominant categories).")
    if "Product" in df.columns and "OrderQuantity" in df.columns:
        most_demanded = df.groupby("Product")["OrderQuantity"].sum().idxmax()
        st.success(f"✔ The most demanded product is **{most_demanded}**.")
    st.write("📌 Use these insights to guide feature engineering and business decisions.")
