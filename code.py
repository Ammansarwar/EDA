import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Full EDA App", layout="wide")
st.title("📊 Automated Exploratory Data Analysis (EDA)")

# File upload
uploaded_file = st.file_uploader("Upload a dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # ----------------- Info & Missing -----------------
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
    st.subheader("📈 Univariate Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        st.write("### Numeric Distributions")
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            plt.title(f"Boxplot of {col}")
            st.pyplot(fig)

    if cat_cols:
        st.write("### Categorical Distributions")
        for col in cat_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().head(20).plot(kind="bar", ax=ax)
            plt.title(f"Top Categories in {col}")
            st.pyplot(fig)

    # ----------------- Bivariate Analysis -----------------
    st.subheader("🔗 Bivariate Analysis")

    if len(num_cols) > 1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if len(num_cols) >= 2:
        st.write("### Scatterplots")
        for col in num_cols[:2]:  # show for first two to avoid overload
            fig, ax = plt.subplots
