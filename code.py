import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Auto EDA App", layout="wide")
st.title("🤖 Automated Exploratory Data Analysis (EDA)")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Dataset preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # Basic Info
    st.subheader("📑 Column Info")
    buffer = []
    for col in df.columns:
        buffer.append({
            "Column": col,
            "Data Type": df[col].dtype,
            "Missing Values": df[col].isnull().sum(),
            "Unique Values": df[col].nunique()
        })
    st.dataframe(pd.DataFrame(buffer))

    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # Missing Values
    st.subheader("🚨 Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(fig)

    # Correlation Heatmap (for numeric data)
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        st.subheader("📈 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Distribution plots
        st.subheader("📉 Distribution of Numeric Columns")
        col = st.selectbox("Select a column to plot", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        # Scatter plot
        st.subheader("⚡ Scatter Plot Between Two Columns")
        col_x = st.selectbox("X-axis", numeric_df.columns, index=0)
        col_y = st.selectbox("Y-axis", numeric_df.columns, index=min(1, len(numeric_df.columns)-1))
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
        st.pyplot(fig)

    else:
        st.info("⚠ No numeric columns available for correlation/distribution plots.")

else:
    st.info("👆 Upload a dataset (CSV/Excel) to start automatic EDA.")
