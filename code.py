import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App Title
st.title("📊 Automated EDA App for CSV Files")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Show first rows
    st.subheader("📋 First 5 Rows")
    st.write(df.head())

    # Dataset Info
    st.subheader("📊 Dataset Info")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # =========================
    # Graphs
    # =========================
    st.subheader("📈 Exploratory Graphs")

    # 1. Numeric column distributions
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        st.write("Histogram of Numeric Columns")
        df[num_cols].hist(figsize=(10, 6))
        st.pyplot(plt.gcf())
        plt.clf()

    # 2. Categorical column counts
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        for col in cat_cols[:3]:  # limit to 3 for clarity
            st.write(f"Value counts for {col}")
            df[col].value_counts().plot(kind='bar', figsize=(6, 4))
            plt.title(f"Distribution of {col}")
            st.pyplot(plt.gcf())
            plt.clf()

    st.success("✅ EDA Completed!")

else:
    st.info("📤 Please upload a CSV file to start EDA.")
