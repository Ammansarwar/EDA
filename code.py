import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="📊 Automated EDA App", layout="wide")
st.title("📊 Automated Exploratory Data Analysis (EDA) App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ----------------------------
    # Step 1: Dataset Overview
    # ----------------------------
    st.header("🔍 Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**First 5 rows:**")
    st.dataframe(df.head())

    # Missing values
    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum())

    # ----------------------------
    # Step 2: Summary Statistics
    # ----------------------------
    st.header("📈 Summary Statistics")
    st.write(df.describe(include="all"))

    # ----------------------------
    # Step 3: Numeric Distributions
    # ----------------------------
    st.header("📊 Numeric Column Distributions")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=['#1f77b4'])
        fig.update_layout(title=f"Distribution of {col}", bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Step 4: Categorical Analysis
    # ----------------------------
    st.header("🗂️ Categorical Column Analysis")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        fig = px.bar(df[col].value_counts().reset_index(),
                     x="index", y=col, color="index",
                     title=f"Distribution of {col}",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Step 5: Correlation Heatmap
    # ----------------------------
    if len(num_cols) > 1:
        st.header("🔗 Correlation Heatmap")
        corr = df[num_cols].corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            colorscale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Step 6: Time Series (if date present)
    # ----------------------------
    if "Order Date" in df.columns:
        st.header("⏳ Time Series Analysis")
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        if df["Order Date"].notnull().sum() > 0:
            time_series = df.groupby("Order Date").sum(numeric_only=True).reset_index()
            for col in num_cols:
                fig = px.line(time_series, x="Order Date", y=col,
                              title=f"{col} Over Time",
                              color_discrete_sequence=["#e63946"])
                st.plotly_chart(fig, use_container_width=True)

    st.success("✅ EDA Completed!")
else:
    st.info("📤 Please upload a CSV file to start EDA.")
