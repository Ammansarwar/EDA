import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="E-Commerce EDA", layout="wide")

st.title("🛒 E-Commerce Exploratory Data Analysis (EDA)")

# Upload file
uploaded_file = st.file_uploader("Upload your E-commerce CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Basic Info")
    st.write(df.describe(include="all"))

    # Drop duplicates + handle missing
    df = df.drop_duplicates()
    df = df.fillna("Unknown")

    st.success("✅ Data cleaned (duplicates removed, missing values filled).")

    # ---------------- Meaningful Graphs ---------------- #

    # 1. Top 10 Most Sold Products
    if "Product" in df.columns:
        st.subheader("🏆 Top 10 Most Sold Products")
        top_products = df["Product"].value_counts().nlargest(10).reset_index()
        top_products.columns = ["Product", "Count"]
        fig1 = px.bar(top_products, x="Product", y="Count", 
                      color="Count", text="Count",
                      title="Top 10 Products", 
                      color_continuous_scale="blues")
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Top 10 Cities by Revenue
    if "City" in df.columns and "Revenue" in df.columns:
        st.subheader("🌆 Top 10 Cities by Revenue")
        top_cities = df.groupby("City")["Revenue"].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_cities, x="Revenue", y="City",
                      orientation="h", color="Revenue", text="Revenue",
                      title="Top 10 Cities by Revenue",
                      color_continuous_scale="viridis")
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Monthly Sales Trend
    if "Date" in df.columns and "Revenue" in df.columns:
        st.subheader("📈 Monthly Sales Trend")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum().reset_index()
        monthly_sales["Date"] = monthly_sales["Date"].astype(str)
        fig3 = px.line(monthly_sales, x="Date", y="Revenue", markers=True,
                       title="Monthly Sales Trend")
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Payment Method Distribution
    if "PaymentMethod" in df.columns:
        st.subheader("💳 Payment Method Distribution")
        payment_counts = df["PaymentMethod"].value_counts().reset_index()
        payment_counts.columns = ["PaymentMethod", "Count"]
        fig4 = px.pie(payment_counts, names="PaymentMethod", values="Count",
                      title="Payment Method Share", hole=0.4)
        st.plotly_chart(fig4, use_container_width=True)

    # 5. Category Contribution
    if "Category" in df.columns and "Revenue" in df.columns:
        st.subheader("📦 Sales Contribution by Category")
        cat_sales = df.groupby("Category")["Revenue"].sum().reset_index()
        fig5 = px.treemap(cat_sales, path=["Category"], values="Revenue",
                          title="Category-wise Revenue Contribution")
        st.plotly_chart(fig5, use_container_width=True)

    # 6. Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        fig6 = px.imshow(numeric_df.corr(), text_auto=True,
                         title="Correlation Heatmap of Numeric Features")
        st.plotly_chart(fig6, use_container_width=True)

    st.success("✅ EDA Completed with Interactive Graphs!")

