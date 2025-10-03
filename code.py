import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# Title
# ---------------------------
st.set_page_config(page_title="📊 E-Commerce EDA", layout="wide")
st.title("🛍️ E-Commerce Exploratory Data Analysis")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your E-commerce CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Check required columns
    required_cols = ["product_id", "quantity", "price", "order_date", "payment_method", "region"]
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"⚠️ Column `{col}` not found in dataset. Some graphs may not display properly.")

    # Create Revenue column
    if "quantity" in df.columns and "price" in df.columns:
        df["Revenue"] = df["quantity"] * df["price"]

    # Ensure date type
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # ---------------------------
    # 1️⃣ Top 10 Selling Products
    # ---------------------------
    if "product_id" in df.columns:
        st.subheader("🏆 Top 10 Selling Products")
        top_products = df.groupby("product_id")["quantity"].sum().nlargest(10).reset_index()
        fig1 = px.bar(top_products, x="quantity", y="product_id", orientation="h",
                      color="quantity", color_continuous_scale="Blues",
                      title="Top 10 Products by Quantity Sold")
        st.plotly_chart(fig1, use_container_width=True)

    # ---------------------------
    # 2️⃣ Revenue by Category
    # ---------------------------
    if "category" in df.columns:
        st.subheader("📦 Revenue by Category")
        revenue_cat = df.groupby("category")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
        fig2 = px.bar(revenue_cat, x="category", y="Revenue", color="Revenue",
                      color_continuous_scale="Purples", title="Revenue by Category")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------
    # 3️⃣ Sales Trend Over Time
    # ---------------------------
    if "order_date" in df.columns:
        st.subheader("📅 Sales Trend Over Time")
        sales_trend = df.groupby(df["order_date"].dt.to_period("M"))["Revenue"].sum().reset_index()
        sales_trend["order_date"] = sales_trend["order_date"].astype(str)
        fig3 = px.line(sales_trend, x="order_date", y="Revenue", markers=True,
                       title="Monthly Revenue Trend", line_shape="spline")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------
    # 4️⃣ Payment Method Distribution
    # ---------------------------
    if "payment_method" in df.columns:
        st.subheader("💳 Payment Method Distribution")
        fig4 = px.pie(df, names="payment_method", title="Share of Payment Methods",
                      color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig4, use_container_width=True)

    # ---------------------------
    # 5️⃣ Customer Location Revenue
    # ---------------------------
    if "region" in df.columns:
        st.subheader("🌍 Customer Location Revenue")
        region_rev = df.groupby("region")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
        fig5 = px.bar(region_rev, x="region", y="Revenue", color="Revenue",
                      color_continuous_scale="Greens", title="Revenue by Region")
        st.plotly_chart(fig5, use_container_width=True)

    st.success("✅ EDA Completed with 5 Most Meaningful Interactive Graphs!")
