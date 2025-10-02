import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="E-Commerce EDA", layout="wide")
st.title("🛒 E-Commerce Exploratory Data Analysis (EDA)")

# Upload file
uploaded_file = st.file_uploader("Upload your E-commerce CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded!")

    # Show basic info
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # ================================
    # 1. Sales Trend Over Time
    # ================================
    if "Order Date" in df.columns and "Sales" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        sales_trend = df.groupby("Order Date")["Sales"].sum().reset_index()
        fig = px.line(sales_trend, x="Order Date", y="Sales",
                      title="📈 Sales Trend Over Time",
                      color_discrete_sequence=["#e63946"])
        st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 2. Top Selling Products
    # ================================
    if "Product Name" in df.columns and "Sales" in df.columns:
        top_products = df.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x="Sales", y="Product Name", orientation="h",
                     title="🏆 Top 10 Products by Sales",
                     color="Sales", color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 3. Sales by Category
    # ================================
    if "Category" in df.columns and "Sales" in df.columns:
        category_sales = df.groupby("Category")["Sales"].sum().reset_index()
        fig = px.treemap(category_sales, path=["Category"], values="Sales",
                         title="📦 Sales by Category",
                         color="Sales", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 4. Sales by Region/City
    # ================================
    if "Region" in df.columns and "Sales" in df.columns:
        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        fig = px.bar(region_sales, x="Region", y="Sales",
                     title="🌍 Sales by Region",
                     color="Sales", color_continuous_scale="plasma")
        st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 5. Payment Methods
    # ================================
    if "Payment Mode" in df.columns:
        payment_counts = df["Payment Mode"].value_counts().reset_index()
        payment_counts.columns = ["Payment Mode", "Count"]
        fig = px.pie(payment_counts, names="Payment Mode", values="Count",
                     title="💳 Payment Method Distribution",
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

    # ================================
    # 6. Order Status
    # ================================
    if "Order Status" in df.columns:
        status_counts = df["Order Status"].value_counts().reset_index()
        status_counts.columns = ["Order Status", "Count"]
        fig = px.bar(status_counts, x="Order Status", y="Count",
                     title="📦 Order Status Distribution",
                     color="Order Status",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    st.success("✅ EDA Completed with Interactive Graphs!")
else:
    st.info("📤 Please upload your dataset to begin EDA.")
