import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App Title
st.set_page_config(page_title="E-Commerce EDA", layout="wide")
st.title("🛍️ E-Commerce Exploratory Data Analysis (EDA)")

# Upload file
uploaded_file = st.file_uploader("📂 Upload your E-commerce CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Dataset summary
    st.subheader("📊 Dataset Info")
    st.write("Shape of dataset:", df.shape)
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # ---- Meaningful Graphs ----

    # 1. Top 10 Selling Products
    st.subheader("🏆 Top 10 Selling Products")
    if "Product" in df.columns:
        top_products = df["Product"].value_counts().nlargest(10)
        fig = px.bar(top_products, 
                     x=top_products.index, 
                     y=top_products.values, 
                     labels={'x':"Product", 'y':"Number of Sales"}, 
                     title="Top 10 Products Sold",
                     color=top_products.values, 
                     color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 This graph shows which products were sold the most. Useful for inventory and marketing focus.")

    # 2. Revenue by Category
    st.subheader("📦 Revenue by Category")
    if "Category" in df.columns and "Revenue" in df.columns:
        category_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
        fig = px.pie(values=category_revenue.values, 
                     names=category_revenue.index, 
                     title="Revenue Contribution by Category",
                     color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 This pie chart shows which categories bring the most revenue.")

    # 3. Sales Trend Over Time
    st.subheader("📅 Sales Trend Over Time")
    if "Date" in df.columns and "Revenue" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        daily_sales = df.groupby("Date")["Revenue"].sum().reset_index()
        fig = px.line(daily_sales, 
                      x="Date", 
                      y="Revenue", 
                      title="Daily Revenue Trend",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 This shows revenue changes over time. Useful for spotting growth, seasonality, or dips.")

    # 4. Payment Method Distribution
    st.subheader("💳 Payment Method Distribution")
    if "Payment_Method" in df.columns:
        fig = px.histogram(df, x="Payment_Method", color="Payment_Method", 
                           title="Payment Methods Used",
                           color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 This shows which payment methods customers prefer the most.")

    # 5. Customer Location Heatmap
    st.subheader("🌍 Customer Location Heatmap")
    if "City" in df.columns:
        city_sales = df["City"].value_counts().nlargest(10)
        fig = px.bar(city_sales, 
                     x=city_sales.index, 
                     y=city_sales.values, 
                     title="Top 10 Cities by Sales",
                     color=city_sales.values,
                     color_continuous_scale="Mint")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 This shows where most customers are located. Useful for regional marketing.")

    # 6. Revenue Distribution by Discount
    st.subheader("💸 Discount Impact on Revenue")
    if "Discount" in df.columns and "Revenue" in df.columns:
        fig = px.scatter(df, x="Discount", y="Revenue", 
                         title="Discount vs Revenue",
                         color="Revenue", 
                         color_continuous_scale="Sunset")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Helps analyze whether giving discounts increases revenue.")

    # 7. Boxplot of Revenue by Category
    st.subheader("📦 Revenue Spread per Category")
    if "Category" in df.columns and "Revenue" in df.columns:
        fig = px.box(df, x="Category", y="Revenue", 
                     color="Category", 
                     title="Revenue Distribution per Category")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Shows how revenue varies within each category (detects outliers & spread).")

    st.success("✅ EDA Completed with Multiple Interactive Graphs!")
