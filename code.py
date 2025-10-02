import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Streamlit App Title
# ------------------------------
st.set_page_config(page_title="E-Commerce EDA", layout="wide")
st.title("🛒 E-Commerce Exploratory Data Analysis (EDA)")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload your E-commerce CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Basic Information
    # ------------------------------
    st.subheader("🔍 Basic Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # ------------------------------
    # Create new column: Revenue
    # ------------------------------
    df["revenue"] = df["quantity"] * df["price"]

    # ------------------------------
    # Graph 1: Top 10 Products by Quantity
    # ------------------------------
    st.subheader("📦 Top 10 Products by Quantity Sold")
    top_products = df.groupby("product_id")["quantity"].sum().nlargest(10).reset_index()
    fig1 = px.bar(top_products, x="product_id", y="quantity", text="quantity",
                  color="quantity", height=500, width=900,
                  title="Top 10 Products by Quantity Sold")
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------
    # Graph 2: Top 10 Products by Revenue
    # ------------------------------
    st.subheader("💰 Top 10 Products by Revenue")
    top_revenue = df.groupby("product_id")["revenue"].sum().nlargest(10).reset_index()
    fig2 = px.bar(top_revenue, x="product_id", y="revenue", text="revenue",
                  color="revenue", height=500, width=900,
                  title="Top 10 Products by Revenue")
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------
    # Graph 3: Top 10 Customers by Spending
    # ------------------------------
    st.subheader("🧑‍🤝‍🧑 Top 10 Customers by Total Spending")
    top_customers = df.groupby("customer_id")["revenue"].sum().nlargest(10).reset_index()
    fig3 = px.bar(top_customers, x="customer_id", y="revenue", text="revenue",
                  color="revenue", height=500, width=900,
                  title="Top 10 Customers by Spending")
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------
    # Graph 4: Price Distribution
    # ------------------------------
    st.subheader("📊 Price Distribution")
    fig4 = px.histogram(df, x="price", nbins=30, color_discrete_sequence=["#FF5733"],
                        height=500, width=900, title="Price Distribution of Products")
    st.plotly_chart(fig4, use_container_width=True)

    # ------------------------------
    # Graph 5: Quantity Distribution
    # ------------------------------
    st.subheader("📊 Quantity Distribution")
    fig5 = px.histogram(df, x="quantity", nbins=30, color_discrete_sequence=["#33FF57"],
                        height=500, width=900, title="Quantity Distribution")
    st.plotly_chart(fig5, use_container_width=True)

    # ------------------------------
    # Graph 6: Discount vs Revenue
    # ------------------------------
    st.subheader("💸 Discount Impact on Revenue")
    fig6 = px.scatter(df, x="discount", y="revenue", size="quantity", color="price",
                      height=500, width=900,
                      title="Discount vs Revenue (Bubble Size = Quantity, Color = Price)")
    st.plotly_chart(fig6, use_container_width=True)

    # ------------------------------
    # Graph 7: Correlation Heatmap
    # ------------------------------
    st.subheader("📌 Correlation Heatmap")
    corr = df[["quantity", "price", "discount", "revenue"]].corr()
    fig7 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
                     height=600, width=900, title="Correlation Heatmap of Numeric Features")
    st.plotly_chart(fig7, use_container_width=True)

    # ------------------------------
    # Finish
    # ------------------------------
    st.success("✅ EDA Completed with Interactive Graphs!")

else:
    st.warning("⚠️ Please upload a CSV file to start the analysis.")
