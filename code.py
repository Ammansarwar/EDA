import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 E-Commerce EDA", layout="wide")
st.title("🛒 E-Commerce Exploratory Data Analysis (EDA)")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Calculate revenue
    df["Revenue"] = df["quantity"] * df["price"]

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    # 1. Histogram of quantity
    st.subheader("📦 Quantity Distribution")
    fig = px.histogram(df, x="quantity", nbins=20, color_discrete_sequence=["#5DADE2"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Shows how many items customers usually order.")

    # 2. Histogram of discount
    st.subheader("💸 Discount Distribution")
    fig = px.histogram(df, x="discount", nbins=20, color_discrete_sequence=["#58D68D"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Shows the distribution of discounts applied to orders.")

    # 3. Scatter: Price vs Quantity
    st.subheader("📈 Price vs Quantity (Scatter)")
    fig = px.scatter(df, x="price", y="quantity", color="quantity",
                     color_continuous_scale="Blues", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Helps detect if high-priced products are bought in smaller quantities.")

    # 4. Scatter: Discount vs Quantity
    st.subheader("🎯 Discount vs Quantity (Scatter)")
    fig = px.scatter(df, x="discount", y="quantity", color="discount",
                     color_continuous_scale="Viridis", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Shows if discounts lead to higher purchase quantities.")

    # 5. Top 10 Products by Revenue
    st.subheader("🏆 Top 10 Products by Revenue")
    top_products = df.groupby("product_id")["Revenue"].sum().nlargest(10)
    fig = px.bar(top_products, x=top_products.index, y=top_products.values,
                 color=top_products.values, color_continuous_scale="Tealgrn")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Highlights the products contributing the most revenue.")

    # 6. Top 10 Customers by Spending
    st.subheader("👥 Top 10 Customers by Spending")
    top_customers = df.groupby("customer_id")["Revenue"].sum().nlargest(10)
    fig = px.bar(top_customers, x=top_customers.index, y=top_customers.values,
                 color=top_customers.values, color_continuous_scale="Sunset")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Identifies the most valuable customers.")

    # 7. Correlation Heatmap
    st.subheader("📊 Correlation Heatmap")
    corr = df[["quantity", "price", "discount", "Revenue"]].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)
    st.info("Shows correlations between numerical features.")

else:
    st.warning("📂 Please upload a CSV file to begin EDA.")
