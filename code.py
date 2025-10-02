import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------
# Streamlit Page Setup
# --------------------------------
st.set_page_config(page_title="E-Commerce EDA", layout="wide")
st.title("🛒 E-Commerce Exploratory Data Analysis (EDA)")

# --------------------------------
# File Upload
# --------------------------------
uploaded_file = st.file_uploader("📂 Upload your E-commerce CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------
    # Step 1: Data Cleaning / New Columns
    # --------------------------------
    st.subheader("🧹 Basic Information")
    st.write(f"🔹 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("🔹 Missing Values:")
    st.write(df.isnull().sum())
    st.write("🔹 Duplicate Rows:", df.duplicated().sum())

    # Add revenue column
    df["revenue"] = df["quantity"] * df["price"]

    # --------------------------------
    # Step 2: Univariate Analysis
    # --------------------------------
    st.header("📊 Univariate Analysis")

    # Price Distribution
    fig1 = px.histogram(df, x="price", nbins=30, color_discrete_sequence=["#FF6347"],
                        title="Price Distribution", height=500, width=900)
    st.plotly_chart(fig1, use_container_width=True)

    # Quantity Distribution
    fig2 = px.histogram(df, x="quantity", nbins=30, color_discrete_sequence=["#4682B4"],
                        title="Quantity Distribution", height=500, width=900)
    st.plotly_chart(fig2, use_container_width=True)

    # Discount Distribution
    fig3 = px.histogram(df, x="discount", nbins=30, color_discrete_sequence=["#32CD32"],
                        title="Discount Distribution", height=500, width=900)
    st.plotly_chart(fig3, use_container_width=True)

    # --------------------------------
    # Step 3: Bivariate Analysis
    # --------------------------------
    st.header("🔗 Bivariate Analysis")

    # Price vs Quantity
    fig4 = px.scatter(df, x="price", y="quantity", size="revenue", color="discount",
                      title="Price vs Quantity (Bubble = Revenue, Color = Discount)",
                      height=500, width=900)
    st.plotly_chart(fig4, use_container_width=True)

    # Discount vs Quantity
    fig5 = px.scatter(df, x="discount", y="quantity", size="revenue", color="price",
                      title="Discount vs Quantity (Bubble = Revenue, Color = Price)",
                      height=500, width=900)
    st.plotly_chart(fig5, use_container_width=True)

    # Discount vs Revenue
    fig6 = px.scatter(df, x="discount", y="revenue", size="quantity", color="price",
                      title="Discount vs Revenue (Bubble = Quantity, Color = Price)",
                      height=500, width=900)
    st.plotly_chart(fig6, use_container_width=True)

    # --------------------------------
    # Step 4: Aggregates (Top Entities)
    # --------------------------------
    st.header("🏆 Aggregate Insights")

    # Top 10 Products by Revenue
    top_products_revenue = df.groupby("product_id")["revenue"].sum().nlargest(10).reset_index()
    fig7 = px.bar(top_products_revenue, x="product_id", y="revenue", text="revenue",
                  color="revenue", title="Top 10 Products by Revenue",
                  height=500, width=900)
    st.plotly_chart(fig7, use_container_width=True)

    # Top 10 Products by Quantity
    top_products_quantity = df.groupby("product_id")["quantity"].sum().nlargest(10).reset_index()
    fig8 = px.bar(top_products_quantity, x="product_id", y="quantity", text="quantity",
                  color="quantity", title="Top 10 Products by Quantity",
                  height=500, width=900)
    st.plotly_chart(fig8, use_container_width=True)

    # Top 10 Customers by Spending
    top_customers = df.groupby("customer_id")["revenue"].sum().nlargest(10).reset_index()
    fig9 = px.bar(top_customers, x="customer_id", y="revenue", text="revenue",
                  color="revenue", title="Top 10 Customers by Spending",
                  height=500, width=900)
    st.plotly_chart(fig9, use_container_width=True)

    # --------------------------------
    # Step 5: Correlation Heatmap
    # --------------------------------
    st.header("📌 Correlation Heatmap")
    corr = df[["quantity", "price", "discount", "revenue"]].corr()
    fig10 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
                      title="Correlation Heatmap of Numeric Features",
                      height=600, width=900)
    st.plotly_chart(fig10, use_container_width=True)

    # --------------------------------
    st.success("✅ Full EDA Completed with 10 Interactive Graphs!")

else:
    st.warning("⚠️ Please upload a CSV file to start the EDA.")
