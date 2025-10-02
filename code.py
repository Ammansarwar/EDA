# 📌 EDA Script for E-commerce Dataset
# Save this as eda_ecommerce.py or EDA.ipynb for GitHub

# Import libraries
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Make Plotly charts show in browser
pio.renderers.default = "notebook_connected"

# 1. Load dataset
df = pd.read_csv("ecommerce_dataset.csv")

# 2. Basic dataset info
print("🔎 Dataset Shape:", df.shape)
print("\n📋 First 5 rows:\n", df.head())
print("\n📊 Column Info:\n")
print(df.info())
print("\n📈 Missing Values:\n", df.isnull().sum())

# =============================
# 📊 Meaningful Graphs
# =============================

# 3. Sales Trend Over Time
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    sales_trend = df.groupby('Order Date')['Sales'].sum().reset_index()
    fig1 = px.line(sales_trend, x='Order Date', y='Sales', title="Sales Trend Over Time")
    fig1.show()

# 4. Top-Selling Products
if 'Product Name' in df.columns:
    top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
    fig2 = px.bar(top_products, x='Sales', y='Product Name', orientation='h',
                  title="Top 10 Selling Products")
    fig2.show()

# 5. Sales by Category
if 'Category' in df.columns:
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig3 = px.pie(category_sales, names='Category', values='Sales', title="Sales by Category")
    fig3.show()

# 6. Sales by Region/City
if 'Region' in df.columns:
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    fig4 = px.bar(region_sales, x='Region', y='Sales', title="Sales by Region")
    fig4.show()

# 7. Payment Method Distribution
if 'Payment Mode' in df.columns:
    payment_counts = df['Payment Mode'].value_counts().reset_index()
    payment_counts.columns = ['Payment Mode', 'Count']
    fig5 = px.pie(payment_counts, names='Payment Mode', values='Count', 
                  title="Payment Method Distribution")
    fig5.show()

# 8. Order Status Tracking
if 'Order Status' in df.columns:
    status_counts = df['Order Status'].value_counts().reset_index()
    status_counts.columns = ['Order Status', 'Count']
    fig6 = px.bar(status_counts, x='Order Status', y='Count', 
                  title="Order Status Distribution")
    fig6.show()

print("\n✅ EDA Completed! All interactive charts displayed.")
