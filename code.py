import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Meaningful EDA", layout="wide")

st.title("🛒 Interactive EDA Dashboard")
st.write("Upload your dataset and explore meaningful insights with cool-colored interactive graphs.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("🔍 Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Columns:", list(df.columns))

    # Split types
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # ---------- Summary Statistics ----------
    st.subheader("📊 Summary Statistics (Numerical)")
    st.write(df[num_cols].describe().T)
    st.info("ℹ️ This shows key descriptive statistics (mean, std, min, max, quartiles) "
            "for numeric columns.")

    # ---------- Top Categories Pie Chart ----------
    if cat_cols:
        st.subheader("🥧 Category Distribution (Pie Chart)")
        for col in cat_cols[:1]:  # Just first categorical col
            top10 = df[col].value_counts().nlargest(10)
            fig = px.pie(values=top10.values, names=top10.index, color_discrete_sequence=px.colors.sequential.Blues)
            fig.update_layout(title=f"Top 10 {col} Distribution")
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"ℹ️ This pie chart shows the proportion of the **Top 10 categories in {col}**. "
                    "Helps visualize dominance of categories.")

    # ---------- Revenue or Quantity Analysis ----------
    if set(["Quantity", "Price"]).issubset(df.columns):
        st.subheader("💸 Revenue Analysis (Bar)")
        df["Revenue"] = df["Quantity"] * df["Price"]
        top_products = df.groupby("Product")["Revenue"].sum().nlargest(10)
        fig = px.bar(top_products, x=top_products.index, y=top_products.values,
                     color=top_products.values, color_continuous_scale="Tealgrn")
        fig.update_layout(title="Top 10 Products by Revenue", xaxis_title="Product", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This bar chart shows the **Top 10 revenue-generating products**. "
                "Useful to identify best-sellers.")

    # ---------- Scatter Plot ----------
    if set(["Quantity", "Price"]).issubset(df.columns):
        st.subheader("📈 Quantity vs Price (Scatter Plot)")
        fig = px.scatter(df, x="Quantity", y="Price", color="Price", 
                         color_continuous_scale="Blues", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This scatter plot shows how **Price relates to Quantity**. "
                "Helps detect outliers (very high price or quantity orders).")

    # ---------- Boxplot ----------
    if "Price" in df.columns:
        st.subheader("📦 Price Distribution by Product (Boxplot)")
        top_products = df["Product"].value_counts().nlargest(5).index
        fig = px.box(df[df["Product"].isin(top_products)], x="Product", y="Price", color="Product")
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This boxplot shows **price spread for top products**. "
                "Helps detect extreme values and variation.")

    # ---------- Line Plot (if Date column exists) ----------
    date_cols = [col for col in df.columns if "date" in col.lower()]
    if date_cols and "Revenue" in df.columns:
        st.subheader("📆 Revenue Trend Over Time (Line Plot)")
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        df_time = df.groupby(df[date_cols[0]].dt.to_period("M"))["Revenue"].sum().reset_index()
        df_time[date_cols[0]] = df_time[date_cols[0]].astype(str)
        fig = px.line(df_time, x=date_cols[0], y="Revenue", markers=True, color_discrete_sequence=["#4BC0C0"])
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This line plot shows **how revenue changes over time**. "
                "Helps identify seasonality or growth trends.")

    # ---------- Correlation Heatmap ----------
    if len(num_cols) > 1:
        st.subheader("📊 Correlation Heatmap")
        corr = df[num_cols].corr()
        plt.figure(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
        st.info("ℹ️ The heatmap shows correlations between numeric columns. "
                "Useful for detecting relationships between features.")

    # ---------- Missing Values ----------
    st.subheader("❌ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig = px.bar(missing, x=missing.index, y=missing.values,
                     color=missing.values, color_continuous_scale="Purples")
        fig.update_layout(title="Missing Values per Column", xaxis_title="Column", yaxis_title="Missing Count")
        st.plotly_chart(fig, use_container_width=True)
        st.info("ℹ️ This chart highlights columns with missing values. "
                "Understanding missing data is crucial before modeling.")
    else:
        st.success("✅ No missing values found in this dataset!")

else:
    st.warning("📂 Please upload a CSV file to begin EDA.")
