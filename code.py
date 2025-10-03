# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid", palette="cool")

# 2. Load dataset
data = pd.read_csv("ecommerce_dataset.csv")

# 3. Preview dataset
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\nSummary statistics:\n", data.describe())
print("\nInfo:")
print(data.info())

# 4. Missing values
print("\nMissing values:\n", data.isnull().sum())

# 5. Remove duplicates
data.drop_duplicates(inplace=True)

# 6. Feature Engineering: Revenue
data["Revenue"] = data["Quantity"] * data["Price"]

# 7. Outlier detection (example for Price)
q1, q3 = np.percentile(data['Price'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
print(f"\nPrice Outlier Range: {lower_bound} to {upper_bound}")

# --- 📊 Visualizations ---

# 8. Quantity distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Quantity'], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Quantity Ordered")
plt.show()

# 9. Price distribution
plt.figure(figsize=(8,5))
sns.histplot(data['Price'], bins=30, kde=True, color="orange")
plt.title("Distribution of Product Prices")
plt.show()

# 10. Revenue distribution (boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x=data['Revenue'], color="lightgreen")
plt.title("Revenue Distribution per Order")
plt.show()

# 11. Top 10 products by revenue
top_products = data.groupby("Product_ID")["Revenue"].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_products.values, y=top_products.index, palette="Blues_r")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Product ID")
plt.show()

# 12. Top 10 customers by spending
top_customers = data.groupby("Customer_ID")["Revenue"].sum().nlargest(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_customers.values, y=top_customers.index, palette="Purples_r")
plt.title("Top 10 Customers by Spending")
plt.xlabel("Total Spending")
plt.ylabel("Customer ID")
plt.show()

# 13. Scatter: Price vs Quantity
plt.figure(figsize=(8,5))
sns.scatterplot(x="Price", y="Quantity", data=data, alpha=0.6, color="red")
plt.title("Price vs Quantity")
plt.show()

# 14. Scatter: Discount vs Quantity
plt.figure(figsize=(8,5))
sns.scatterplot(x="Discount", y="Quantity", data=data, alpha=0.6, color="green")
plt.title("Discount vs Quantity Ordered")
plt.show()

# 15. Correlation heatmap
plt.figure(figsize=(8,6))
corr = data[["Quantity", "Price", "Discount", "Revenue"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Key Variables")
plt.show()
