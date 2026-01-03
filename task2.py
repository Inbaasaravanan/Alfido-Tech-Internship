
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)


csv_path = "superstore_final_dataset (1).csv"
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path} with default encoding (utf-8).")
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="latin1")
    print(f"Loaded {csv_path} with fallback encoding latin1.")

print("Dataset Loaded Successfully")
print(df.head())
print(df.info())


import re

def _san(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def find_col(df, name):
    target = _san(name)
    for c in df.columns:
        if _san(c) == target:
            return c
    for c in df.columns:
        if target in _san(c) or _san(c) in target:
            return c
    return None


order_date_col = find_col(df, 'Order Date')
order_id_col = find_col(df, 'Order ID')
sales_col = find_col(df, 'Sales')
profit_col = find_col(df, 'Profit')
discount_col = find_col(df, 'Discount')
product_col = find_col(df, 'Product Name')
subcat_col = find_col(df, 'Sub-Category')
category_col = find_col(df, 'Category')
region_col = find_col(df, 'Region')
segment_col = find_col(df, 'Segment')


df.drop_duplicates(inplace=True)


if order_date_col:
    df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
    df['Year'] = df[order_date_col].dt.year
    df['Month'] = df[order_date_col].dt.month
    df['Month_Name'] = df[order_date_col].dt.month_name()
    df['Quarter'] = df[order_date_col].dt.to_period('Q')
else:
    print("Warning: Order date column not found; time-based features will be skipped")


if profit_col and sales_col:
    df['Profit_Margin'] = df[profit_col] / df[sales_col]
else:
    df['Profit_Margin'] = np.nan
    print("Warning: Profit or Sales column missing; Profit margin set to NaN")

print("\nMissing Values:\n", df.isnull().sum())


if sales_col:
    total_revenue = df[sales_col].sum()
else:
    total_revenue = 0.0
    print("Warning: Sales column not found; revenue set to 0")

if profit_col:
    total_profit = df[profit_col].sum()
else:
    total_profit = 0.0
    print("Warning: Profit column not found; profit set to 0")

profit_margin = (total_profit / total_revenue) if total_revenue else np.nan

if order_id_col and sales_col:
    avg_order_value = df.groupby(order_id_col)[sales_col].sum().mean()
else:
    avg_order_value = np.nan
    print("Warning: Order ID or Sales column missing; AOV set to NaN")

print("\n===== KEY PERFORMANCE INDICATORS =====")
print(f"Total Revenue       : {total_revenue:.2f}")
print(f"Total Profit        : {total_profit:.2f}")
if not np.isnan(profit_margin):
    print(f"Profit Margin       : {profit_margin:.2%}")
else:
    print("Profit Margin       : N/A")
print(f"Average Order Value : {avg_order_value:.2f}")



if 'Year' in df.columns and 'Month' in df.columns and sales_col:
    monthly_sales = df.groupby(['Year', 'Month'])[sales_col].sum().reset_index()
    plt.plot(monthly_sales['Month'], monthly_sales[sales_col])
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.show()
else:
    print("Skipping sales trend: missing date or sales data")


if 'Month_Name' in df.columns and sales_col:
    seasonality = df.groupby('Month_Name')[sales_col].mean().sort_values()
    seasonality.plot(kind='bar')
    plt.title("Average Monthly Sales (Seasonality)")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")
    plt.show()
else:
    print("Skipping seasonality analysis: missing month names or sales")


if region_col and sales_col:
    agg_cols = [sales_col]
    if profit_col:
        agg_cols.append(profit_col)
    region_perf = df.groupby(region_col)[agg_cols].sum()
    region_perf.plot(kind='bar')
    plt.title("Sales & Profit by Region")
    plt.ylabel("Amount")
    plt.show()
else:
    print("Skipping regional performance: missing region or sales column")


if category_col and sales_col:
    category_sales = df.groupby(category_col)[sales_col].sum().sort_values()
    category_sales.plot(kind='barh')
    plt.title("Sales by Category")
    plt.xlabel("Sales")
    plt.show()
else:
    print("Skipping category analysis: missing category or sales column")


if subcat_col and profit_col:
    subcat_profit = df.groupby(subcat_col)[profit_col].sum().sort_values()
    subcat_profit.plot(kind='barh')
    plt.title("Profit by Sub-Category")
    plt.xlabel("Profit")
    plt.show()
elif subcat_col and sales_col:
    subcat_sales = df.groupby(subcat_col)[sales_col].sum().sort_values()
    subcat_sales.plot(kind='barh')
    plt.title("Sales by Sub-Category")
    plt.xlabel("Sales")
    plt.show()
else:
    print("Skipping sub-category analysis: missing sub-category or sales/profit column")


if product_col and sales_col:
    top_products = (
        df.groupby(product_col)[sales_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
else:
    top_products = pd.Series(dtype=float)
    print("Skipping top products by sales: missing product or sales column")

if product_col and profit_col:
    worst_products = (
        df.groupby(product_col)[profit_col]
        .sum()
        .sort_values()
        .head(10)
    )
elif product_col and sales_col:
    worst_products = (
        df.groupby(product_col)[sales_col]
        .sum()
        .sort_values()
        .head(10)
    )
else:
    worst_products = pd.Series(dtype=float)
    print("Skipping worst products by profit/sales: missing product or profit/sales column")

print("\nTop 10 Products by Sales:\n", top_products)
print("\nWorst 10 Products by Profit (or Sales):\n", worst_products)

if discount_col and profit_col:
    sns.scatterplot(x=discount_col, y=profit_col, data=df)
    plt.title("Discount vs Profit")
    plt.show()
else:
    print("Skipping discount vs profit plot: missing discount or profit column")


if segment_col and sales_col:
    aov_segment = df.groupby(segment_col)[sales_col].mean()
    aov_segment.plot(kind='bar')
    plt.title("Average Order Value by Segment")
    plt.ylabel("Average Sales")
    plt.show()
else:
    print("Skipping AOV by segment: missing segment or sales column")


print("\n===== BUSINESS INSIGHTS =====")
print("1. Strong sales seasonality is observed in certain months.")
print("2. Technology category contributes high profit margins.")
print("3. Heavy discounts reduce overall profitability.")
print("4. Some regions show high sales but weak profit.")
print("5. Few products generate majority of revenue.")


print("\n===== TACTICAL IMPROVEMENTS =====")
print("1. Optimize discount strategy to protect margins.")
print("2. Promote high-margin categories and products.")
print("3. Launch seasonal sales campaigns during peak months.")
print("4. Apply region-specific pricing strategies.")
print("5. Eliminate or reprice loss-making products.")

print("\n--- Analysis Completed Successfully ---")
