
import os
import glob
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


sns.set(style="whitegrid")
plt.rcParams["figure.autolayout"] = True


print("Loading dataset...")

def find_csv_in_dir(directory):
    pattern = os.path.join(directory, "*.csv")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None

def resolve_dataset_path(path_arg=None):
    
    if path_arg:
        if os.path.isdir(path_arg):
            found = find_csv_in_dir(path_arg)
            if not found:
                raise FileNotFoundError(f"No CSV files found in directory: {path_arg}")
            return found
        if os.path.isfile(path_arg):
            return path_arg
        
        alt = os.path.join(os.path.dirname(__file__), path_arg)
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"Dataset path not found: {path_arg}")

def load_dataset(path_arg=None):
    
    if path_arg is None:
        default_dir = os.path.join(os.path.dirname(__file__), "Datasets")
        if os.path.isdir(default_dir):
            csv = find_csv_in_dir(default_dir)
            if csv:
                return pd.read_csv(csv, low_memory=False)
            raise FileNotFoundError(f"No CSV files found in default directory: {default_dir}")
    resolved = resolve_dataset_path(path_arg)
    return pd.read_csv(resolved, low_memory=False)

df = None
try:
    
    if "__file__" in globals():
        df = load_dataset()
except Exception:
    
    df = None

if df is not None:
    print("Dataset loaded successfully\n")


def process(df):
    
    print("Original Columns in Dataset:")
    print(df.columns, "\n")

    
    df.columns = df.columns.str.strip()

    
    lower_cols = {c.lower(): c for c in df.columns}

    def find_col_by_substrings(subs_any=None, subs_all=None):
        subs_any = subs_any or []
        subs_all = subs_all or []
        for low, orig in lower_cols.items():
            if subs_all and not all(s in low for s in subs_all):
                continue
            if subs_any and not any(s in low for s in subs_any):
                continue
            return orig
        return None

    
    cust = find_col_by_substrings(subs_all=["customer", "id"]) or find_col_by_substrings(subs_any=["customer id","customerid","customer_id"]) or find_col_by_substrings(subs_any=["customer"]) 
    if cust:
        df.rename(columns={cust: "Customer_ID"}, inplace=True)

    date_col = find_col_by_substrings(subs_any=["date","purchase date","invoice date"]) or find_col_by_substrings(subs_any=["date"]) 
    if date_col:
        df.rename(columns={date_col: "Transaction_Date"}, inplace=True)

    
    lower_cols = {c.lower(): c for c in df.columns}

    
    trans_id_col = find_col_by_substrings(subs_any=["invoice","transaction","order","order id","invoice no","order_no"]) 
    if trans_id_col and trans_id_col not in ("Customer_ID","Transaction_Date"):
        df.rename(columns={trans_id_col: "Transaction_ID"}, inplace=True)

    
    unit_price_col = find_col_by_substrings(subs_any=["unitprice","unit price","product price","price"]) or find_col_by_substrings(subs_any=["price"]) 
    if unit_price_col:
        df.rename(columns={unit_price_col: "UnitPrice"}, inplace=True)

    
    if "Transaction_Amount" not in df.columns:
        
        total_col = None
        
        total_col = find_col_by_substrings(subs_all=["total","amount"]) or find_col_by_substrings(subs_any=["total purchase amount","totalpurchaseamount","total_purchase_amount"]) or find_col_by_substrings(subs_any=["total","amount"]) 
        if total_col and total_col not in ("Customer_ID","Transaction_Date","Transaction_ID"):
            df["Transaction_Amount"] = df[total_col]
        elif "Quantity" in df.columns and "UnitPrice" in df.columns:
            df["Transaction_Amount"] = df["Quantity"] * df["UnitPrice"]
        elif "Amount" in df.columns:
            df["Transaction_Amount"] = df["Amount"]
        else:
            raise Exception(f"Transaction amount columns not found in dataset. Available columns: {list(df.columns)}")

    
    df.drop_duplicates(inplace=True)
    df = df[df["Transaction_Amount"] > 0]
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])

    print("Data cleaning completed\n")

    
    print("Performing RFM analysis...")
    snapshot_date = df["Transaction_Date"].max() + pd.Timedelta(days=1)

    
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = np.arange(len(df))

    rfm = df.groupby("Customer_ID").agg({
        "Transaction_Date": lambda x: (snapshot_date - x.max()).days,
        "Transaction_ID": "count",
        "Transaction_Amount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    print("RFM table created\n")

    
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"], 4, labels=[1,2,3,4])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])
    rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].astype(int).sum(axis=1)

    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    print("Customer segmentation completed\n")

    
    numeric_cols = rfm.select_dtypes(include=[np.number]).columns.tolist()
    segment_profile = rfm.groupby("Cluster")[numeric_cols].mean()
    print("Segment Profile:")
    print(segment_profile, "\n")

    
    plt.figure()
    sns.countplot(x="Cluster", data=rfm)
    plt.title("Customer Segment Distribution")
    plt.savefig("segment_distribution.png")
    plt.close()

    
    plt.figure(figsize=(8,6))
    sns.heatmap(segment_profile, annot=True, cmap="coolwarm")
    plt.title("RFM Segment Profile")
    plt.savefig("rfm_heatmap.png")
    plt.close()

    
    plt.figure(figsize=(10,5))
    df.groupby(df["Transaction_Date"].dt.to_period("M"))["Transaction_Amount"].sum().plot()
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue")
    plt.savefig("monthly_revenue.png")
    plt.close()

    
    churn_threshold = rfm["Recency"].quantile(0.75)
    rfm["Churn_Risk"] = np.where(rfm["Recency"] > churn_threshold, "High Risk", "Low Risk")

    print("Churn Risk Distribution:")
    print(rfm["Churn_Risk"].value_counts(), "\n")

    
    recommendations = [
        "1. Reward loyal and high-value customers using exclusive loyalty programs.",
        "2. Upsell premium products to frequent but low-spending customers.",
        "3. Launch onboarding campaigns for newly acquired customers.",
        "4. Provide win-back discounts to customers with high churn risk.",
        "5. Use RFM-based personalization for targeted marketing."
    ]

    print("Actionable Recommendations for Alfido Tech:")
    for rec in recommendations:
        print(rec)


def main():
    parser = argparse.ArgumentParser(description="Run customer behavior analysis")
    parser.add_argument("--file", "-f", help="Path to CSV file or directory containing CSVs", default=None)
    args = parser.parse_args()

    df_to_use = df
    try:
        if args.file:
            df_to_use = load_dataset(args.file)
        elif df_to_use is None:
            
            df_to_use = load_dataset(None)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    try:
        process(df_to_use)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()