# src/dc_view.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gas_price_boxplot(df):
    if "address" not in df.columns or "lock_gas_price" not in df.columns:
        raise ValueError("DataFrame must contain 'address' and 'lock_gas_price' columns.")
    df = df.copy()
    df["short_addr"] = df["address"].astype(str).str[:6] + "..." + df["address"].astype(str).str[-4:]
    plt.figure(figsize=(max(8, len(df["short_addr"].unique()) * 0.8), 6))
    sns.boxplot(x="short_addr", y="lock_gas_price", data=df)
    plt.xlabel("address (shortened)")
    plt.ylabel("lock_gas_price (gwei)")
    plt.title("lock_gas_price by address")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def compute_gas_group_stats(df):
    group_stats = df.groupby("address")["lock_gas_price"].agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()
    group_stats["best_range"] = group_stats.apply(lambda r: f"{r.q25:.2f}~{r.q75:.2f}", axis=1)
    return group_stats

def filter_good_suppliers(group_stats, min_count=10, max_std=5):
    return group_stats[(group_stats["count"] >= min_count) & (group_stats["std"] <= max_std)]
