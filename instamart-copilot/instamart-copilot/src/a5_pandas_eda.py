"""Part A, step 5: pandas cross-check of Q2 and exploratory statistics of completed order_amount."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                 .merge(p[["product_id", "category"]], on="product_id")

pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, margins_name="Total")
print("pivot grand total:", round(pv.loc["Total", "Total"], 2))
print(pv.round(2).to_string())

by_city = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                   n=("order_amount", "count")).round(2)
print(by_city.to_string())

a = comp["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"mean {a.mean():.2f} median {a.median():.2f} skew {a.skew():.2f} Q1 {q1:.2f} Q3 {q3:.2f} "
      f"fences {lo:.2f}/{hi:.2f} below {int((a < lo).sum())} above {int((a > hi).sum())}")
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
import numpy as np
mask = ~np.eye(len(corr), dtype=bool)
pairs = corr.abs().where(mask).stack()
best = pairs.idxmax()
print("largest |corr| pair:", best, round(float(corr.loc[best]), 3))
