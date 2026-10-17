"""Part B, tasks 1-3: RFM table, K-means selection, segment profile/names, PCA, outputs/customer_segments.csv."""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = orders[orders["status"] == "completed"]

# ---- task 1: RFM -------------------------------------------------------------------------------
g = comp.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
})
print("== RFM table")
print(f"customers={len(rfm)}  recency median={rfm.recency.median():.0f} min={rfm.recency.min()} max={rfm.recency.max()}")
print(f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}  "
      f"monetary mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}")

# ---- task 2: K-means for K = 2..8 ---------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows)
print("\n== K-means table")
print(ktab.to_string(index=False))

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---- task 3: profile, names, premium flag --------------------------------------------------------
prof = rfm.groupby("segment_id").agg(size=("monetary", "size"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)
premium_id = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
rest = rest.drop(index=loyal_id)
dormant_id = rest["mean_recency"].idxmax()
occasional_id = rest.drop(index=dormant_id).index[0]
NAMES = {premium_id: ("Premium Big-Basket", "Protect: early access to Electronics launches, a priority-support line and personal win-back calls."),
         loyal_id: ("Loyal Regulars", "Grow: subscription bundles and loyalty points to raise basket size without discounts."),
         dormant_id: ("Dormant Lapsers", "Win back: time-limited comeback coupon plus a reminder campaign before they churn for good."),
         occasional_id: ("Occasional Shoppers", "Nudge: free-delivery thresholds and category cross-sell to lift order frequency.")}
rfm["segment_name"] = rfm["segment_id"].map(lambda s: NAMES[s][0])
rfm["is_premium"] = rfm["segment_id"] == premium_id
prof["segment_name"] = [NAMES[s][0] for s in prof.index]
prof["is_premium"] = [s == premium_id for s in prof.index]
prof["action"] = [NAMES[s][1] for s in prof.index]
print("\n== segment profile (K=%d); cluster numbers are arbitrary labels with no ordering" % FINAL_K)
print(prof.to_string())

pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print("\n== PCA explained_variance_ratio_:", evr.round(3), "total", round(evr.sum(), 3))

out = rfm.reset_index()[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
out["monetary"] = out["monetary"].round(2)
out.to_csv("outputs/customer_segments.csv", index=False)
print("\nwrote outputs/customer_segments.csv rows =", len(out), " premium customers =", int(out.is_premium.sum()))
