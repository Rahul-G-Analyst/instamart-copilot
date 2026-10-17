"""Part A, step 2: clean customers and orders, write CSVs and SQLite database."""
import os, sqlite3
import numpy as np
import pandas as pd

RAW = "data/raw/"
os.makedirs("data/clean", exist_ok=True)

# ---------------- customers ----------------
cust = pd.read_csv(RAW + "customers.csv")
cust["city"] = cust["city"].str.strip().str.title()
g = cust["gender"].fillna("").str.strip().str.lower()
cust["gender"] = g.map({"male": "male", "m": "male", "female": "female", "f": "female"}).fillna("unknown")
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
n_bad_age = int(((cust["age"] < 10) | (cust["age"] > 100)).sum())
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
n_missing_age = int(cust["age"].isna().sum())
cust["age"] = cust["age"].fillna(cust.groupby("city")["age"].transform("median"))
cust["age"] = cust["age"].round().astype(int)          # city medians may be x.5; whole years
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
print(f"customers: {n_bad_age} impossible ages set missing; {n_missing_age} ages filled with city median")

# ---------------- products ----------------
prod = pd.read_csv(RAW + "products.csv")

# ---------------- orders ----------------
o = pd.read_csv(RAW + "orders.csv")
n0 = len(o)
o["order_ts"] = pd.to_datetime(o["order_ts"])                                   # rule 1
o = o.drop_duplicates()                                                         # rule 2
n_exact = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")   # rule 3
n_partial = n1 - len(o)
amt = pd.to_numeric(o["order_amount"], errors="coerce")                         # rule 4
n_unparse = int(amt.isna().sum())
o["order_amount"] = amt
price = o["product_id"].map(prod.set_index("product_id")["price_inr"])
calc = (price * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(calc)
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")   # rule 5
n_neg = int((o["delivery_minutes"] < 0).sum())
o.loc[o["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
o["category"] = o["product_id"].map(prod.set_index("product_id")["category"])
comp = o["status"] == "completed"
med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
n_fill = int((o["delivery_minutes"].isna() & comp).sum())
o["delivery_minutes"] = o["delivery_minutes"].where(~(o["delivery_minutes"].isna() & comp), med)
o = o.drop(columns="category").sort_values("order_id").reset_index(drop=True)
print(f"orders: rule2 removed {n_exact} exact duplicates; rule3 removed {n_partial} superseded records; "
      f"rule4 filled {n_unparse} amounts; rule5 set {n_neg} negatives missing, filled {n_fill} completed delivery times")

# ---------------- store ----------------
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = cust.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv("data/clean/customers.csv", index=False)
prod.to_csv("data/clean/products.csv", index=False)
o_out.to_csv("data/clean/orders.csv", index=False)
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
with sqlite3.connect("data/instamart.db") as con:
    c_out.to_sql("customers", con, index=False)
    prod.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
print("stored: data/clean/*.csv and data/instamart.db")
print(cust["city"].value_counts().to_dict(), cust["gender"].value_counts().to_dict(),
      int(cust["age"].min()), int(cust["age"].max()))
print(len(o), o["status"].value_counts().to_dict(),
      round(o.loc[o.status == "completed", "order_amount"].sum(), 2))
