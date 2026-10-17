"""Part A, step 1: audit the raw tables (read-only)."""
import pandas as pd

RAW = "data/raw/"
# Only truly empty cells count as missing: the literal text "N/A" in order_amount is a planted bad
# value, not a blank, so default NA parsing must be switched off for orders and tickets.
def load(n):
    if n == "tickets":
        return pd.read_csv(RAW + f"{n}.csv", keep_default_na=False)
    if n == "orders":
        return pd.read_csv(RAW + f"{n}.csv", keep_default_na=False, na_values=[""])
    return pd.read_csv(RAW + f"{n}.csv")
tables = {n: load(n) for n in ["customers", "products", "orders", "tickets"]}

for name, df in tables.items():
    print(f"\n=== {name}: shape={df.shape}")
    summary = pd.DataFrame({"dtype": df.dtypes.astype(str),
                            "missing": df.isna().sum(),
                            "missing_pct": (df.isna().mean() * 100).round(2)})
    print(summary.to_string())

o, c = tables["orders"], tables["customers"]
amount_num = pd.to_numeric(o["order_amount"], errors="coerce")
deliv = pd.to_numeric(o["delivery_minutes"], errors="coerce")
age = pd.to_numeric(c["age"], errors="coerce")
audit = {
    "raw order rows": len(o),
    "exact duplicate rows": int(o.duplicated().sum()),
    "pending rows": int((o["status"] == "pending").sum()),
    "distinct order_id": int(o["order_id"].nunique()),
    "non-numeric order_amount": int((amount_num.isna() & o["order_amount"].notna()).sum()),
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(deliv.isna().sum()),
    "blank age": int(age.isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": int(c["city"].nunique()),
    "distinct non-blank raw gender spellings": int(c["gender"].dropna().nunique()),
    "blank gender": int(c["gender"].isna().sum()),
}
print("\n=== audit figures")
for k, v in audit.items():
    print(f"{k:42s} {v}")
