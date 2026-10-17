"""Part A, step 3: assert the reference checkpoints on the clean data and the database."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
comp = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

checks = [
    ("orders rows", len(o) == 12309),
    ("one row per order_id", o["order_id"].is_unique),
    ("completed count", len(comp) == 11110),
    ("cancelled count", len(canc) == 1199),
    ("no pending rows", (o["status"] == "pending").sum() == 0),
    ("no missing order_amount", o["order_amount"].isna().sum() == 0),
    ("no missing delivery among completed", comp["delivery_minutes"].isna().sum() == 0),
    ("missing delivery among cancelled", canc["delivery_minutes"].isna().sum() == 1199),
    ("completed revenue", round(comp["order_amount"].sum(), 2) == 2998668.80),
    ("six cities", c["city"].nunique() == 6),
    ("city counts", c["city"].value_counts().to_dict() ==
        {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("no missing age", c["age"].isna().sum() == 0),
    ("age range 18-60", c["age"].between(18, 60).all()),
]
with sqlite3.connect("data/instamart.db") as con:
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    checks.append(("db tables", {"customers", "products", "orders"} <= tables))
    checks.append(("db revenue", round(con.execute(
        "select sum(order_amount) from orders where status='completed'").fetchone()[0], 2) == 2998668.80))

failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
assert not failed, f"failed: {failed}"
print("ALL CHECKPOINTS PASSED")
