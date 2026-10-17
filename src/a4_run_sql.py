"""Part A, step 4: run the seven queries in sql/queries.sql and print the reported results."""
import re, sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d) .*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
assert len(queries) == 7

with sqlite3.connect("data/instamart.db") as con:
    r = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

print("Q1 customers:", len(r["Q1"]))
q2 = r["Q2"]
print("Q2 rows:", len(q2), "sum:", round(q2.revenue.sum(), 2), "largest:", q2.iloc[0].to_dict())
print("Q3 products:", len(r["Q3"]))
print("Q4 customers without completed order:", len(r["Q4"]))
q5 = r["Q5"]
print("Q5 customers above average:", len(q5), "average:", q5.avg_total.iloc[0])
q6 = r["Q6"]
print("Q6 top per city:")
print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with rank != row_number:", int((q6.rnk != q6.row_num).sum()))
q7 = r["Q7"]
print("Q7 rows:", len(q7), "last cumulative:", q7.cumulative_revenue.iloc[-1])
