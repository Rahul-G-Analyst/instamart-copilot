"""Part B, tasks 4-8: lapse problem, leakage-free preprocessing, tuning, balance experiment, interpretation,
persistence and the retention list. Reads data/clean/*.csv and outputs/customer_segments.csv."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")          # exclusive end of the label window (September)

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id")

# ---- task 4: cohort, label, features (only orders strictly before the cutoff) -------------------
hist = orders[orders["order_ts"] < CUTOFF]                                  # leakage guard: nothing on/after cutoff
assert hist["order_ts"].max() < CUTOFF
hist_comp = hist[hist["status"] == "completed"]
final = hist[hist["status"].isin(["completed", "cancelled"])]
g = hist_comp.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
})
feat["cancel_rate"] = (final["status"] == "cancelled").groupby(final["customer_id"]).mean().reindex(feat.index)
fav = (hist_comp.groupby(["customer_id", "category"]).size().rename("n").reset_index()
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"])
feat["fav_category"] = fav
feat = feat.join(cust.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

sept = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]
active_in_sept = set(sept["customer_id"])
y = pd.Series((~feat.index.isin(active_in_sept)).astype(int), index=feat.index, name="lapsed")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X = feat[NUM + CAT]
assert not any(c.startswith("segment") or c == "is_premium" for c in X.columns), "segment labels must not be features"
print(f"cohort={len(X)} lapsed={int(y.sum())} ({y.mean()*100:.2f}%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train={len(X_train)} test={len(X_test)}  train lapsed={int(y_train.sum())} test lapsed={int(y_test.sum())}")

# ---- task 5: preprocessing fitted on training rows only -----------------------------------------
def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT)])

print("missing values in feature matrix:", int(X.isna().sum().sum()))
assert X.isna().sum().sum() == 0
prep_demo = make_prep().fit(X_train)
learned = prep_demo.named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"scaler mean of frequency (train rows only) = {learned:.4f}; full-cohort mean = {X['frequency'].mean():.4f}")

def metrics(model, Xs, ys):
    p = model.predict(Xs)
    return {"precision": precision_score(ys, p, zero_division=0), "recall": recall_score(ys, p, zero_division=0),
            "f1": f1_score(ys, p, zero_division=0), "cm": confusion_matrix(ys, p)}

def show(name, m):
    print(f"{name}: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}  "
          f"confusion [[TN FP],[FN TP]] = {m['cm'].tolist()}")

# ---- task 6: baseline and tuning ---------------------------------------------------------------
base_pred = np.zeros(len(y_test), dtype=int)
print(f"\nbaseline (always not lapsed): accuracy={accuracy_score(y_test, base_pred)*100:.1f}% "
      f"F1(lapsed)={f1_score(y_test, base_pred, zero_division=0):.3f}")

lr_pipe = Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_pipe = Pipeline([("prep", make_prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_train, y_train)
print(f"LogReg best params={lr_gs.best_params_} CV F1={lr_gs.best_score_:.3f}")
print(f"RandomForest best params={rf_gs.best_params_} CV F1={rf_gs.best_score_:.3f}")

lr_m, rf_m = metrics(lr_gs.best_estimator_, X_test, y_test), metrics(rf_gs.best_estimator_, X_test, y_test)
show("LogReg  test", lr_m)
show("RandForest test", rf_m)

# Selection uses cross-validated F1 only (the test set is not used to choose).
best_is_rf = rf_gs.best_score_ >= lr_gs.best_score_
selected, sel_name = (rf_gs.best_estimator_, "RandomForest") if best_is_rf else (lr_gs.best_estimator_, "LogisticRegression")
print("selected model:", sel_name, "(higher cross-validated F1)")
assert metrics(selected, X_test, y_test)["f1"] >= 0.55

# ---- task 7: class-balance experiment (train only) ----------------------------------------------
n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
extra = resample(X_train[y_train == 1], replace=True, n_samples=n0 - n1, random_state=42)
X_bal = pd.concat([X_train, extra]); y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
print(f"\nbalanced training set: class0={int((y_bal==0).sum())} class1={int((y_bal==1).sum())}")
lr_bal = clone(lr_gs.best_estimator_).fit(X_bal, y_bal)
bal_m = metrics(lr_bal, X_test, y_test)
show("LogReg unbalanced", lr_m)
show("LogReg oversampled", bal_m)

# ---- task 8: interpretation, persistence, retention list ----------------------------------------
rf_best = rf_gs.best_estimator_
names = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop-5 random-forest importances:\n", imp.head(5).round(4).to_string())
lr_best = lr_gs.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["prep"].get_feature_names_out())
print("\nLogReg coefficients (standardised numeric features):\n", coef.round(3).sort_values().to_string())

joblib.dump(selected, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg.loc[seg["is_premium"], "customer_id"])
proba = pd.Series(selected.predict_proba(X_test)[:, 1], index=X_test.index)
ret = pd.DataFrame({"customer_id": X_test.index, "lapse_probability": proba.values.round(4),
                    "actually_lapsed": y_test.values})
ret = ret[ret["customer_id"].isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
print("\nretention list (illustrative, Premium flag is a whole-window label):")
print(ret.to_string(index=False))
print(f"truly lapsed among the {len(ret)} listed: {int(ret['actually_lapsed'].sum())}")
