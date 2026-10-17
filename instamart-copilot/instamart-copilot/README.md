# Instamart Customer Intelligence & Support Copilot

## Overview

An internal tool built in four connected Parts on synthetic, fictional data (the brand name only frames the scenario). All money is in Indian Rupees (INR).

| Part | What it does | Main files |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, answers seven SQL questions and cross-checks in pandas | `src/a1..a5`, `sql/queries.sql`, `data/clean/`, `data/instamart.db` |
| B | RFM segments (K-means) and a lapse-prediction model | `src/b1_segments.py`, `src/b2_lapse_model.py`, `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Support-ticket triage with privacy, retry, validation, budget and routing guardrails | `src/c_triage.py`, `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Policy RAG assistant and a tool-calling refund agent with safe stopping and crash recovery | `src/d1_rag.py`, `src/d2_agent.py`, `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Every model call goes through the supplied offline `mock_llm.py` (no key, no network). `generate_data.py` and `mock_llm.py` are committed unmodified. Libraries: standard library, pandas, numpy, scikit-learn, joblib. Everything runs from the repository root on branch `main`.

## Reproduce

Python 3.11 or newer. Run from the repository root, in this order:

1. `python -m venv .venv`
2. `source .venv/bin/activate`   (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`   (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/a1_audit.py`
6. `python src/a2_clean_store.py`
7. `python src/a3_validate.py`   (prints `ALL CHECKPOINTS PASSED`)
8. `python src/a4_run_sql.py`
9. `python src/a5_pandas_eda.py`
10. `python src/b1_segments.py`
11. `python src/b2_lapse_model.py`
12. `python src/c_triage.py`
13. `python src/d1_rag.py`
14. `python src/d2_agent.py`

Every script is deterministic (seeds 42 throughout). The only run-to-run differences are the timestamps in the ledger files. `requirements.txt` is a `pip freeze` of the environment used.

---

## Part A

### Audit of the raw data

| Table | Shape | Missing values (count, %) |
|---|---|---|
| customers | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%); customer_id, city, signup_date 0 |
| products | 40 x 4 | none |
| orders | 13,084 x 9 | delivery_minutes 2,541 (19.42%); all other columns 0 (the 120 bad amounts are text such as `N/A`/`error`, not blanks) |
| tickets | 164 x 5 | none when read with `keep_default_na=False` (4 tickets have an empty-string text) |

Data types as loaded by pandas: strings for ids, city, gender, status, timestamps and (in orders) `order_amount`; integers for quantity, discount_pct, price_inr; floats for age and delivery_minutes.

| Audit figure | Value | Reference |
|---|---|---|
| raw orders.csv rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| rows with status pending | 656 | 656 |
| distinct order_id | 12,309 | 12,309 |
| non-numeric order_amount | 120 | 120 |
| negative delivery_minutes | 106 | 106 |
| blank delivery_minutes | 2,541 | 2,541 |
| blank age | 61 | 61 |
| age outside 10–100 | 6 | 6 |
| distinct raw city spellings | 24 | 24 |
| distinct non-blank raw gender spellings | 10 | 10 |
| blank gender | 30 | 30 |

### Cleaning rules and what each changed

Customers
1. City: strip spaces + title case → exactly six cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184).
2. Gender mapped to male / female / unknown (blank → unknown): female 610, male 560, unknown 30.
3. Ages outside 10–100 set to missing (6 values), then all 67 missing ages (61 blank + 6) filled with the median age of the customer's city via `groupby().transform("median")` (rounded to whole years). 0 missing, range 18–60.
4. `signup_date` converted to datetime.

Orders (in this order)
1. `order_ts` converted to datetime.
2. Exact duplicates removed: **119 rows**.
3. For each `order_id`, only the record with the latest `order_ts` is kept: **656 superseded earlier `pending` rows removed** → 12,309 rows, one per order.
4. `order_amount` converted to numeric (unparseable → missing): **112 missing amounts** (120 raw bad values, 8 of them sat on rows already removed) refilled as `round(price_inr × quantity × (1 − discount_pct/100), 2)`.
5. Negative `delivery_minutes` set to missing (**105** after de-duplication; 106 raw). For completed orders, every missing delivery time (**774** values) is filled with the median of that order's product category among completed orders (`groupby().transform`). Cancelled orders stay missing (1,199), because a never-delivered order has no delivery time to invent.

Result: 12,309 rows; 11,110 completed and 1,199 cancelled; no pending; 0 missing amounts; 0 missing delivery among completed; 1,199 missing among cancelled; completed revenue **INR 2,998,668.80**. All reference checkpoints match.

### Storage and validation

`data/clean/{customers,products,orders}.csv` and `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/a3_validate.py` asserts 16 checkpoints (row counts, status counts, revenue, city and gender counts, age range, table presence, DB revenue) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run by `src/a4_run_sql.py` with `pd.read_sql_query`)

| Query | Result | Reference |
|---|---|---|
| Q1 WHERE + HAVING (completed, amount > 500, customer total > 8000) | 30 customers | 30 |
| Q2 three-table join, revenue by city × category | 30 rows, total INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 | same |
| Q3 scalar subquery, price above catalogue average | 11 products | 11 |
| Q4 LEFT JOIN, customers without a completed order | 25 customers | 25 |
| Q5 two CTEs, spend above average | 550 customers; average INR 2,552.06 | same |
| Q6 RANK / DENSE_RANK / ROW_NUMBER per city | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows with RANK ≠ ROW_NUMBER | same |
| Q7 daily revenue with running total | 183 daily rows; last cumulative INR 2,998,668.80 | same |

Why ties cause rank ≠ row number: `RANK()` gives tied customers the same rank and then skips the following numbers, whereas `ROW_NUMBER()` always counts 1, 2, 3… and breaks the tie arbitrarily, so the two disagree for the tied rows and every row after them.

### pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` grand total = **INR 2,998,668.80**, identical to the SQL total. Per-city completed `order_amount` (INR):

| City | Sum | Mean | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences −283.50 / 645.30, outliers 0 below and **897 above**. Correlation matrix (order_amount, discount_pct, quantity, delivery_minutes): largest |r| is order_amount vs discount_pct = **−0.095**; order_amount vs quantity = +0.087; all others |r| ≤ 0.013.

**Analysis (under 200 words).** *What is this data?* 11,110 completed quick-commerce orders (INR 2,998,668.80) from 1,175 customers in six cities, April–September 2025. *Distributions:* order_amount is strongly right-skewed (mean 269.91 far above median 135.00, skewness 4.61); delivery time has median 25 minutes with a long tail to 75. *Outliers:* 897 orders exceed the 645.30 fence and none fall below; 635 of them are Electronics, so they are genuine high-ticket purchases rather than errors (the planted bad amounts and delivery times were fixed earlier). *Relationships:* every correlation is weak; the largest in absolute value is order_amount with discount_pct (−0.095), just ahead of order_amount with quantity (+0.087). A correlation this small explains under 1% of the variance, and correlation alone does not establish causation.

---

## Part B

Only completed orders are used for behaviour. Snapshot for RFM: 2025-10-01 00:00:00.

### RFM table
1,175 customers. Recency median 19 (min 0, max 180); frequency mean 9.46 (max 33); monetary mean 2,552.06 (max 14,766.55). All match the reference.

### K-means (StandardScaler, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The elbow is at K=4 (inertia falls 41% from K=3 to K=4, then only 10–19% per step), K=4 also has the highest silhouette (0.526), and four segments is a number the business can run distinct campaigns for. Inertia at K=8 is always lower than at K=2 because it is the within-cluster sum of squares and any extra centre can only reduce it (with K = number of customers it would be 0); so a lower inertia alone says nothing about whether the clusters are meaningful, which is why the elbow shape, the silhouette and interpretability decide.

### Segment profile (cluster numbers are only labels; they carry no order)

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | no | Free-delivery thresholds and category cross-sell to lift frequency |
| 1 | Premium Big-Basket | 138 | 24.12 | 6.77 | 7,129.70 | **yes** | Early access to Electronics launches, priority support, personal win-back calls |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | no | Subscription bundles and loyalty points to grow basket size without discounts |
| 3 | Dormant Lapsers | 182 | 109.04 | 2.20 | 685.99 | no | Time-limited comeback coupon and reminder campaign |

Premium is assigned programmatically to the segment with the highest mean monetary value (segment names are derived from the profile by rule, not from the cluster number).

**PCA** (2 components on the scaled RFM): explained variance ratio 0.634 and 0.218, total **0.852**. Written to `outputs/customer_segments.csv` (columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### Lapse problem
Cutoff 2025-09-01. Cohort = customers with at least one completed order before the cutoff: **1,156** customers, **443 lapsed (38.32%)** (no completed order 2025-09-01 to 2025-09-30). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train / 232 test** (354 / 89 lapsed). Baseline "always not lapsed" on the test set: accuracy **61.6%**, lapsed-class F1 **0.000**.

Features (11, all from orders dated before the cutoff plus the customer table): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days.

**No order dated 2025-09-01 or later entered any feature** (asserted in code), and **the segment labels from Part B task 3 were not used as features** (also asserted).

### Preprocessing without leakage
A scikit-learn `Pipeline` with `ColumnTransformer`: `StandardScaler` on the eight numeric features and `OneHotEncoder(drop="first")` (k−1 columns) on city, gender, favourite category. **No encoder, scaler or resampler saw test rows**: inside every pipeline they are fitted on training rows only (and, inside `GridSearchCV`, on the training part of each fold). The feature matrix has 0 missing values (asserted). The scaler's learned mean of frequency is **8.1147** versus **8.0285** for the full cohort; they differ because the scaler only saw the 924 training rows, and the full-cohort mean also includes the 232 test rows.

### Models (lapsed class is positive)

| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression (C ∈ {0.01, 0.1, 1, 10, 100}, cv=5) | C = 100 (edge of the grid) | 0.651 |
| Random forest (depth ∈ {3,5,8} × leaf ∈ {5,10,20}, 100 trees, cv=5, 45 fits) | max_depth 8, min_samples_leaf 10 | 0.681 |

Untouched test set (232 rows):

| Model | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Baseline (never lapsed) | 143 | 0 | 89 | 0 | 0 | 0 | 0.000 |
| Logistic regression | 111 | 32 | 31 | 58 | 0.644 | 0.652 | 0.648 |
| Random forest | 113 | 30 | 31 | 58 | 0.659 | 0.652 | 0.655 |

**Choice: random forest** (highest cross-validated F1, chosen before looking at the test set; test F1 0.655 ≥ 0.55). Missing a customer who is about to lapse loses all their future orders, while wrongly targeting a loyal customer costs only a small coupon or message, so recall matters more than precision here; both models catch the same 58 of 89 lapsing customers, and the forest produces two fewer false alarms and captures non-linear effects. The test-set difference between the two models is only one or two customers, so the choice rests mainly on cross-validation. If the team wants to catch more lapsers, the oversampled model below is the lever.

### Class-balance experiment
Over-sampling the training set only (duplicates of lapsed rows drawn with replacement, `random_state=42`) gave 570 / 570 rows. Refitting the best logistic regression and evaluating on the original test set:

| Logistic regression | Precision | Recall | F1 | TP | FP |
|---|---|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 | 58 | 32 |
| Over-sampled | 0.589 | **0.854** | 0.697 | 76 | 53 |

**Recall moved most** (+0.20); precision did not rise (it fell by 0.055). For the retention team this means catching 76 instead of 58 of the 89 lapsing customers at the price of 21 more wrongly targeted customers, which is attractive when a retention offer is cheap compared with a lost customer.

### Interpretation, persistence, retention list
Top five random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, average delivery minutes 0.061, cancellation rate 0.053 (how often and how much someone buys drives lapse far more than city or demographics).

Logistic-regression coefficients (standardised features): frequency **−1.70** (more orders → much less likely to lapse), monetary **+0.28** (for the same number of orders, bigger baskets go with infrequent big-ticket buyers who lapse more), average delivery minutes **+0.08** (slightly slower deliveries, slightly more lapse; weak). Favourite category Electronics (−1.14) is the largest category effect.

The selected model (a full pipeline, preprocessing included) is saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading it with `joblib.load` reproduces the original test predictions **exactly** (asserted).

`outputs/retention_list.csv` holds the top 10 Premium-segment test customers ranked by predicted lapse probability (0.452–0.487); **1 of the 10 truly lapsed** (C0525). The Premium flag comes from the whole-window RFM, which includes September, the window that defines the label, so this list is illustrative and not a leakage-free evaluation.

---

## Part C

`src/c_triage.py` wraps `mock_llm.py` with guardrails. Tickets are processed in file order.

**Input guardrails.** 4 empty tickets → `skipped_empty`; 4 long tickets (est_tokens 734 > `MAX_INPUT_TOKENS` 400) → `flagged_long`. Neither is ever sent to the model.

**Redaction** (regexes and order exactly as specified; the user message is exactly the redacted text):

| Type | Substitutions |
|---|---|
| email → `[EMAIL]` | 55 |
| phone → `[PHONE]` | 91 |
| order id → `[ORDER_ID]` | 132 |
| **Total** | **278** |

Re-applying the three patterns to the redacted texts finds **0** matches. Every prompt is also asserted PII-free immediately before each call (175 prompts checked per full run).

**Prompts.** Zero-shot (est. 41 tokens) and few-shot (est. 124 tokens, with an `EXAMPLES:` line and four labelled examples including a fraud one); both contain `CLASSIFY`, list the four categories and demand JSON `{"category": "...", "reason": "..."}`.

**Retries.** Each call is wrapped with up to 3 retries and wait `0.05 × 2**n × uniform(0.9, 1.15)` seconds, one log line per retry (for example `[retry] ticket=T0002 retry=0 wait=0.0530s`). A full run had exactly **19 TransientLLMErrors, all recovered at retry 0**, with waits between 0.0453 and 0.0562 s (allowed 0.045–0.0575 s); no ticket ended in a permanent error. `mock_llm.reset()` is called before each run.

**Validation.** Valid JSON with an allowed category → `classified_json`; otherwise the lowercase words (`re.findall(r"[a-z]+", ...)`) are searched and exactly one allowed word → `classified_keyword`; anything else, including valid JSON with a category outside the list (`shipping`), → `flagged_invalid_output`, sent to a human, never guessed. Full-run status counts: **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output** (plus 4 skipped_empty, 4 flagged_long).

### Zero-shot versus few-shot (no budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 10,804 | 73.5 |
| Few-shot | 147 | 100% (147/147) | 23,752 | 161.6 |

Zero-shot confusion (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | **11** | 0 | 23 | 0 |
| technical | **10** | 0 | 0 | 29 |

The few-shot prompt costs about 2.2 times as many tokens (161.6 versus 73.5 per ticket, +12,948 tokens in total), roughly 617 extra tokens for each of the 21 tickets it fixes. That is worth paying when a mistake is expensive: 11 of the zero-shot errors are fraud tickets sent to billing, and fraud must reach the security team within 2 hours. It is not worth it if all four categories were equally harmless to misroute and the budget were tight.

### Budget, routing and output
* **Budget demo:** zero-shot with `TOKEN_BUDGET = 6000` → **69 tickets `budget_exhausted`** (6,033 tokens spent; the budget is checked before each call, so the last call may overshoot slightly). Saved to `outputs/triage_budget_demo.csv`.
* **Final run:** few-shot with `TOKEN_BUDGET = 100000` → `outputs/triage_results.csv` (164 rows; columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used).

| Route | Tickets | Priority |
|---|---|---|
| support_queue | 102 | NORMAL |
| security_team (fraud) | 34 | HIGH |
| premium_desk | 11 | HIGH |
| human_review | 17 (9 invalid output + 4 empty + 4 long) | NORMAL |

**Rule-validated vs. human:** 147 tickets (135 JSON + 12 keyword) passed rule validation and were routed automatically; 17 went to human review. Every ticket's customer had a row in the segment file from Part B (none treated as not-premium by default).

---

## Part D

### Chunking and retrieval
40-word chunks, 10-word overlap: **15 chunks** (refund 4, delivery 3, escalation 3, payment 3, privacy 2). `retrieve("refund policy late delivery")` returns three chunks, all from `refund_policy.txt`. `retrieve(query, k=3)` returns dictionaries with `source`, `score`, `text` using `TfidfVectorizer()` fitted on the chunks and cosine similarity.

*TF and IDF.* Term frequency measures how often a word occurs in a chunk; inverse document frequency down-weights words that appear in many chunks, so rare, informative words ("approved", "lead") count more than common ones ("refund"). A limitation: TF-IDF only matches the same words, so a question phrased with different words but the same meaning ("money back" versus "refund") scores low.

### Augment and answer
`answer(question)` retrieves three chunks and sends a system message (contains `POLICY ASSISTANT`, "answer only from the context, otherwise say you could not find the information") and a user message `CONTEXT:\n[file]\nchunk…\nQUESTION:\n<question>` to `mock_llm.chat`; it returns the text and the sources.

### Evaluation (`data/raw/policy_eval.csv`)

| Setting | Expected doc in top 3 (of 12) | Answer contains expected fact (of 12) | Fallback on unanswerable (of 2) |
|---|---|---|---|
| 40 words, overlap 10 | **12** | **12** | **2** |
| 40 words, overlap 0 (13 chunks) | 12 | **10** | 2 |

With overlap 0, two questions break because the answer sentence is split across a chunk boundary:
* "What delivery time makes a completed order eligible for a full refund?" — the sentence "…delivered in more than 45 minutes, the customer | receives a full refund…" is cut after "customer", so the best-matching fragment ("receives a full refund of the order amount.") lacks "more than 45 minutes".
* "How soon is money returned when a payment fails but is deducted?" — "…returned automatically | within 48 hours." is cut before "within 48 hours", so the answer misses "48 hours".

The 10-word overlap repeats the end of one chunk at the start of the next, keeping such sentences whole.

### Tools
`TOOLS_SPEC` (name, one-sentence description, JSON-schema parameters) and `TOOL_MAP` hold five tools: `get_order` (reads `data/instamart.db`), `search_policy` (the same retrieval as above), `calculate_refund` (delivered in more than 45 minutes → 100% of the amount, otherwise 0), `issue_refund` and `escalate_to_human` (each append a timestamped JSON line to `outputs/refund_ledger.jsonl`; `issue_refund` also refuses anything above INR 1000 as a last line of defence).

### The seven scenarios (default limits: 6 iterations, 6,000 tokens, 10 s)

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,177 |
| S2 | REFUND | 135.00 | 5 | 1,177 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,164 |
| S4 | DENIED | 0.00 | 4 | 829 |
| S5 | DENIED | 0.00 | 3 | 505 |
| S6 | NOT_FOUND | 0.00 | 2 | 184 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,336 |

All refund amounts equal `expected_refund_inr`; S7 completes all six model calls under the default token limit. `outputs/agent_runs.jsonl` has one line per scenario run (scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool, arguments and observation). `outputs/refund_ledger.jsonl` holds the three real actions (two issued, one escalated).

### Stop-safely demonstrations (S1)

| Demonstration | Status | Detail |
|---|---|---|
| `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | 214 tokens after the second model call; stopped before executing the tool it requested (only `get_order` had run) |
| `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped before the first model call (0 calls, 0 steps) |

Every fallback returns the completed steps plus a short human-handoff summary in `final_answer`. These demo runs are saved in `outputs/agent_demos.jsonl`.

### Crash and resume (S1)
After every tool observation a checkpoint `state/<run id>.json` is written (run id, scenario id, full message list, tokens used, step number, tools executed so far, plus the steps and limits so the run can continue). A simulated crash was raised right after the second tool execution; `resume_agent("S1-crash-demo")` then loaded the checkpoint and finished.

* Tools executed before the crash: `get_order`, `search_policy`
* Tools executed after the resume: `calculate_refund`, `issue_refund`
* **Across both parts, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`** — four executions, none repeated; final status **REFUND 128.25** (5 model calls in total).

The demos write to `outputs/refund_ledger_demo.jsonl` so the real ledger only contains the seven scenario runs. Note that the time limit restarts when a run is resumed.

### ReAct versus plan-and-execute
My ReAct agent asks the model for one decision per step: 5 calls for S1, S2 and S3, 4 for S4, 3 for S5 (cancelled order), 2 for S6 (order not found) and 6 for the looping S7. Tokens grow with every step because the full history is re-sent (1,177 for S1 against 184 for S6). A plan-and-execute design would ask once for the whole step list and call the model again only to replan or summarise: about 2 calls for a normal refund and far fewer tokens, because the history is not re-sent. Its weakness is the unexpected observation. With NOT_FOUND or a cancelled order, my ReAct agent stopped after 2 or 3 calls, whereas a fixed plan would carry on towards calculating and issuing a refund unless every step is guarded, or needs an extra replanning call. For this refund workflow I would choose ReAct: the sequence is short, money moves in the last steps, and the next action truly depends on the last observation (status, delivery time, amount above INR 1000). The extra cost (an estimate, not measured: roughly 1,000 tokens per request) is acceptable, and it is bounded by the iteration, token and time limits, with checkpoints for recovery. A hybrid (plan up front, observation checks before the money-moving steps) would be the next optimisation.

---

## Limitations and Next Steps

* The data is synthetic and generated from simple rules, so the weak correlations, the mock model's keyword behaviour and the clean cluster structure do not reflect real customers or a real LLM.
* **The Premium flag is a post-hoc whole-window label:** it is computed from RFM that includes September, the window that defines the lapse label, so `outputs/retention_list.csv` is illustrative and not a leakage-free evaluation (the model's own features still use only orders before 2025-09-01). Only 1 of the 10 listed customers actually lapsed. A fair version would build segments from data before the cutoff.
* The lapse model has a modest ceiling (test F1 0.655); the logistic-regression best C=100 sits at the edge of the grid, and the test set (232 rows) is small, so differences of one or two customers between models are noise. Next steps: wider grids, calibrated probabilities, a threshold chosen from the cost of an offer versus a lost customer, and time-based validation.
* The triage classifier is a keyword mock; the 9 invalid outputs and 12 chatty answers are by design. A real model would need output-schema enforcement and monitoring of the human-review queue.
* TF-IDF retrieval is word-matching only; embeddings and sentence-aware chunking would remove the chunk-boundary failures seen with zero overlap.
* The refund agent treats all requests as inside the 7-day window and has no authentication or idempotency key for refunds; a production system would need both, and a real approval workflow for escalations.
