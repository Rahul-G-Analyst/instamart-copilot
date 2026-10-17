"""Part C: support-ticket triage with guardrails (redaction, retries, output validation, token budget, routing).
Run from the repository root:  python src/c_triage.py"""
import json
import os
import random
import re
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402  (supplied file, used unmodified)
from mock_llm import TransientLLMError, est_tokens  # noqa: E402

CATS = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

# ---------------- redaction: exactly these patterns, in this order ----------------
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text):
    counts = {}
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts

def contains_pii(text):
    return any(p.search(text) for _, p, _ in PATTERNS)

# ---------------- prompts ----------------
JSON_RULE = ('Reply with JSON only, in exactly this form: {"category": "...", "reason": "..."} '
             'where category is one of billing, delivery, technical, fraud.')
ZERO_SHOT = "You CLASSIFY support tickets into billing, delivery, technical or fraud. " + JSON_RULE
FEW_SHOT = ("You CLASSIFY support tickets into billing, delivery, technical or fraud. " + JSON_RULE + "\n"
            "EXAMPLES:\n"
            "Ticket: I was charged twice for the same order. -> {\"category\": \"billing\", \"reason\": \"double charge\"}\n"
            "Ticket: My order is very late and the rider is not answering. -> {\"category\": \"delivery\", \"reason\": \"late rider\"}\n"
            "Ticket: The app crashes when I open the cart. -> {\"category\": \"technical\", \"reason\": \"app crash\"}\n"
            "Ticket: There is an unauthorized transaction on my card. -> {\"category\": \"fraud\", \"reason\": \"unauthorized payment\"}")
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

# ---------------- resilient call ----------------
class RunStats:
    def __init__(self):
        self.transient_errors = 0
        self.waits = []          # (retry_number, wait)
        self.log = []
        self.prompts_checked = 0

def call_with_retry(messages, ticket_id, stats):
    """One model call, max 3 retries, exponential backoff with jitter, one log line per retry."""
    for retry in range(MAX_RETRIES + 1):
        # guardrail: nothing identifiable may reach the model
        for m in messages:
            assert not contains_pii(m["content"]), f"PII leaked into prompt of {ticket_id}"
        stats.prompts_checked += 1
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats.transient_errors += 1
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            stats.waits.append((retry, wait))
            line = f"[retry] ticket={ticket_id} retry={retry} wait={wait:.4f}s"
            stats.log.append(line)
            time.sleep(wait)

# ---------------- output validation ----------------
def validate(content):
    """Return (status, category). Never guesses."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATS}
        return ("classified_keyword", next(iter(found))) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and obj.get("category") in CATS:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""

# ---------------- routing ----------------
def route(status, category, is_premium):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

# ---------------- pipeline ----------------
def run_pipeline(tickets, system_prompt, seg, token_budget=None, seed=42, verbose=False):
    mock_llm.reset()                       # counters restart before every full run
    random.seed(seed)                      # reproducible jitter
    stats = RunStats()
    seg_idx = seg.set_index("customer_id")
    total_tokens, rows = 0, []
    for t in tickets.itertuples(index=False):
        text, used, category = t.text, 0, ""
        if not text.strip():
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            redacted, _ = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redacted}]
            assert messages[1]["content"] == redacted
            resp = call_with_retry(messages, t.ticket_id, stats)
            used = resp.usage.total_tokens
            total_tokens += used
            status, category = validate(resp.choices[0].message.content)
        in_seg = t.customer_id in seg_idx.index
        seg_name = seg_idx.loc[t.customer_id, "segment_name"] if in_seg else ""
        premium = bool(seg_idx.loc[t.customer_id, "is_premium"]) if in_seg else False
        r, p = route(status, category, premium)
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": status, "category": category,
                     "segment_name": seg_name, "priority": p, "route": r, "tokens_used": used})
    return pd.DataFrame(rows), stats

def summarise(df, tickets, stats):
    cl = df[df["status"].isin(["classified_json", "classified_keyword"])]
    truth = tickets.set_index("ticket_id").loc[cl["ticket_id"], "true_category"].values
    correct = int((cl["category"].values == truth).sum())
    tot = int(df["tokens_used"].sum())
    return {"classified": len(cl), "correct": correct, "accuracy": correct / len(cl) * 100, "total_tokens": tot,
            "tokens_per_classified": tot / len(cl), "transient_errors": stats.transient_errors,
            "status_counts": df["status"].value_counts().to_dict()}, cl, truth

def main():
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    tickets = pd.read_csv(os.path.join(ROOT, "data/raw/tickets.csv"), keep_default_na=False)
    seg = pd.read_csv(os.path.join(ROOT, "outputs/customer_segments.csv"))

    # --- task 1: input guardrails
    empty = int((tickets["text"].str.strip() == "").sum())
    longs = tickets[tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS]
    print(f"== guardrails: empty={empty} long={len(longs)} long est_tokens={sorted(set(longs['text'].map(est_tokens)))}")

    # --- task 2: redaction counts across all 164 tickets
    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted_all = []
    for txt in tickets["text"]:
        red, c = redact(txt)
        redacted_all.append(red)
        for k in totals:
            totals[k] += c[k]
    left = sum(len(p.findall(r)) for r in redacted_all for _, p, _ in PATTERNS)
    print(f"== redaction: {totals} total={sum(totals.values())}; matches left after re-applying patterns={left}")
    assert left == 0

    # --- tasks 4-6: full runs, no budget
    results = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        df, st = run_pipeline(tickets, prompt, seg)
        s, cl, truth = summarise(df, tickets, st)
        results[name] = (df, st, s)
        print(f"\n== {name} run (prompt est_tokens={est_tokens(prompt)})")
        print("status counts:", s["status_counts"])
        print(f"transient errors={st.transient_errors} (all recovered; no ticket permanently failed); "
              f"PII assertion checked {st.prompts_checked} prompts")
        print(f"classified={s['classified']} correct={s['correct']} accuracy={s['accuracy']:.1f}% "
              f"total_tokens={s['total_tokens']} tokens/classified ticket={s['tokens_per_classified']:.1f}")
        if name == "zero-shot":
            print("confusion table (rows=true, cols=predicted):")
            print(pd.crosstab(pd.Series(truth, name="true"), pd.Series(cl["category"].values, name="predicted")).to_string())
            for line in st.log[:3]:
                print(line)
            print(f"... {len(st.log)} retry log lines in total")
            for n in range(3):
                ws = [w for r, w in st.waits if r == n]
                if ws:
                    lo, hi = BASE_DELAY * 2 ** n * 0.9, BASE_DELAY * 2 ** n * 1.15
                    assert all(lo <= w <= hi for w in ws)
                    print(f"retry number {n}: {len(ws)} waits in [{min(ws):.4f}, {max(ws):.4f}], allowed [{lo:.4f}, {hi:.4f}]")

    # --- task 7a: budget demo (zero-shot)
    BUDGET = 6000
    demo, st = run_pipeline(tickets, ZERO_SHOT, seg, token_budget=BUDGET)
    demo.to_csv(os.path.join(ROOT, "outputs/triage_budget_demo.csv"), index=False)
    n_ex = int((demo["status"] == "budget_exhausted").sum())
    print(f"\n== budget demo (zero-shot, TOKEN_BUDGET={BUDGET}): budget_exhausted={n_ex}; tokens spent={int(demo.tokens_used.sum())}")
    assert n_ex >= 1

    # --- task 7b: final results (few-shot, budget 100000)
    final, st = run_pipeline(tickets, FEW_SHOT, seg, token_budget=100000)
    final.to_csv(os.path.join(ROOT, "outputs/triage_results.csv"), index=False)
    print(f"\n== final triage_results.csv rows={len(final)}")
    print("statuses:", final["status"].value_counts().to_dict())
    print("routes:", final["route"].value_counts().to_dict())
    print("priorities:", final["priority"].value_counts().to_dict())
    print("tickets from customers with no segment row:", int((final["segment_name"] == "").sum()))
    assert len(final) == 164

if __name__ == "__main__":
    main()
