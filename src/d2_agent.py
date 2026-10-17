"""Part D, tasks 5-9: refund agent (tools, ReAct loop, safe stopping, checkpoint/resume).
Run from the repository root:  python src/d2_agent.py"""
import glob
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import mock_llm  # noqa: E402  (supplied file, used unmodified)
from d1_rag import retrieve  # noqa: E402  (retrieval built in tasks 1-3 is reused here)

DB_PATH = os.path.join(ROOT, "data", "instamart.db")
OUT_DIR = os.path.join(ROOT, "outputs")
STATE_DIR = os.path.join(ROOT, "state")
LEDGER_PATH = os.path.join(OUT_DIR, "refund_ledger.jsonl")
EXECUTION_LOG = []          # every real tool execution, in order (used to prove nothing is repeated)

AGENT_SYSTEM = ("You are an AGENT that decides refund requests for Instamart support. Look up the order, check the refund "
                "policy, calculate the refund and then issue it. A refund above INR 1000 must be escalated to a human "
                "and never issued automatically. Finish with one line that starts with REFUND:, ESCALATE:, DENIED: "
                "or NOT_FOUND:.")

# ---------------- task 5: tools ----------------
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or (isinstance(row[3], float) and math.isnan(row[3])) else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": dm}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]}
                        for h in retrieve(query, k=3)]}

def calculate_refund(order_amount, delivery_minutes):
    """Refund rule: delivered in more than 45 minutes -> 100% of the order amount, otherwise 0."""
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")

def issue_refund(order_id, refund_inr):
    if float(refund_inr) > 1000:            # last line of defence: big refunds are never issued automatically
        return {"error": "refund above INR 1000 requires human approval"}
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def _spec(name, description, props, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}

TOOLS_SPEC = [
    _spec("get_order", "Look up one order in the Instamart database by its order id.",
          {"order_id": {"type": "string", "description": "e.g. ORD-100123"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three most relevant chunks.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and the delivery time in minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

def execute_tool(name, arguments):
    EXECUTION_LOG.append(name)
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**arguments)
    except Exception as exc:  # a failing tool becomes an observation, not a crash
        return {"error": f"{type(exc).__name__}: {exc}"}

# ---------------- tasks 6-8: ReAct loop, stop limits, checkpoints ----------------
class SimulatedCrash(Exception):
    pass

def _checkpoint_path(run_id, state_dir):
    return os.path.join(state_dir, f"{run_id}.json")

def _save_checkpoint(state, state_dir):
    os.makedirs(state_dir, exist_ok=True)
    with open(_checkpoint_path(state["run_id"], state_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)

def _fallback(state, status, reason):
    tools = ", ".join(s["tool"] for s in state["steps"]) or "none"
    summary = (f"HANDOFF to human agent: stopped with {status} ({reason}) after {len(state['steps'])} completed step(s) "
               f"[{tools}]. Request: {state['messages'][1]['content'][:80]}")
    return _result(state, status, summary)

def _result(state, status, final_answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": final_answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"]}

def _loop(state, state_dir, crash_after_tools=None):
    p = state["params"]
    start = time.monotonic()
    while state["llm_calls"] < p["max_iterations"]:
        if time.monotonic() - start >= p["time_limit_seconds"]:                       # (c) wall-clock limit
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit {p['time_limit_seconds']}s reached")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > p["token_limit"]:                                   # (b) token limit, tool NOT run
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"{state['tokens_used']} tokens used, limit {p['token_limit']}")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                        # final answer
            text = msg.content or ""
            status = text.split(":")[0].strip() if ":" in text else "UNKNOWN"
            return _result(state, status, text)
        for tc in msg.tool_calls:
            state["messages"].append({"role": "assistant", "content": None,
                                      "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            obs = execute_tool(tc.name, tc.arguments)
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
            state["tools_executed"].append(tc.name)
            state["step"] += 1
            _save_checkpoint(state, state_dir)                                        # after every observation
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool execution #{crash_after_tools}")
    return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"{p['max_iterations']} model calls without a final answer")  # (a)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, state_dir=STATE_DIR, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}-run", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "tools_executed": [], "steps": [],
             "params": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds}}
    return _loop(state, state_dir, crash_after_tools)

def resume_agent(run_id, state_dir=STATE_DIR):
    with open(_checkpoint_path(run_id, state_dir), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state, state_dir)

# ---------------- driver ----------------
def refund_of(result):
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            return float(s["observation"]["refund_inr"])
    return 0.0

def jsonl_line(result, extra=None):
    d = {"scenario_id": result["scenario_id"], "status": result["status"], "final_answer": result["final_answer"],
         "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"], "steps": result["steps"]}
    d.update(extra or {})
    return json.dumps(d)

def main():
    global LEDGER_PATH
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(STATE_DIR, "*.json")):
        os.remove(f)
    real_ledger = LEDGER_PATH
    demo_ledger = os.path.join(OUT_DIR, "refund_ledger_demo.jsonl")
    for p in (real_ledger, demo_ledger, os.path.join(OUT_DIR, "agent_runs.jsonl"), os.path.join(OUT_DIR, "agent_demos.jsonl")):
        open(p, "w").close()

    sc = pd.read_csv(os.path.join(ROOT, "data/raw/agent_scenarios.csv"))
    print("== seven scenarios (default limits)")
    print("scenario | status | refund | model calls | tokens | expected")
    runs = {}
    with open(os.path.join(OUT_DIR, "agent_runs.jsonl"), "a", encoding="utf-8") as out:
        for r in sc.itertuples():
            res = run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}-main")
            runs[r.scenario_id] = res
            out.write(jsonl_line(res) + "\n")
            amt = refund_of(res)
            exp_ok = (res["status"] == r.expected_outcome or (r.expected_outcome == "FALLBACK" and res["status"] == "FALLBACK_MAX_ITERATIONS")) \
                and abs(amt - float(r.expected_refund_inr)) <= 0.01
            print(f"{r.scenario_id} | {res['status']} | {amt:.2f} | {res['llm_calls']} | {res['tokens_used']} | "
                  f"{r.expected_outcome} {r.expected_refund_inr:.2f} -> {'OK' if exp_ok else 'MISMATCH'}")
            assert exp_ok, r
    assert runs["S1"]["llm_calls"] == 5 and runs["S7"]["llm_calls"] == 6 and runs["S7"]["tokens_used"] <= 6000

    # demos use their own ledger so the real ledger holds only the seven scenario runs
    LEDGER_PATH = demo_ledger
    s1 = sc.iloc[0]
    with open(os.path.join(OUT_DIR, "agent_demos.jsonl"), "a", encoding="utf-8") as out:
        print("\n== stop-safely demos (S1)")
        d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token-limit-demo")
        print("token_limit=200      ->", d1["status"], "| tokens", d1["tokens_used"], "| steps run:", [s["tool"] for s in d1["steps"]])
        print("   final_answer:", d1["final_answer"])
        out.write(jsonl_line(d1, {"demo": "token_limit=200"}) + "\n")
        d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
        print("time_limit_seconds=0 ->", d2["status"], "| model calls", d2["llm_calls"], "| steps", len(d2["steps"]))
        print("   final_answer:", d2["final_answer"])
        out.write(jsonl_line(d2, {"demo": "time_limit_seconds=0"}) + "\n")

        print("\n== crash and resume (S1)")
        EXECUTION_LOG.clear()
        try:
            run_agent(s1.message, "S1", run_id="S1-crash-demo", crash_after_tools=2)
            raise AssertionError("crash expected")
        except SimulatedCrash as e:
            print("crash raised:", e)
        before = list(EXECUTION_LOG)
        with open(_checkpoint_path("S1-crash-demo", STATE_DIR)) as f:
            cp = json.load(f)
        print("checkpoint keys:", sorted(cp), "| tools in checkpoint:", cp["tools_executed"])
        res = resume_agent("S1-crash-demo")
        after = EXECUTION_LOG[len(before):]
        print("tools executed before crash:", before)
        print("tools executed after resume:", after)
        print("tools executed across both parts:", EXECUTION_LOG)
        print("final:", res["status"], f"{refund_of(res):.2f}", "| total model calls", res["llm_calls"], "| tokens", res["tokens_used"])
        assert EXECUTION_LOG == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
        assert res["status"] == "REFUND" and abs(refund_of(res) - 128.25) <= 0.01
        out.write(jsonl_line(res, {"demo": "crash_and_resume", "tools_executed": EXECUTION_LOG}) + "\n")
    LEDGER_PATH = real_ledger
    print("\nledger lines:", sum(1 for _ in open(real_ledger)), "| runs lines:", sum(1 for _ in open(os.path.join(OUT_DIR, 'agent_runs.jsonl'))),
          "| checkpoints:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(STATE_DIR, '*.json'))))

if __name__ == "__main__":
    main()
