"""Part D, tasks 1-4: chunking, TF-IDF retrieval, grounded answering and its evaluation.
Run from the repository root:  python src/d1_rag.py"""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402  (supplied file, used unmodified)

POLICY_DIR = os.path.join(ROOT, "data", "policies")

# ---------------- task 1: chunking ----------------
def chunk_documents(size=40, overlap=10, policy_dir=POLICY_DIR):
    """Word chunks of `size` words, each starting size-overlap words after the previous one.
    Stops after the first chunk whose end reaches or passes the last word."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(policy_dir, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "start_word": start,
                           "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += size - overlap
    return chunks

# ---------------- task 2: vectorise and retrieve ----------------
class Index:
    def __init__(self, size=40, overlap=10):
        self.chunks = chunk_documents(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in order]

_DEFAULT = Index(40, 10)

def retrieve(query, k=3):
    """Top-k policy chunks for a query (default index: 40-word chunks, overlap 10)."""
    return _DEFAULT.retrieve(query, k)

# ---------------- task 3: augment and answer ----------------
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for Instamart support. Answer the question using only the supplied "
                 "context. If the context does not contain the answer, say that you could not find the information.")

def build_user_message(question, hits):
    blocks = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    return f"CONTEXT:\n{blocks}\nQUESTION:\n{question}"

def answer(question, index=None, k=3):
    hits = (index or _DEFAULT).retrieve(question, k)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, hits)}]
    resp = mock_llm.chat(messages)
    return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits]}

# ---------------- task 4: evaluation ----------------
def evaluate(index, verbose=True):
    qa = pd.read_csv(os.path.join(ROOT, "data/raw/policy_eval.csv"), keep_default_na=False)
    ans = qa[qa["expected_doc"] != "NONE"]
    doc_hits = fact_hits = fb_hits = 0
    failures = []
    for r in ans.itertuples():
        out = answer(r.question, index)
        d, f = r.expected_doc in out["sources"], r.expected_fact in out["answer"]
        doc_hits += d; fact_hits += f
        if not (d and f):
            failures.append((r.question, r.expected_doc, r.expected_fact, out["sources"], out["answer"]))
    unans = qa[qa["expected_doc"] == "NONE"]
    for r in unans.itertuples():
        out = answer(r.question, index)
        fb_hits += out["answer"].startswith("I could not find")
        if verbose:
            print(f"  unanswerable: {r.question!r} -> {out['answer']!r}")
    res = {"answerable": len(ans), "doc_hits": doc_hits, "fact_hits": fact_hits,
           "unanswerable": len(unans), "fallback_hits": fb_hits}
    return res, failures

def main():
    chunks = chunk_documents(40, 10)
    counts = pd.Series([c["source"] for c in chunks]).value_counts().sort_index()
    print(f"== chunks (size 40, overlap 10): total={len(chunks)}  {counts.to_dict()}")
    top = retrieve("refund policy late delivery")
    print("== retrieve('refund policy late delivery'):", [(h["source"], round(h["score"], 3)) for h in top])
    assert all(h["source"] == "refund_policy.txt" for h in top)

    print("\n== evaluation, overlap 10")
    res, fails = evaluate(Index(40, 10))
    print(res)
    for f in fails:
        print("  FAIL", f)

    print("\n== evaluation, overlap 0")
    idx0 = Index(40, 0)
    print("chunks:", len(idx0.chunks), pd.Series([c["source"] for c in idx0.chunks]).value_counts().sort_index().to_dict())
    res0, fails0 = evaluate(idx0)
    print(res0)
    for q, doc, fact, srcs, a in fails0:
        print(f"  FAIL question={q!r}\n       expected {doc} / {fact!r}; sources={srcs}\n       answer={a!r}")
        # show where the expected fact sits relative to chunk boundaries
        for c in idx0.chunks:
            if c["source"] == doc:
                print(f"       chunk@{c['start_word']}: ...{c['text'][-60:]!r}" if fact not in c["text"] else f"       fact is whole inside chunk@{c['start_word']}")

    print("\n== sample grounded answers")
    for q in ["What delivery time makes a completed order eligible for a full refund?", "Who must approve a refund above INR 1000?"]:
        print(q, "->", answer(q))

if __name__ == "__main__":
    main()
