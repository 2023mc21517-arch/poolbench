"""Diagnose which seed words cause contamination flags per concept."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from poolbench.data.concepts import CONCEPTS

corpora = Path(__file__).parent.parent / "data" / "corpora"
targets = ["conditionality", "frustration", "uncertainty", "planning",
           "hedging", "legal_formality", "deference", "causation",
           "contrast"]

for concept in targets:
    meta = CONCEPTS.get(concept, {})
    sw = [s.lower() for s in meta.get("seed_words", [])]
    if not sw:
        print(f"\n{concept}: NO seed_words defined")
        continue
    f = corpora / concept / "train_neg.jsonl"
    if not f.exists():
        print(f"\n{concept}: no train_neg.jsonl")
        continue
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    print(f"\n{concept} (train_neg={len(recs)})  seed_words={sw}")
    for w in sw:
        hits = sum(1 for r in recs if w in r["text"].lower())
        multi = len(w.split()) > 1
        tag = "multi" if multi else "SINGLE"
        print(f"  [{tag}] '{w}': {hits}/{len(recs)} = {hits/len(recs)*100:.0f}%")
