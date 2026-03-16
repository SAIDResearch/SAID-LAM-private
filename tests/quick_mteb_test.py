#!/usr/bin/env python3
"""Quick test: LAM + mteb.evaluate() on needle/passkey tasks.

Run from repo root:  PYTHONPATH=. python tests/quick_mteb_test.py
Or from said-lam/:  python tests/quick_mteb_test.py  (if said_lam is on path)
"""
import sys
from pathlib import Path

# Ensure repo root is on path so "said_lam" is found
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from said_lam import LAM
import mteb

model = LAM("SAIDResearch/SAID-LAM-v1")
tasks = mteb.get_tasks(tasks=["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"])
results = mteb.evaluate(model=model, tasks=tasks)

print("\n--- Results ---")
for tr in results.task_results:
    print(f"  {tr.task_name}: {tr.get_score():.4f}")
print("Done.")
