"""Diagnostics behind the summary: output length per model and confusion matrices (gold rows x predicted columns)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent


def main() -> None:
    gold = pd.read_csv(ROOT / "data" / "sample_300.csv").set_index("id")["label"]
    for f in sorted((ROOT / "results").glob("preds_*.csv")):
        p = pd.read_csv(f).set_index("id")
        print(f"\n== {p['model'].iloc[0]}: mean output tokens {p['eval_count'].mean():.1f}, "
              f"mean decode time per item {p['eval_ns'].mean() / 1e9:.2f}s")
        print(pd.crosstab(gold, p["pred"].reindex(gold.index), rownames=["gold"], colnames=["pred"]))


if __name__ == "__main__":
    main()
