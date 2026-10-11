"""Score saved predictions against gold labels: accuracy, macro-F1, invalid rate, throughput, baselines."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
LABELS = ["positive", "negative", "neutral"]


def macro_f1(gold: pd.Series, pred: pd.Series) -> float:
    """Unweighted mean of per-class F1. Invalid outputs (None) count as wrong for every class."""
    scores: list[float] = []
    for c in LABELS:
        tp = ((pred == c) & (gold == c)).sum()
        fp = ((pred == c) & (gold != c)).sum()
        fn = ((pred != c) & (gold == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(scores) / len(scores)


def metrics(gold: pd.Series, pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float((pred == gold).mean()),
        "macro_f1": macro_f1(gold, pred),
        "invalid_rate": float(pred.isna().mean()),
    }


def main() -> None:
    sample = pd.read_csv(ROOT / "data" / "sample_300.csv")
    gold = sample.set_index("id")["label"]
    first100 = gold.loc[gold.index < 100]
    majority = gold.mode().iloc[0]
    rows: list[dict[str, object]] = []

    rows.append({"system": f"baseline: always '{majority}'", **metrics(gold, pd.Series(majority, index=gold.index)),
                 "acc_first100": float((first100 == majority).mean()), "items_per_min": None, "gen_tok_s": None})

    for f in sorted((ROOT / "results").glob("preds_*.csv")):
        p = pd.read_csv(f).set_index("id").reindex(gold.index)
        pred = p["pred"].where(p["pred"].isin(LABELS))
        wall = float(p["wall_seconds"].iloc[0])
        tok = p["eval_count"].sum()
        ns = p["eval_ns"].sum()
        rows.append({
            "system": str(p["model"].iloc[0]),
            **metrics(gold, pred),
            "acc_first100": float((pred.loc[first100.index] == first100).mean()),
            "macro_f1_first100": macro_f1(first100, pred.loc[first100.index]),
            "items_per_min": len(gold) / wall * 60,
            "gen_tok_s": tok / ns * 1e9 if ns else None,  # per-request decode speed (requests overlap 4-way)
            # API spend for the 300 items; local models have no per-item API cost
            "api_cost_usd": float(p["api_cost_usd"].sum()) if "api_cost_usd" in p else 0.0,
        })

    claude_path = ROOT / "data" / "claude_labels_100.csv"
    if claude_path.exists():
        c = pd.read_csv(claude_path).set_index("id")["label"].reindex(first100.index)
        m = metrics(first100, c)
        rows.append({"system": "Claude (frontier reference, 100 items)", "accuracy": None, "macro_f1": None,
                     "invalid_rate": m["invalid_rate"], "acc_first100": m["accuracy"],
                     "macro_f1_first100": m["macro_f1"], "items_per_min": None, "gen_tok_s": None})

    rows[0]["macro_f1_first100"] = macro_f1(first100, pd.Series(majority, index=first100.index))

    table = pd.DataFrame(rows)[["system", "accuracy", "macro_f1", "invalid_rate", "acc_first100",
                                "macro_f1_first100", "items_per_min", "gen_tok_s", "api_cost_usd"]]
    table.to_csv(ROOT / "results" / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
