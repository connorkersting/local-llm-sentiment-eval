"""Escalation router, replayed from saved predictions (no model calls).

Policy: run qwen3.5 and gemma4 locally over the whole batch (model by model: both are 6.6 GB and can't share
8 GB of VRAM); accept the label when they agree, send disagreements to Claude Haiku 5.5. Compared against
each single system, with API cost per 1,000 items.
"""
from pathlib import Path

import pandas as pd

from score import LABELS, macro_f1

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
QWEN = RESULTS / "preds_qwen3.5_9b-q4_K_M.csv"
GEMMA = RESULTS / "preds_gemma4_e4b-it-q4_K_M.csv"
HAIKU = RESULTS / "preds_claude-haiku-5-5.csv"


def load_preds(path: Path, index: pd.Index) -> pd.DataFrame:
    p = pd.read_csv(path).set_index("id").reindex(index)
    p["pred"] = p["pred"].where(p["pred"].isin(LABELS))
    return p


def main() -> None:
    gold = pd.read_csv(ROOT / "data" / "sample_300.csv").set_index("id")["label"]
    qwen = load_preds(QWEN, gold.index)["pred"]
    gemma = load_preds(GEMMA, gold.index)["pred"]
    haiku_df = load_preds(HAIKU, gold.index)
    haiku = haiku_df["pred"]
    haiku_cost_per_item = haiku_df["api_cost_usd"]

    agree = qwen.notna() & (qwen == gemma)
    routed = qwen.where(agree, haiku)
    escalated = ~agree

    def row(name: str, pred: pd.Series, cost_items: pd.Series | None) -> dict[str, object]:
        cost = float(cost_items.sum()) if cost_items is not None else 0.0
        return {"system": name, "accuracy": float((pred == gold).mean()), "macro_f1": macro_f1(gold, pred),
                "escalated": int(escalated.sum()) if name.startswith("router") else None,
                "api_cost_per_1k_items_usd": cost / len(gold) * 1000}

    table = pd.DataFrame([
        row("qwen3.5:9b alone", qwen, None),
        row("gemma4:e4b alone", gemma, None),
        row("claude-haiku-5-5 alone", haiku, haiku_cost_per_item),
        row("router: agree -> local, disagree -> haiku", routed, haiku_cost_per_item.where(escalated, 0.0)),
    ])
    table.to_csv(RESULTS / "router_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
    print(f"\nagreement {agree.mean():.1%}; accuracy when local models agree "
          f"{(qwen[agree] == gold[agree]).mean():.1%}; haiku accuracy on the escalated items "
          f"{(haiku[escalated] == gold[escalated]).mean():.1%} (n={escalated.sum()})")


if __name__ == "__main__":
    main()
