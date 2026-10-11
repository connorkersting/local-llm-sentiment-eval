"""Draw a fixed, reproducible 300-item sample from Financial PhraseBank (all-annotators-agree subset)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "unzipped" / "FinancialPhraseBank-v1.0" / "Sentences_AllAgree.txt"
OUT = ROOT / "data" / "sample_300.csv"
SEED = 42
N = 300


def load_phrasebank(path: Path) -> pd.DataFrame:
    """Each line is `sentence@label`; the file is Latin-1 encoded."""
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        rows.append({"text": text.strip(), "label": label.strip()})
    return pd.DataFrame(rows)


def main() -> None:
    full = load_phrasebank(RAW)
    sample = (
        full.sample(n=N, random_state=SEED)
        .reset_index(drop=True)
        .rename_axis("id")
        .reset_index()
    )
    sample.to_csv(OUT, index=False)
    print(f"full subset: {len(full)} rows; class mix: {full['label'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"sample: {len(sample)} rows (seed {SEED}); class mix: {sample['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
