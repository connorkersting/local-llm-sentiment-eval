# Data source

**Dataset:** Financial PhraseBank v1.0, subset `Sentences_AllAgree` (sentences on which every annotator agreed).

- **Authors:** Pekka Malo, Ankur Sinha, Pyry Takala, Pekka Korhonen, Jyrki Wallenius (Aalto University School of Business).
- **Paper:** Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the Association for Information Science and Technology, 65(4), 782–796.
- **Downloaded from:** https://huggingface.co/datasets/takala/financial_phrasebank, file `data/FinancialPhraseBank-v1.0.zip` (681,890 bytes, SHA-256 starts `0E1A06C4900FDAE4`), downloaded 2026-10-11.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0), stated in the archive's `License.txt` and the dataset card. This project is non-commercial. The sampled sentences in `data/sample_300.csv` are redistributed under the same license with this attribution. Commercial use needs a license from the authors.

## Task

Each sentence is a sentence from English-language financial news about (mostly Finnish) listed companies, labelled `positive`, `negative` or `neutral` from an investor's point of view.

## Sampling

- Full `AllAgree` subset: 2,264 sentences (61.4% neutral, 25.2% positive, 13.4% negative).
- `sample.py` draws **300** sentences with `pandas.DataFrame.sample(n=300, random_state=42)` (simple random sample, no stratification). Sample mix: 189 neutral, 77 positive, 34 negative.
- Rows are re-indexed `id` 0–299. Items 0–99 form the subset used for the frontier-model reference.

## Frontier reference labels

`data/claude_labels_100.csv` holds labels for items 0–99 assigned by Claude (Anthropic), working in an interactive session. Claude read only the sentence text, not the gold labels, and labelled all 100 in one pass with no second attempt. It's a single human-style annotation pass, not an API run with a fixed prompt, so treat it as an approximate ceiling rather than a reproducible benchmark.
