# QSS 20 Final Project — Linguistic Framing of Sexual-Assault Judicial Opinions

**Author:** Sarnika Ali
**Spring 2026, QSS 20 Final Project**

This project is an extension of my senior QSS thesis. It analyzes how U.S. judicial opinions in sexual-assault cases linguistically frame victims and perpetrators, and whether that framing differs between male-victim and female-victim cases. The repo contains the data-pull, feature-extraction, and analysis pipeline plus the processed case-level dataset that powers the project's findings.

---

## Research question

How does victim gender influence the relative frequency of perpetrator-reducing and credibility-undermining linguistic structures in U.S. judicial opinions? Specifically, I compare per-case rates of five families of linguistic framing (passive constructions, victim-vs-perpetrator agency ratios, direct-violence vocabulary, euphemism / clinical language, and hedging / credibility-challenge markers) between male-victim and female-victim cases, stratified by whether the case is a criminal prosecution or a civil suit.

---

## Repository layout

```
qss20-final-project/
├── README.md                          this file
├── code/
│   ├── 00_pull.ipynb                  fetch opinion documents from CourtListener
│   ├── 01_features.ipynb              NLP pipeline: extract 140+ per-case features → cases.csv
│   └── 02_analyze.ipynb               effect sizes, FDR, and all figures (runs from cases.csv)
├── data/
│   └── cases.csv                      processed case-level dataset (97 cases × opinion-only features)
└── output/
    ├── figure1_corpus_composition.png         corpus composition by gender × case type
    ├── figure2_direct_violence_by_gender.png  direct-violence finding (Cliff's δ = −0.62, q_FDR < 0.01)
    ├── hero_act_vs_actor.png                  headline "act vs. actor" figure (δ = −0.62 / +0.63)
    ├── 09_forest_plot.png                     forest plot of Cliff's δ across framing detectors
    ├── headline_stats.csv                     headline effect-size table (5 measures × 3 strata)
    └── forest_stats.csv                       per-detector stats, criminal stratum (with FDR + Bonferroni)
```

The notebooks are numbered to run in order: `00_pull` → `01_features` → `02_analyze`.

---

## What each notebook does

Functions are defined at the top of each notebook; paths are resolved relative to the repo (no hardcoded paths).

### `code/00_pull.ipynb` — pull
**Inputs:** a corpus manifest of case captions/citations + optional `CL_API_TOKEN` (raises the rate limit from ~5,000/day to ~10,000/hr).
**What it does:** for each case, builds a search query from the parsed caption, searches the [CourtListener REST API](https://www.courtlistener.com/api/rest/v4/), scores candidates on caption / year / jurisdiction, and downloads the opinion text, HTML, and PDF; prints per-case retrieval diagnostics and falls back when a case is unfindable.
**Outputs:** raw opinion files under `data/raw_opinions/` (not committed) + a per-case metadata table.

### `code/01_features.ipynb` — features (clean + merge)
**Inputs:** the raw opinion `.docx` / `.txt` files from `00_pull`.
**What it does:** strips editorial headnotes, segments each opinion into sentences, and extracts 140+ per-case features — readability, part-of-speech shares, Empath topic-vocabulary rates, VADER sentiment, and **19 custom sexual-assault framing detectors** (victim-blaming, credibility challenges, procedural deflection, hedging, perpetrator naming, direct-violence vocabulary, …). Per-case rows are merged into the case table, with **before/after merge diagnostics** (row counts, duplicate check, stratum balance).
**Outputs:** `data/cases.csv` (one row per case) + per-case sentence-level `.xlsx` and stats `.json`. Requires `spacy`, `empath`, `vaderSentiment`.

### `code/02_analyze.ipynb` — analyze
**Inputs:** `data/cases.csv`.
**What it does:** derives per-sentence framing rates and compares male- vs female-victim cases **within each stratum** (criminal / civil) using Mann-Whitney U, Cliff's δ with 500-resample bootstrap CIs, and Benjamini–Hochberg FDR (+ Bonferroni) across the framing detectors.
**Outputs:** `output/headline_stats.csv`, `output/forest_stats.csv`, and `figure1_corpus_composition.png`, `figure2_direct_violence_by_gender.png`, `hero_act_vs_actor.png`, `09_forest_plot.png`.

---

## How to reproduce

`02_analyze.ipynb` regenerates every stats table and figure from the committed `data/cases.csv` — **no raw opinions or network access required**:

```bash
pip install pandas numpy scipy matplotlib jupyter
jupyter nbconvert --to notebook --execute code/02_analyze.ipynb
```

`00_pull.ipynb` and `01_features.ipynb` reproduce the upstream pipeline (CourtListener retrieval → feature extraction). They require the raw `.docx` judicial opinions, which are not committed to this repo — see the **Data** section below.

---

## Data

The processed dataset `data/cases.csv` has **one row per case** (n = 97 U.S. judicial opinions) and ~140 columns. **All text-derived features are computed on the opinion text only**; editorial headnotes / syllabi are stripped before feature extraction, so that the analysis measures the court's own language rather than a reporter's summary. Key column families:

| Column family | Examples | What it captures |
|---|---|---|
| Identifiers | `case_id`, `case_caption`, `plaintiff_last`, `defendant_last`, `year`, `citation` | Per-case metadata |
| Stratification | `folder_victim_gender_label`, `is_govt_prosecution`, `appellate_criminal_flip` | Male/female label, criminal/civil indicator |
| Structural / readability | `n_sentences`, `flesch_kincaid_grade`, `gunning_fog`, `avg_sentence_length` | Document shape |
| Part of speech | `pct_noun`, `pct_verb`, `pct_passive_aux`, etc. | spaCy POS share |
| Empath topic vocab | `empath_crime`, `empath_violence`, `empath_power`, `empath_masculine`, … | 30+ topic-vocabulary rates |
| VADER sentiment | `vader_doc_compound`, `vader_para_mean_compound`, etc. | Document + paragraph sentiment |
| **Framing detectors** | `sa_direct_violence_term`, `sa_victim_blame_direct`, `sa_credibility_challenge`, `sa_perp_ref_name`, `sa_clinical_legal_term`, `sa_hedging_allegation`, … | The 19 custom detectors — raw counts; divide by `n_sentences` for per-case rates |

The **raw `.docx` judicial opinions** are not committed to this public repository because they are large and were retrieved through CourtListener / the Caselaw Access Project under terms that favor not re-hosting bulk court documents publicly. They are available on request for replication purposes.

---

## Key finding (so far)

Within criminal cases, female-victim opinions name the violence directly **much more often** than male-victim opinions. The `sa_direct_violence_term` rate has a Cliff's δ of **−0.62** (q_FDR < 0.01, Bonferroni-significant), meaning in roughly 80% of male-vs-female case pairs the female-victim opinion has the higher direct-violence-term rate. See `output/figure2_direct_violence_by_gender.png`.

Perpetrator naming runs in the opposite direction (δ = **+0.63**, q_FDR < 0.01) — male-victim opinions name the perpetrator by surname / proper noun more often. Together these two findings form the project's "act vs. actor" headline: criminal opinions foreground the violent **act** when the victim is a woman and the named **actor** when the victim is a man (`output/hero_act_vs_actor.png`). The forest plot across all framing detectors is at `output/09_forest_plot.png`.

---

## Citation

If you use this code or dataset, please cite the senior thesis (in progress) and link back to this repository.
