# QSS 20 Final Project — Linguistic Framing of Sexual-Assault Judicial Opinions

**Author:** Sarnika Ali
**Spring 2026, QSS 20 Final Project (Milestone 2)**

This project is an extension of my senior QSS thesis. It analyzes how U.S. judicial opinions in sexual-assault cases linguistically frame victims and perpetrators, and whether that framing differs between male-victim and female-victim cases. The repo contains the data-pull, feature-extraction, and analysis pipeline plus the processed case-level dataset that powers the project's findings.

---

## Research question

How does victim gender influence the relative frequency of perpetrator-reducing and credibility-undermining linguistic structures in U.S. judicial opinions? Specifically, I compare per-case rates of five families of linguistic framing — passive constructions, victim-vs-perpetrator agency ratios, direct-violence vocabulary, euphemism / clinical language, and hedging / credibility-challenge markers between male-victim and female-victim cases, stratified by whether the case is a criminal prosecution or a civil suit.

---

## Repository layout

```
qss20-final-project/
├── README.md                          this file
├── code/
│   ├── 01_data_pull.py                fetch opinion documents from CourtListener
│   ├── 02_extract_features.py         NLP pipeline: extract 140+ per-case features
│   ├── 03_analyze_and_visualize.py    compute Cliff's δ + FDR + thesis figures
│   └── 04_milestone1_figures.py       reproduce the two Milestone 1 starter figures
├── data/
│   └── cases.csv                      processed case-level dataset (97 cases × 140+ features)
└── output/
    ├── figure1_corpus_composition.png         corpus composition by gender × case type
    ├── figure2_direct_violence_by_gender.png  headline finding (Cliff's δ = −0.61, q_FDR < 0.01)
    ├── 09_forest_plot.png                     forest plot of Cliff's δ across all 19 framing detectors
    ├── headline_stats.csv                     within-criminal-stratum effect-size table
    └── forest_stats.csv                       per-feature stats stratified by case type
```

---

## What each script does

### `code/01_data_pull.py` — data pull
Fetches official court-opinion documents from the [CourtListener REST API](https://www.courtlistener.com/api/rest/v4/). For each case in the corpus it: (1) builds a search query from the parsed caption (plaintiff, defendant, year, citation), (2) searches CourtListener for the matching opinion, (3) scores candidates on caption similarity / year / jurisdiction, (4) downloads the opinion text, HTML, and PDF, and (5) writes a per-case metadata row. Falls back when a case is unfindable (CourtListener has uneven state-court coverage).

Optional environment variable: `CL_API_TOKEN` (raises the rate limit from ~5,000/day to ~10,000/hr).

### `code/02_extract_features.py` — clean / extract features
Ingests each `.docx` (or `.txt`) judicial opinion, segments it into sentences, and extracts the 140+ per-case linguistic features that end up as columns in `data/cases.csv`. Features include readability scores, part-of-speech distributions, Empath topic-vocabulary rates, VADER sentiment (document- and paragraph-level), and a set of **19 sexual-assault-specific framing detectors** (victim-blaming, credibility challenges, procedural deflection, hedging, perpetrator naming, direct-violence vocabulary, etc.). Outputs a sentence-level annotated `.xlsx` and a per-case stats `.json`.

### `code/03_analyze_and_visualize.py` — analyze
Loads `data/cases.csv`, computes within-stratum statistics (Mann-Whitney U, Cliff's δ with bootstrap CIs, Benjamini-Hochberg FDR correction across all 19 framing detectors), and regenerates the full set of thesis figures. This is the analytical component of the project.

### `code/04_milestone1_figures.py` — analyze (minimal example)
A shorter, portable script that reproduces just the two starter figures from Milestone 1 directly from `data/cases.csv`. Use this if you want a clean entry point for understanding the data. It demonstrates the pandas + matplotlib workflow without the full statistical machinery.

---

## How to reproduce

The Milestone 1 figures regenerate from `data/cases.csv` alone:

```bash
pip install pandas numpy matplotlib
python code/04_milestone1_figures.py
```

This will write `figure1_corpus_composition.png` and `figure2_direct_violence_by_gender.png` into `output/`.

The full analysis script (`03_analyze_and_visualize.py`) regenerates every thesis figure and stats table from `data/cases.csv`. It additionally requires `scipy` and assumes a specific thesis-folder layout, which is documented at the top of the script.

The data-pull (`01_data_pull.py`) and feature-extraction (`02_extract_features.py`) scripts require the raw `.docx` judicial opinions, which are not committed to this repo. See the **Data** section below.

---

## Data

The processed dataset `data/cases.csv` has **one row per case** (n = 97 U.S. judicial opinions) and ~140 columns. Key column families:

| Column family | Examples | What it captures |
|---|---|---|
| Identifiers | `case_id`, `case_caption`, `plaintiff_last`, `defendant_last`, `year`, `citation` | Per-case metadata |
| Stratification | `folder_victim_gender_label`, `is_govt_prosecution`, `appellate_criminal_flip` | Male/female label, criminal/civil indicator |
| Structural / readability | `n_sentences`, `flesch_kincaid_grade`, `gunning_fog`, `avg_sentence_length` | Document shape |
| Part of speech | `pct_noun`, `pct_verb`, `pct_passive_aux`, etc. | spaCy POS share |
| Empath topic vocab | `empath_crime`, `empath_violence`, `empath_power`, `empath_masculine`, … | 30+ topic-vocabulary rates |
| VADER sentiment | `vader_doc_compound`, `vader_para_mean_compound`, etc. | Document + paragraph sentiment |
| **Framing detectors** | `sa_direct_violence_term`, `sa_victim_blame_direct`, `sa_credibility_challenge`, `sa_perp_ref_name`, `sa_clinical_legal_term`, `sa_hedging_allegation`, … | The 19 custom detectors — raw counts; divide by `sa_total_sentences` for per-case rates |

The **raw `.docx` judicial opinions** are not committed to this public repository because they are large and were retrieved through CourtListener / the Caselaw Access Project under terms that favor not re-hosting bulk court documents publicly. They are available on request for replication purposes.

---

## Key finding (so far)

Within criminal cases, female-victim opinions name the violence directly **much more often** than male-victim opinions. The `sa_direct_violence_term` rate has a Cliff's δ of **−0.61** (q_FDR < 0.01, Bonferroni-significant), meaning in roughly 80% of male-vs-female case pairs the female-victim opinion has the higher direct-violence-term rate. See `output/figure2_direct_violence_by_gender.png`.

Perpetrator naming runs in the opposite direction (δ = +0.57, q_FDR < 0.01), so male-victim opinions name the perpetrator by surname / proper noun more often. The full forest plot of all 19 detectors is at `output/09_forest_plot.png`.

---

## Citation

If you use this code or dataset, please cite the senior thesis (in progress) and link back to this repository.
