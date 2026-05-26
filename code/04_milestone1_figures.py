"""
04_milestone1_figures.py
========================
Reproduces the two Milestone 1 starter visualizations from the processed
case-level dataset (data/cases.csv). Uses pandas for data shaping and
matplotlib for plotting, in the soft pink/blue palette established for
the thesis figures.

Outputs (saved to output/):
    figure1_corpus_composition.png       Bar chart of victim gender x case type
    figure2_direct_violence_by_gender.png Boxplot of direct-violence-term rate,
                                          criminal cases, by victim gender

Run from the repository root:
    python code/04_milestone1_figures.py

Or from anywhere:
    python /path/to/qss20-final-project/code/04_milestone1_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Paths (resolved relative to this script so the repo is portable)
# -------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
CSV  = REPO / "data"   / "cases.csv"
OUT  = REPO / "output"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Aesthetic constants (matched to the thesis figure palette)
# -------------------------------------------------------------------
FEMALE_FILL  = "#F4B5BD"
FEMALE_EDGE  = "#D88997"
MALE_FILL    = "#B5C8D8"
MALE_EDGE    = "#7A95B5"

TITLE_COLOR    = "#1F2937"
SUBTITLE_COLOR = "#3A4A5C"
BODY_COLOR     = "#4A5563"
GRID_COLOR     = "#E8E8E8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": BODY_COLOR,
    "axes.labelcolor": BODY_COLOR,
    "xtick.color": BODY_COLOR,
    "ytick.color": BODY_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# -------------------------------------------------------------------
# Load + derive
# -------------------------------------------------------------------
df = pd.read_csv(CSV)
df["stratum"]       = np.where(df["is_govt_prosecution"], "Criminal", "Civil")
df["victim_gender"] = df["folder_victim_gender_label"].str.capitalize()
df["sa_direct_violence_term_rate"] = (
    df["sa_direct_violence_term"] / df["sa_total_sentences"]
)

# ===================================================================
# FIGURE 1: Corpus composition - victim gender x case type
# ===================================================================
ct = pd.crosstab(df["stratum"], df["victim_gender"])[["Female", "Male"]]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = np.arange(len(ct.index))
w = 0.36

bars_f = ax.bar(x - w / 2, ct["Female"], w,
                color=FEMALE_FILL, edgecolor=FEMALE_EDGE, linewidth=1.2,
                label="Female victim")
bars_m = ax.bar(x + w / 2, ct["Male"], w,
                color=MALE_FILL, edgecolor=MALE_EDGE, linewidth=1.2,
                label="Male victim")

for bars in (bars_f, bars_m):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{int(b.get_height())}", ha="center", va="bottom",
                fontsize=10, color=BODY_COLOR)

ax.set_xticks(x)
ax.set_xticklabels(ct.index, color=BODY_COLOR)
ax.set_ylabel("Number of cases", color=BODY_COLOR)
ax.set_xlabel("Case type", color=BODY_COLOR)
ax.set_title("Figure 1. Corpus composition by victim gender and case type",
             loc="left", fontsize=12, fontweight="bold", color=TITLE_COLOR,
             pad=24)
ax.text(0, 1.04,
        "n = 97 U.S. judicial opinions; female victims concentrate in criminal "
        "prosecutions, male victims in civil suits.",
        transform=ax.transAxes, fontsize=9.5, color=SUBTITLE_COLOR,
        style="italic", ha="left", va="bottom")
ax.legend(loc="upper right", frameon=False, labelcolor=BODY_COLOR)
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, max(ct.values.max() + 6, 40))
plt.tight_layout()
plt.savefig(OUT / "figure1_corpus_composition.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'figure1_corpus_composition.png'}")

# ===================================================================
# FIGURE 2: Direct-violence-term rate, by victim gender (criminal stratum)
# ===================================================================
crim   = df[df["stratum"] == "Criminal"].copy()
female = crim.loc[crim["victim_gender"] == "Female", "sa_direct_violence_term_rate"]
male   = crim.loc[crim["victim_gender"] == "Male",   "sa_direct_violence_term_rate"]

fig, ax = plt.subplots(figsize=(7.2, 4.6))

bp = ax.boxplot([female, male], positions=[1, 2], widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=2.4),
                whiskerprops=dict(color=BODY_COLOR, linewidth=1),
                capprops=dict(color=BODY_COLOR, linewidth=1),
                boxprops=dict(linewidth=1.2))
for patch, fill, edge in zip(bp["boxes"],
                              [FEMALE_FILL, MALE_FILL],
                              [FEMALE_EDGE, MALE_EDGE]):
    patch.set_facecolor(fill)
    patch.set_edgecolor(edge)
    patch.set_alpha(0.9)

rng = np.random.default_rng(7)
for i, (vals, edge) in enumerate(zip([female, male],
                                      [FEMALE_EDGE, MALE_EDGE]), start=1):
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               s=28, color=edge, alpha=0.7, edgecolor="white", linewidth=0.6,
               zorder=3)

ax.set_xticks([1, 2])
ax.set_xticklabels([f"Female victim\nn = {len(female)}",
                    f"Male victim\nn = {len(male)}"],
                   color=BODY_COLOR)
ax.set_ylabel("Direct-violence-term rate\n(share of sentences)", color=BODY_COLOR)
ax.set_title("Figure 2. Direct-violence vocabulary by victim gender, criminal cases",
             loc="left", fontsize=12, fontweight="bold", color=TITLE_COLOR,
             pad=24)
ax.text(0, 1.04,
        "Each point is one judicial opinion. Cliff's $\\delta$ = $-$0.61, "
        "q$_{FDR}$ < 0.01 - female-victim opinions name the violence directly more often.",
        transform=ax.transAxes, fontsize=9.5, color=SUBTITLE_COLOR,
        style="italic", ha="left", va="bottom")
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "figure2_direct_violence_by_gender.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'figure2_direct_violence_by_gender.png'}")
print(f"Female median: {female.median():.3f}  Male median: {male.median():.3f}")
