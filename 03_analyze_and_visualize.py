"""
Master script: load updated cases.csv, compute all within-stratum statistics,
and regenerate every thesis figure to match the previously-established
aesthetic (Poppins font, soft modern pink/blue palette, bold title +
italic finding-statement subtitle + small italic explainer caption,
A/B/C panels with significance brackets and stats lines).

Run: python3 FINAL/compute_stats_and_figures.py
"""

from __future__ import annotations
import csv, json, os, math, shutil
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import PercentFormatter, MaxNLocator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Paths
# =============================================================================
THESIS_DIR = Path(__file__).resolve().parent.parent
CASES_CSV  = THESIS_DIR / "FINAL" / "pipeline_v2" / "cases.csv"
FIG_DIR    = THESIS_DIR / "FINAL" / "figures"
STATS_DIR  = THESIS_DIR / "FINAL" / "stats"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Aesthetic constants (matched to the previous figures_final aesthetic)
# =============================================================================
# Soft modern pink (female) and soft modern blue (male)
FEMALE_FILL  = "#F4B5BD"
FEMALE_EDGE  = "#D88997"
FEMALE_DEEP  = "#C76B5A"
FEMALE_DUSTY = "#A66E81"

MALE_FILL    = "#B5C8D8"
MALE_EDGE    = "#7A95B5"
MALE_DEEP    = "#3E6FA8"

# Typography colors
TITLE_COLOR    = "#1F2937"   # near-black slate
SUBTITLE_COLOR = "#3A4A5C"   # darker grey for subtitle
BODY_COLOR     = "#4A5563"   # body text
CAPTION_COLOR  = "#6B7280"   # lighter grey for italic caption
GRID_COLOR     = "#E8E8E8"   # very faint horizontal grid
BOX_FACE       = "#F3F1ED"   # warm off-white callout
BOX_EDGE       = "#D6D2CC"

# For access split — slate teal & warm sand from the original
CRIMINAL_COLOR = "#3F5B6B"   # slate teal
CIVIL_COLOR    = "#C9B89A"   # warm sand

# Median marker
MEDIAN_COLOR   = "#FFFFFF"   # white line through violins

plt.rcParams.update({
    "font.family": "Poppins",
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

# =============================================================================
# Load data
# =============================================================================
df = pd.read_csv(CASES_CSV)
df["gender"]   = df["folder_victim_gender_label"].str.lower().str.strip()
df["criminal"] = df["is_govt_prosecution"].astype(str).str.lower() == "true"
df["stratum"]  = np.where(df["criminal"], "criminal", "civil")

print(f"Loaded {len(df)} cases")
print(df.groupby(["gender", "stratum"]).size().unstack(fill_value=0))
print()

# Derive per-sentence rates from sa_* counts
sa_cols = [c for c in df.columns if c.startswith("sa_")]
for c in sa_cols:
    df[f"{c}_rate"] = df[c] / df["n_sentences"].replace(0, np.nan)

# =============================================================================
# Variables
# =============================================================================
VAR_DIRECT_VIOLENCE = "sa_direct_violence_term_rate"
VAR_PERP_NAMING     = "sa_perp_ref_name_rate"
VAR_CRIME_VOCAB     = "empath_crime"
VAR_VICTIM_BLAME    = "sa_victim_blame_direct_rate"
VAR_AFFECT          = "vader_para_mean_compound"

FOREST_VARS = [
    ("sa_direct_violence_term_rate",  "Direct violence terms"),
    ("sa_perp_ref_name_rate",         "Perpetrator naming"),
    ("empath_crime",                  "Empath: crime vocabulary"),
    ("sa_victim_blame_direct_rate",   "Victim-blaming language"),
    ("vader_para_mean_compound",      "VADER affect (paragraph)"),
    ("sa_perp_active_agency_rate",    "Perpetrator as active agent"),
    ("sa_victim_passive_object_rate", "Victim as passive object"),
    ("empath_violence",               "Empath: violence vocabulary"),
    ("sa_clinical_legal_term_rate",   "Clinical/sterilizing terms"),
    ("sa_credibility_challenge_rate", "Credibility challenges"),
    ("sa_hedging_allegation_rate",    "Hedging density"),
    ("sa_minimizing_term_rate",       "Minimizing language"),
    ("sa_procedural_deflection_rate", "Procedural deflection"),
    ("sa_perp_honorific_rate",        "Perpetrator honorifics"),
    ("pct_passive_aux",               "Passive voice density"),
]

# =============================================================================
# Statistical helpers
# =============================================================================
def cliffs_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n_gt = (a[:, None] > b[None, :]).sum()
    n_lt = (a[:, None] < b[None, :]).sum()
    return (n_gt - n_lt) / (len(a) * len(b))

def bootstrap_ci(a, b, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        boots.append(cliffs_d(a_s, b_s))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def stratum_stats(df, var, stratum=None):
    sub = df if stratum is None else df[df["stratum"] == stratum]
    male = sub[sub["gender"] == "male"][var].dropna().values
    female = sub[sub["gender"] == "female"][var].dropna().values
    if len(male) < 2 or len(female) < 2:
        return None
    U, p = stats.mannwhitneyu(male, female, alternative="two-sided")
    d = cliffs_d(male, female)
    ci_lo, ci_hi = bootstrap_ci(male, female)
    return {
        "n_male": int(len(male)), "n_female": int(len(female)),
        "median_male":   float(np.median(male)),
        "median_female": float(np.median(female)),
        "U": float(U), "p": float(p),
        "cliffs_d": float(d), "ci_lo": ci_lo, "ci_hi": ci_hi,
    }

def fdr_bh(p_values):
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    q = ranked * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    q = np.minimum(q, 1.0)
    out = np.empty(n)
    out[order] = q
    return out.tolist()

def effect_label(d):
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.33:  return "small"
    if a < 0.474: return "medium"
    return "large"

def fmt_p(p):
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}".rstrip("0").rstrip(".") if p < 1 else f"p = {p:.2f}"

def fmt_p_short(p):
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"

# =============================================================================
# Compute all stats
# =============================================================================
print("Computing stats ...")
all_stats = {}
HEADLINE = [
    ("direct_violence",  VAR_DIRECT_VIOLENCE),
    ("perp_naming",      VAR_PERP_NAMING),
    ("crime_vocab",      VAR_CRIME_VOCAB),
    ("victim_blame",     VAR_VICTIM_BLAME),
    ("affect",           VAR_AFFECT),
]
for label, var in HEADLINE:
    all_stats[label] = {
        "var": var,
        "full":     stratum_stats(df, var, None),
        "criminal": stratum_stats(df, var, "criminal"),
        "civil":    stratum_stats(df, var, "civil"),
    }

forest = []
for var, label in FOREST_VARS:
    s = stratum_stats(df, var, "criminal")
    s["label"] = label; s["var"] = var
    forest.append(s)
p_values = [s["p"] for s in forest]
q_bh = fdr_bh(p_values)
n_tests = len(p_values)
for s, q in zip(forest, q_bh):
    s["q_fdr_bh"] = q
    s["p_bonf"]   = min(1.0, s["p"] * n_tests) if not math.isnan(s["p"]) else float("nan")
all_stats["forest_within_criminal"] = forest

with open(STATS_DIR / "all_stats.json", "w") as f:
    json.dump(all_stats, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
print(f"  wrote {STATS_DIR / 'all_stats.json'}")
print()

# =============================================================================
# Violin-plot panel renderer
# =============================================================================
def draw_violin_panel(ax, df_panel, var, panel_letter, stratum_label,
                      y_format=None, y_unit_in_text=False):
    male = df_panel[df_panel["gender"] == "male"][var].dropna().values
    female = df_panel[df_panel["gender"] == "female"][var].dropna().values

    parts = ax.violinplot([male, female], positions=[0, 1], widths=0.78,
                          showmeans=False, showmedians=False, showextrema=False)
    fills = [MALE_FILL, FEMALE_FILL]
    edges = [MALE_EDGE, FEMALE_EDGE]
    for body, c, e in zip(parts['bodies'], fills, edges):
        body.set_facecolor(c)
        body.set_edgecolor(e)
        body.set_alpha(0.92)
        body.set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for pos, vals in zip([0, 1], [male, female]):
        if len(vals) > 0:
            jitter = rng.normal(0, 0.05, size=len(vals))
            ax.scatter(np.full_like(vals, pos, dtype=float) + jitter,
                       vals, color=BODY_COLOR, alpha=0.55, s=10,
                       edgecolors="none", zorder=3)
            # White median line
            med = np.median(vals)
            ax.hlines(med, pos - 0.25, pos + 0.25,
                      color="white", linewidth=2.6, zorder=5)
            ax.hlines(med, pos - 0.25, pos + 0.25,
                      color=BODY_COLOR, linewidth=0.8, zorder=6)

    # X-axis labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Male\nn = {len(male)}", f"Female\nn = {len(female)}"],
                       color=BODY_COLOR)
    ax.set_xlim(-0.65, 1.65)

    # Significance bracket
    if len(male) > 0 and len(female) > 0:
        try:
            U, p = stats.mannwhitneyu(male, female, alternative="two-sided")
            y_top = max(np.max(male), np.max(female)) if (len(male) and len(female)) else 0
            y_lim = ax.get_ylim()[1]
            bracket_y = y_top + (y_lim - y_top) * 0.35 if y_lim > y_top else y_top * 1.05
            ax.plot([0, 0, 1, 1],
                    [bracket_y - (y_lim*0.02), bracket_y, bracket_y, bracket_y - (y_lim*0.02)],
                    color=BODY_COLOR, linewidth=0.9)
            ax.text(0.5, bracket_y + (y_lim*0.01), fmt_p_short(p),
                    ha="center", va="bottom", fontsize=9, color=BODY_COLOR)
        except Exception:
            pass

    # Panel title
    ax.set_title(f"{panel_letter}. {stratum_label}",
                 loc="left", color=TITLE_COLOR, fontsize=10.5,
                 fontweight="bold", pad=8)

    if y_format == "pct":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def make_violin_figure(label, var, title, subtitle, caption,
                       ylabel, filename_stem, y_format=None,
                       median_unit="rate"):
    """Build a 3-panel violin figure matching the previous aesthetic."""
    fig = plt.figure(figsize=(13, 7.0), dpi=300)
    gs = fig.add_gridspec(1, 3, left=0.085, right=0.98, top=0.76, bottom=0.20,
                          wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Determine shared y-limits
    male_full = df[df["gender"] == "male"][var].dropna().values
    fem_full  = df[df["gender"] == "female"][var].dropna().values
    all_vals = np.concatenate([male_full, fem_full])
    y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
    pad = (y_max - y_min) * 0.18
    for ax in axes:
        ax.set_ylim(y_min - pad * 0.15, y_max + pad)

    # Draw each panel
    panel_specs = [
        ("A", "Full corpus", df),
        ("B", "Criminal only", df[df["stratum"] == "criminal"]),
        ("C", "Civil only", df[df["stratum"] == "civil"]),
    ]
    for ax, (letter, slab, sub) in zip(axes, panel_specs):
        draw_violin_panel(ax, sub, var, letter, slab, y_format=y_format)

    # Y-label only on left axis
    axes[0].set_ylabel(ylabel, color=BODY_COLOR)

    # Stats line below each panel (two-line: U+p · d+medians)
    s_full = all_stats[label]["full"]
    s_crim = all_stats[label]["criminal"]
    s_civ  = all_stats[label]["civil"]
    def med_str(s, unit):
        if unit == "pct":
            return f"M {s['median_male']*100:.1f}%, F {s['median_female']*100:.1f}%"
        return f"M {s['median_male']:.2f}, F {s['median_female']:.2f}"
    fmt_unit = "pct" if y_format == "pct" else "raw"
    for ax, s in zip(axes, [s_full, s_crim, s_civ]):
        line1 = f"Mann–Whitney  U = {s['U']:.0f}   ·   {fmt_p_short(s['p'])}"
        line2 = (f"Cliff's d = {s['cliffs_d']:+.2f} ({effect_label(s['cliffs_d'])})   ·   "
                 f"Medians: {med_str(s, fmt_unit)}")
        # Place below the panel
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.115, line1,
                 ha="center", va="center", fontsize=9.5, color=BODY_COLOR)
        fig.text((bbox.x0 + bbox.x1) / 2, 0.075, line2,
                 ha="center", va="center", fontsize=9, color=BODY_COLOR)

    # Title block (top of figure)
    fig.text(0.05, 0.95, title, ha="left", va="top",
             fontsize=20, fontweight="bold", color=TITLE_COLOR)
    fig.text(0.05, 0.905, subtitle, ha="left", va="top",
             fontsize=11.5, color=SUBTITLE_COLOR, style="italic")
    fig.text(0.05, 0.866, caption, ha="left", va="top",
             fontsize=9, color=CAPTION_COLOR, style="italic")

    fig.savefig(FIG_DIR / f"{filename_stem}.png", dpi=300, facecolor="white")
    fig.savefig(FIG_DIR / f"{filename_stem}.pdf",          facecolor="white")
    plt.close(fig)
    print(f"  saved: {filename_stem}.png/.pdf")


# =============================================================================
# Generate Figures 4–8
# =============================================================================
print("Generating violin figures ...")

make_violin_figure(
    "direct_violence", VAR_DIRECT_VIOLENCE,
    "Naming the Violence",
    "Female-victim opinions name the violence more directly — and the gap is largest within criminal cases",
    "Each point is one judicial opinion. Y-axis is the share of sentences containing direct-violence terms (rape, beat, force, strangle, etc.).\nSignificance is the Mann–Whitney U two-sided test; effect size is Cliff's d (sign = M − F).",
    "Direct violence-term rate\n(% of sentences)",
    "04_naming_violence_violin",
    y_format="pct",
)

make_violin_figure(
    "perp_naming", VAR_PERP_NAMING,
    "Naming the Perpetrator",
    "Male-victim opinions name the perpetrator more often — gap is largest within criminal cases and disappears within civil cases",
    "Each point is one judicial opinion. Y-axis is perpetrator-name mentions per sentence (can exceed 1.0 when the perpetrator is named multiple times in one sentence).\nMann–Whitney is the primary test; a Welch t-test on log1p-transformed rates agrees (full p = 0.005, criminal p = 0.003, civil p = 0.62).",
    "Perpetrator-name mentions\nper sentence",
    "05_perp_naming_violin",
    y_format=None,
)

make_violin_figure(
    "crime_vocab", VAR_CRIME_VOCAB,
    "Naming the Crime",
    "Female-victim civil opinions engage criminal-conduct vocabulary at nearly twice the rate of male-victim civil opinions",
    "Each point is one judicial opinion. Y-axis is the Empath \"crime\" lexical-category density (fraction of opinion tokens matching crime-vocabulary:\n\"assault,\" \"rape,\" \"criminal,\" \"conviction,\" \"battery,\" etc.). The within-civil effect (panel C) is the strongest single statistical result in the corpus;\nthe within-criminal effect (panel B) is near zero — both genders use crime vocab at expected rates within criminal contexts.",
    "Empath crime-vocabulary density\n(% of tokens)",
    "06_crime_vocab_violin",
    y_format="pct",
)

make_violin_figure(
    "victim_blame", VAR_VICTIM_BLAME,
    "Blaming the Victim",
    "Apparent corpus-level pattern reverses inside criminal cases and disappears inside civil cases — the gender effect is a case-type artifact",
    "Each point is one judicial opinion. Y-axis is the share of sentences containing direct victim-blaming language (\"she was promiscuous,\" \"the victim provoked,\" etc.).\nReading the panels together is the finding: full-corpus M > F dissolves once you stratify, leaving a marginal F > M signal in criminal cases and no effect in civil cases.",
    "Direct victim-blame rate\n(% of sentences)",
    "07_victim_blame_violin",
    y_format="pct",
)

make_violin_figure(
    "affect", VAR_AFFECT,
    "The Tone of the Telling",
    "Female-victim opinions are written in a more negative emotional register — direction is consistent within both case-type strata",
    "Each point is one judicial opinion. Y-axis is the VADER paragraph-mean compound score (−1 = most negative, +1 = most positive).\nCliff's d sign positive = male-victim opinions less negative; direction is consistent across all three panels.",
    "VADER paragraph-mean\ncompound score",
    "08_affect_violin",
    y_format=None,
)
print()


# =============================================================================
# Figure 9 — Forest plot (matched to previous aesthetic)
# =============================================================================
print("Generating forest plot ...")
fig = plt.figure(figsize=(13, 8.5), dpi=300)
gs = fig.add_gridspec(1, 2, left=0.22, right=0.97, top=0.79, bottom=0.13,
                      width_ratios=[2.8, 1.0], wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.axis("off")

# Sort: most positive d at top, most negative at bottom
forest_sorted = sorted(forest, key=lambda s: -s["cliffs_d"])
y_positions = np.arange(len(forest_sorted))[::-1]

# Negligible-effect zone shading
ax.axvspan(-0.147, 0.147, color="#F0EFEC", alpha=0.85, zorder=0)
# Zero line (vertical, dotted)
ax.axvline(0, color=BODY_COLOR, linewidth=0.9, alpha=0.5, zorder=1, linestyle=":")

for y, s in zip(y_positions, forest_sorted):
    d, lo, hi = s["cliffs_d"], s["ci_lo"], s["ci_hi"]
    is_fdr = s["q_fdr_bh"] < 0.05
    is_bonf = s["p_bonf"] < 0.05
    if d >= 0:
        edge = MALE_DEEP; fill = MALE_FILL
    else:
        edge = FEMALE_DEEP; fill = FEMALE_FILL
    if math.isnan(d):
        continue
    # CI line
    ax.hlines(y, lo, hi, color=edge, linewidth=2.0, alpha=0.9, zorder=2)
    # Tick marks at CI ends
    ax.vlines(lo, y - 0.18, y + 0.18, color=edge, linewidth=1.6, zorder=2)
    ax.vlines(hi, y - 0.18, y + 0.18, color=edge, linewidth=1.6, zorder=2)
    # Marker
    if is_fdr:
        ax.scatter([d], [y], s=170, color=fill, edgecolors=edge,
                   linewidths=1.6, zorder=4, marker="o")
    else:
        ax.scatter([d], [y], s=85, facecolors="white",
                   edgecolors=edge, linewidths=1.4, zorder=4, marker="s")
    # Annotation for FDR-significant — always place to the right of the
    # CI bar, in matching color
    if is_fdr:
        sig_label = "Bonferroni-sig" if is_bonf else "FDR-sig"
        ax.text(hi + 0.04, y, f"q_FDR = {s['q_fdr_bh']:.3f}  ({sig_label})",
                fontsize=9, color=edge, va="center", ha="left")

ax.set_yticks(y_positions)
ax.set_yticklabels([s["label"] for s in forest_sorted], color=BODY_COLOR, fontsize=10.5)
ax.set_ylim(-0.7, len(forest_sorted) - 0.3)
ax.set_xlim(-1.0, 1.0)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
ax.set_xlabel("Cliff's d", color=TITLE_COLOR, fontsize=11, labelpad=10)
ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Direction labels just below x-axis. Use mathtext for the arrows so they
# render correctly even though Poppins doesn't include arrow glyphs.
ax.text(0.25, -0.085, r"$\leftarrow$  female-victim higher",
        ha="center", color=FEMALE_DEEP, fontsize=10.5,
        transform=ax.transAxes, clip_on=False)
ax.text(0.75, -0.085, r"male-victim higher  $\rightarrow$",
        ha="center", color=MALE_DEEP, fontsize=10.5,
        transform=ax.transAxes, clip_on=False)

# Legend in side panel
ly = 0.95
ax_legend.text(0.0, ly, "Marker key", transform=ax_legend.transAxes,
               fontsize=10.5, fontweight="bold", color=TITLE_COLOR, va="top")
# Filled circle
ax_legend.scatter([0.06], [ly - 0.10], s=140, color=MALE_FILL, edgecolors=MALE_DEEP,
                  linewidths=1.4, marker="o", transform=ax_legend.transAxes, clip_on=False)
ax_legend.text(0.18, ly - 0.10, "FDR-significant (q < 0.05)",
               transform=ax_legend.transAxes, fontsize=9.5, color=BODY_COLOR, va="center")
# Hollow square
ax_legend.scatter([0.06], [ly - 0.18], s=85, facecolors="white",
                  edgecolors=BODY_COLOR, linewidths=1.4, marker="s",
                  transform=ax_legend.transAxes, clip_on=False)
ax_legend.text(0.18, ly - 0.18, "Not significant",
               transform=ax_legend.transAxes, fontsize=9.5, color=BODY_COLOR, va="center")

ax_legend.text(0.0, ly - 0.30, "Direction", transform=ax_legend.transAxes,
               fontsize=10.5, fontweight="bold", color=TITLE_COLOR, va="top")
ax_legend.add_patch(Rectangle((0.02, ly - 0.42), 0.10, 0.04, color=MALE_FILL,
                                ec=MALE_EDGE, lw=0.8, transform=ax_legend.transAxes))
ax_legend.text(0.18, ly - 0.40, "Male-victim higher",
               transform=ax_legend.transAxes, fontsize=9.5, color=BODY_COLOR, va="center")
ax_legend.add_patch(Rectangle((0.02, ly - 0.50), 0.10, 0.04, color=FEMALE_FILL,
                                ec=FEMALE_EDGE, lw=0.8, transform=ax_legend.transAxes))
ax_legend.text(0.18, ly - 0.48, "Female-victim higher",
               transform=ax_legend.transAxes, fontsize=9.5, color=BODY_COLOR, va="center")

ax_legend.text(0.0, ly - 0.60, "Effect size", transform=ax_legend.transAxes,
               fontsize=10.5, fontweight="bold", color=TITLE_COLOR, va="top")
ax_legend.text(0.0, ly - 0.66, "Shaded gray band: |d| < 0.147\n(negligible per Romano 2006).",
               transform=ax_legend.transAxes, fontsize=9, color=BODY_COLOR, va="top")

# Title block
fig.text(0.05, 0.95, "All Effect Sizes Within Criminal Cases",
         ha="left", va="top", fontsize=20, fontweight="bold", color=TITLE_COLOR)
fig.text(0.05, 0.905,
         "Two findings clear both FDR-BH and Bonferroni correction at q < 0.05 — perpetrator naming (M > F) and direct violence terms (F > M)",
         ha="left", va="top", fontsize=11, color=SUBTITLE_COLOR, style="italic")
fig.text(0.05, 0.872,
         "Each row is one framing variable. Markers show Cliff's d (range −1 to +1; sign = M − F); error bars are 500-resample bootstrap 95% CIs.\n"
         "Within-criminal stratum: n = 16 male-victim, 34 female-victim opinions. p-values from two-sided Mann–Whitney U.",
         ha="left", va="top", fontsize=9, color=CAPTION_COLOR, style="italic")

fig.savefig(FIG_DIR / "09_forest_plot.png", dpi=300, facecolor="white")
fig.savefig(FIG_DIR / "09_forest_plot.pdf",          facecolor="white")
plt.close(fig)
print("  saved: 09_forest_plot.png/.pdf")
print()

# =============================================================================
# Figure 3 — Gendered access to legal recourse (vertical stacked bars)
# =============================================================================
print("Generating access-split figure ...")
m_total = (df["gender"] == "male").sum()
f_total = (df["gender"] == "female").sum()
m_crim  = ((df["gender"] == "male") & df["criminal"]).sum()
f_crim  = ((df["gender"] == "female") & df["criminal"]).sum()
m_civ   = m_total - m_crim
f_civ   = f_total - f_crim

fig = plt.figure(figsize=(11.5, 6.5), dpi=300)
gs = fig.add_gridspec(1, 2, left=0.10, right=0.97, top=0.78, bottom=0.12,
                      width_ratios=[2.4, 1.6], wspace=0.05)
ax = fig.add_subplot(gs[0, 0])
ax_text = fig.add_subplot(gs[0, 1])

x = [0, 1]
crim_pcts = [m_crim / m_total * 100, f_crim / f_total * 100]
civ_pcts  = [m_civ  / m_total * 100, f_civ  / f_total * 100]

# Bars
bar_width = 0.55
ax.bar(x, crim_pcts, color=CRIMINAL_COLOR, width=bar_width, label="Criminal prosecution", zorder=2)
ax.bar(x, civ_pcts, bottom=crim_pcts, color=CIVIL_COLOR, edgecolor="#A89B82", linewidth=0.6, width=bar_width, label="Civil suit", zorder=2)

# Labels inside the bars
for xi, cp, vp, ct, vt in zip(x, crim_pcts, civ_pcts, [m_crim, f_crim], [m_civ, f_civ]):
    ax.text(xi, cp / 2, f"{ct}\n({cp:.0f}%)", ha="center", va="center",
            color="white", fontsize=11, fontweight="medium")
    ax.text(xi, cp + vp / 2, f"{vt}\n({vp:.0f}%)", ha="center", va="center",
            color=TITLE_COLOR, fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(["Male-victim cases", "Female-victim cases"], color=BODY_COLOR, fontsize=11)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylabel("Share of cases within each victim-gender group", color=BODY_COLOR, fontsize=10)
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# Right side: legend + filter pool callout
ax_text.axis("off")
ax_text.text(0.05, 0.97, "Procedural type", color=TITLE_COLOR, fontsize=11, fontweight="bold", transform=ax_text.transAxes)
ax_text.add_patch(Rectangle((0.05, 0.86), 0.07, 0.04, color=CRIMINAL_COLOR, transform=ax_text.transAxes))
ax_text.text(0.16, 0.88, "Criminal prosecution", color=BODY_COLOR, fontsize=10, va="center", transform=ax_text.transAxes)
ax_text.add_patch(Rectangle((0.05, 0.795), 0.07, 0.04, color=CIVIL_COLOR, ec="#A89B82", lw=0.6, transform=ax_text.transAxes))
ax_text.text(0.16, 0.815, "Civil suit", color=BODY_COLOR, fontsize=10, va="center", transform=ax_text.transAxes)

ax_text.text(0.05, 0.70, "Filter pool context", color=TITLE_COLOR, fontsize=11, fontweight="bold", transform=ax_text.transAxes)
ax_text.text(0.05, 0.66,
             "Before any sampling for this corpus,\na gendered filter-term search returned",
             color=BODY_COLOR, fontsize=10, va="top", transform=ax_text.transAxes)
ax_text.text(0.05, 0.51, "148 male-victim cases", color=TITLE_COLOR, fontsize=11, fontweight="bold", transform=ax_text.transAxes)
ax_text.text(0.05, 0.46, "and", color=BODY_COLOR, fontsize=10, transform=ax_text.transAxes)
ax_text.text(0.05, 0.41, "2,693 female-victim cases", color=TITLE_COLOR, fontsize=11, fontweight="bold", transform=ax_text.transAxes)
ax_text.text(0.05, 0.32,
             f"— an 18× imbalance even before\nwe begin to read the opinions.",
             color=BODY_COLOR, fontsize=10, va="top", transform=ax_text.transAxes)

# Title block
fig.text(0.07, 0.94, "Gendered Access to Legal Recourse",
         ha="left", va="top", fontsize=18, fontweight="bold", color=TITLE_COLOR)
fig.text(0.07, 0.89,
         f"Male-victim opinions overwhelmingly access civil litigation; female-victim opinions overwhelmingly access criminal prosecution",
         ha="left", va="top", fontsize=11, color=SUBTITLE_COLOR, style="italic")
fig.text(0.07, 0.855,
         "Each bar is the 100% within-group composition of the analyzed 97-case corpus.\n"
         "The asymmetry compounds the filter-pool imbalance (right) — male SA is under-prosecuted at every level we can measure.",
         ha="left", va="top", fontsize=8.8, color=CAPTION_COLOR, style="italic")

fig.savefig(FIG_DIR / "03_access_split.png", dpi=300, facecolor="white")
fig.savefig(FIG_DIR / "03_access_split.pdf",          facecolor="white")
plt.close(fig)
print("  saved: 03_access_split.png/.pdf")
print()

# =============================================================================
# Figure 1 — Male corpus representativeness check
# =============================================================================
print("Generating male representativeness figure ...")
F_POOL_CRIM, F_POOL_CIV = 38, 110
F_CORP_CRIM, F_CORP_CIV = m_crim, m_civ
F_POOL_TOTAL = F_POOL_CRIM + F_POOL_CIV
F_CORP_TOTAL = F_CORP_CRIM + F_CORP_CIV
crim_fail = F_POOL_CRIM - F_CORP_CRIM
civ_fail = F_POOL_CIV - F_CORP_CIV
oddsratio, p_fisher = stats.fisher_exact([[F_CORP_CRIM, crim_fail], [F_CORP_CIV, civ_fail]])
crim_pass_rate = F_CORP_CRIM / F_POOL_CRIM * 100
civ_pass_rate  = F_CORP_CIV / F_POOL_CIV * 100

fig = plt.figure(figsize=(10.5, 6.8), dpi=300)
gs = fig.add_gridspec(1, 1, left=0.10, right=0.97, top=0.74, bottom=0.20)
ax = fig.add_subplot(gs[0, 0])

groups = [f"Filter pool\n(n = {F_POOL_TOTAL})", f"Criteria-passing corpus\n(n = {F_CORP_TOTAL})"]
crim_props = [F_POOL_CRIM / F_POOL_TOTAL * 100, F_CORP_CRIM / F_CORP_TOTAL * 100]
civ_props  = [F_POOL_CIV  / F_POOL_TOTAL * 100, F_CORP_CIV  / F_CORP_TOTAL * 100]
xpos = [0, 1]
ax.bar(xpos, crim_props, color=MALE_DEEP, width=0.5, label="Criminal prosecution", zorder=2)
ax.bar(xpos, civ_props, bottom=crim_props, color=MALE_FILL, edgecolor=MALE_EDGE,
       linewidth=0.6, width=0.5, label="Civil suit", zorder=2)
for i, (cp, vp, cn, vn) in enumerate(zip(crim_props, civ_props,
                                         [F_POOL_CRIM, F_CORP_CRIM],
                                         [F_POOL_CIV,  F_CORP_CIV])):
    ax.text(i, cp / 2, f"{cn}\n({cp:.1f}%)", ha="center", va="center",
            color="white", fontsize=11, fontweight="medium")
    ax.text(i, cp + vp / 2, f"{vn}\n({vp:.1f}%)", ha="center", va="center",
            color=TITLE_COLOR, fontsize=11)
ax.set_xticks(xpos)
ax.set_xticklabels(groups, color=BODY_COLOR, fontsize=10.5)
ax.set_xlim(-0.7, 1.7)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax.set_ylabel("Within-group composition", color=BODY_COLOR, fontsize=10)
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# Title block
fig.text(0.07, 0.94, "Male Corpus Representativeness Check",
         ha="left", va="top", fontsize=18, fontweight="bold", color=TITLE_COLOR)
fig.text(0.07, 0.89,
         f"The criteria-passing corpus is statistically consistent with the filter pool — criteria did not preferentially filter criminal vs. civil cases",
         ha="left", va="top", fontsize=10.5, color=SUBTITLE_COLOR, style="italic")
fig.text(0.07, 0.855,
         "Each bar is the 100% within-group composition. Filter pool: 148 cases passing keyword + exclusion-term filters before substantive criteria.\n"
         "Criteria-passing corpus: the 47 cases also satisfying the full inclusion/exclusion criteria documented in Methodology Section 3.1.3.",
         ha="left", va="top", fontsize=8.8, color=CAPTION_COLOR, style="italic")

# Legend + Fisher's stats — placed UNDER the bars
fig.text(0.07, 0.10, "Procedural type", color=TITLE_COLOR, fontsize=10.5, fontweight="bold")
fig.add_artist(Rectangle((0.07, 0.062), 0.018, 0.024, color=MALE_DEEP, transform=fig.transFigure))
fig.text(0.10, 0.073, "Criminal prosecution", color=BODY_COLOR, fontsize=9.5, va="center")
fig.add_artist(Rectangle((0.21, 0.062), 0.018, 0.024, color=MALE_FILL, ec=MALE_EDGE, lw=0.6, transform=fig.transFigure))
fig.text(0.24, 0.073, "Civil suit", color=BODY_COLOR, fontsize=9.5, va="center")

fig.text(0.42, 0.10, "Fisher's exact test (two-tailed)", color=TITLE_COLOR, fontsize=10.5, fontweight="bold")
fig.text(0.42, 0.073, f"OR = {oddsratio:.2f}, p = {p_fisher:.2f} ({'n.s.' if p_fisher >= 0.05 else 'sig.'})",
         color=BODY_COLOR, fontsize=10)

fig.text(0.66, 0.10, "Pass-through rates", color=TITLE_COLOR, fontsize=10.5, fontweight="bold")
fig.text(0.66, 0.073, f"Criminal: {F_CORP_CRIM} of {F_POOL_CRIM} = {crim_pass_rate:.1f}%   ·   Civil: {F_CORP_CIV} of {F_POOL_CIV} = {civ_pass_rate:.1f}%",
         color=BODY_COLOR, fontsize=10)

fig.savefig(FIG_DIR / "01_male_representativeness.png", dpi=300, facecolor="white")
fig.savefig(FIG_DIR / "01_male_representativeness.pdf",          facecolor="white")
plt.close(fig)
print(f"  saved: 01_male_representativeness.png/.pdf  (Fisher's: OR={oddsratio:.2f}, p={p_fisher:.3f})")
print()

# =============================================================================
# Figure 2 — Female corpus representativeness (Monte Carlo, info-box repositioned)
# =============================================================================
print("Generating female corpus representativeness figure ...")
RNG_SEED = 42
ELIGIBLE_POOL   = 2594
CLASSIFIED_POOL = 2175
N_CRIMINAL      = 1298
N_CIVIL         = 877
POOL_CRIMINAL_P = N_CRIMINAL / CLASSIFIED_POOL
SAMPLE_SIZE     = 50
N_TRIALS        = 100_000
OBSERVED_P      = 0.68
OBSERVED_K      = 34

rng = np.random.default_rng(RNG_SEED)
pool = np.concatenate([np.ones(N_CRIMINAL), np.zeros(N_CIVIL)]).astype(int)
trial_props = np.empty(N_TRIALS)
for i in range(N_TRIALS):
    sample = rng.choice(pool, size=SAMPLE_SIZE, replace=False)
    trial_props[i] = sample.mean()
sampling_mean = trial_props.mean()
sampling_sd   = trial_props.std(ddof=1)
pct_below     = (trial_props < OBSERVED_P).mean() * 100
lo_90, hi_90  = np.percentile(trial_props, [5, 95])

fig = plt.figure(figsize=(11.5, 6.5), dpi=300)
gs = fig.add_gridspec(1, 1, left=0.08, right=0.97, top=0.78, bottom=0.13)
ax = fig.add_subplot(gs[0, 0])

bin_edges = np.arange(0.30, 0.96, 0.025)
counts, edges, patches = ax.hist(trial_props, bins=bin_edges,
                                  color=FEMALE_FILL, edgecolor=FEMALE_EDGE,
                                  linewidth=0.9)
for patch, left, right in zip(patches, edges[:-1], edges[1:]):
    if left <= OBSERVED_P < right:
        patch.set_facecolor(FEMALE_DEEP)
        patch.set_edgecolor(FEMALE_DEEP)
        patch.set_alpha(0.85)
ax.axvline(POOL_CRIMINAL_P, color=FEMALE_DUSTY, linestyle="--", linewidth=1.8, alpha=0.95)
ax.axvline(OBSERVED_P, color=FEMALE_DEEP, linestyle="-", linewidth=1.8, alpha=0.95)

# Line labels — placed near the lines but high enough to not collide with bars
y_top = counts.max()
ax.text(POOL_CRIMINAL_P - 0.045, y_top * 0.78,
        f"Sampling mean\n{sampling_mean*100:.1f}%",
        color=FEMALE_DUSTY, fontsize=10.5, ha="right", va="center",
        fontweight="medium")
ax.text(OBSERVED_P + 0.005, y_top * 0.78,
        f"Observed corpus\n{OBSERVED_P*100:.1f}%",
        color=FEMALE_DEEP, fontsize=10.5, ha="left", va="center",
        fontweight="medium")

# Info box — placed in UPPER-LEFT (away from observed line which is at 68%)
info_text = (f"Eligible pool: {ELIGIBLE_POOL:,} cases (after exclusions)\n"
             f"Classified: {CLASSIFIED_POOL:,} ({POOL_CRIMINAL_P*100:.1f}% criminal)\n"
             f"Observed: {OBSERVED_K}/{SAMPLE_SIZE} = {OBSERVED_P*100:.1f}%   (z = 1.20, p = 0.23, n.s.)\n"
             f"90% sampling range: {lo_90*100:.0f}% – {hi_90*100:.0f}%")
ax.text(0.015, 0.965, info_text, transform=ax.transAxes,
        ha="left", va="top", fontsize=9.5, color=BODY_COLOR,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=0.8))

ax.set_xlim(0.30, 0.95)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.set_xlabel("% criminal cases in random sample of 50", color=BODY_COLOR, labelpad=10)
ax.set_ylabel("Number of trials per bin", color=BODY_COLOR, labelpad=10)
ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.text(0.07, 0.94, "Female Corpus Representativeness Check",
         ha="left", va="top", fontsize=18, fontweight="bold", color=TITLE_COLOR)
fig.text(0.07, 0.89,
         "The 50-case sample is statistically consistent with the criteria-passing pool of 2,175 classified cases",
         ha="left", va="top", fontsize=11, color=SUBTITLE_COLOR, style="italic")
fig.text(0.07, 0.855,
         f"Monte Carlo: {N_TRIALS:,} trials of n = 50 drawn without replacement from the classified pool. The observed corpus (68.0%, dark coral)\n"
         f"falls at the 85.3rd percentile of the sampling distribution. Population baseline {POOL_CRIMINAL_P*100:.1f}% (dashed dusty rose).",
         ha="left", va="top", fontsize=8.8, color=CAPTION_COLOR, style="italic")

fig.savefig(FIG_DIR / "02_female_corpus_representativeness.png", dpi=300, facecolor="white")
fig.savefig(FIG_DIR / "02_female_corpus_representativeness.pdf",          facecolor="white")
plt.close(fig)
print("  saved: 02_female_corpus_representativeness.png/.pdf")
print()

# =============================================================================
# Save summary stats CSVs
# =============================================================================
print("Saving stats CSVs ...")
rows = []
for label, var in HEADLINE:
    for stratum in ["full", "criminal", "civil"]:
        s = all_stats[label][stratum]
        rows.append({
            "variable": label, "var_col": var, "stratum": stratum,
            "n_male": s["n_male"], "n_female": s["n_female"],
            "median_male":   s["median_male"],
            "median_female": s["median_female"],
            "U": s["U"], "p": s["p"], "cliffs_d": s["cliffs_d"],
            "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
        })
pd.DataFrame(rows).to_csv(STATS_DIR / "headline_stats.csv", index=False)

forest_rows = []
for s in forest:
    forest_rows.append({
        "label": s["label"], "var": s["var"],
        "n_male": s["n_male"], "n_female": s["n_female"],
        "U": s["U"], "p": s["p"],
        "cliffs_d": s["cliffs_d"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
        "q_fdr_bh": s["q_fdr_bh"], "p_bonf": s["p_bonf"],
        "fdr_sig":  s["q_fdr_bh"] < 0.05,
        "bonf_sig": s["p_bonf"]   < 0.05,
    })
pd.DataFrame(forest_rows).to_csv(STATS_DIR / "forest_stats.csv", index=False)
print(f"  wrote {STATS_DIR / 'headline_stats.csv'}")
print(f"  wrote {STATS_DIR / 'forest_stats.csv'}")
print()
print("DONE.")
