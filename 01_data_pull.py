"""
courtlistener_fetcher.py
========================
Fetch official court documents for each case in the corpus from
CourtListener's free REST API (https://www.courtlistener.com/api/rest/v4/).

Strategy per case:
    1.  Build a search query from the parsed plaintiff/defendant/year/citation.
    2.  Hit /search/?type=o for opinions matching the caption.
    3.  Score candidates and pick the best match (caption similarity + year +
        court/jurisdiction hints extracted from the LEXIS citation).
    4.  If a match is found, pull the opinion cluster + opinion text/HTML/PDF
        and (when present) the docket with its RECAP document entries.
    5.  Save everything to <out_dir>/<case_id>/ and return a metadata dict
        with links and counts.

State-court cases (NH, TX, VA, AZ, OR, Minn., Wis., Cal. Super., …) are only
partially covered by CourtListener, and federal PACER dockets often cost money
even when the docket metadata is free. If no match is found the function still
returns a valid row (with `cl_status = "not_found"`) so the main CSV stays rectangular.

A CourtListener API token is optional; supply one via the CL_API_TOKEN
environment variable to raise the anonymous rate limit from ~5000/day to
~10000/hour. Code below falls back to unauthenticated access.

Usage (stand-alone):
    python courtlistener_fetcher.py <path/to/case.docx> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import requests

API_BASE = "https://www.courtlistener.com/api/rest/v4"
HEADERS: dict = {
    "User-Agent": "legalbriefs-research-pipeline/0.1",
    "Accept": "application/json",
}

# Hint CourtListener court_id values for the jurisdictions visible in the
# user's corpus. Value can be:
#   - None: no usable jurisdiction hint (federal district, regional reporters)
#   - str: a single exact court_id slug
#   - tuple[str, ...]: multiple acceptable slugs. Prefix-match a whole family
#     by prefixing an entry with "prefix:" (e.g. "prefix:txctapp" matches
#     txctapp1..txctapp14 — the 14 Texas districts).
#
# State appellate courts often have a generic "texapp" slug *and* per-district
# slugs (txctapp1..txctapp14) in CourtListener; we accept either so we don't
# reject a correct district-specific match just because we can't parse the
# district from the caption.
CourtHint = None  # type alias: None | str | tuple[str, ...]
LEXIS_ABBREV_TO_CL_COURT = {
    "U.S. Dist.": None,        # federal district — too many sub-courts; leave open
    "F. Supp. 3d": None,
    "F. Supp. 2d": None,
    "Fed. Appx.": None,
    "Pa. Super.": ("pasuper", "pasuperct"),
    "Tex. App.": ("texapp", "prefix:txctapp"),
    "Cal. Super.": None,
    "Cal. App.": ("calctapp", "prefix:calctapp"),
    "Tenn. App.": ("tennctapp",),
    "Del.": ("del",),
    "N.H.": ("nh",),
    "Mich. App.": ("michctapp",),
    "Minn. App.": ("minnctapp",),
    "Ariz. App.": ("arizctapp",),
    "Ore. App.": ("orctapp",),
    "Va. App.": ("vactapp",),
    "Fla. App.": ("fladistctapp", "prefix:fladistctapp"),
    "N.J. Super.": ("njsuperctappdiv",),
    "Ill. App.": ("illappct",),
    "Wis. App.": ("wisctapp",),
    "Okla. Crim. App.": ("oklacrimapp",),
    "Okla. Civ. App.": ("oklacivapp",),
    "N.Y.": ("ny",),
    "Ohio": ("ohio",),
    "WI": ("wis",),
    "So. 3d": None,
}


def _court_hint_matches(court_id: str, hint) -> bool:
    """Return True if ``court_id`` is acceptable under ``hint``.

    ``hint`` is the value stored in ``LEXIS_ABBREV_TO_CL_COURT``: ``None``
    (no constraint), a single slug string, or a tuple. Each tuple entry is
    either an exact slug or a ``"prefix:<stem>"`` marker for family matches.
    """
    if hint is None:
        return True
    if isinstance(hint, str):
        return court_id == hint
    for entry in hint:
        if entry.startswith("prefix:"):
            if court_id.startswith(entry[len("prefix:"):]):
                return True
        elif court_id == entry:
            return True
    return False


@dataclass
class CLResult:
    # not_attempted / not_found / matched_metadata_only / matched / auth_required / error
    cl_status: str = "not_attempted"
    cl_query: str = ""
    cl_cluster_id: Optional[int] = None
    cl_docket_id: Optional[int] = None
    cl_court_id: str = ""
    cl_court_name: str = ""
    cl_case_name: str = ""
    cl_citations: str = ""
    cl_date_filed: str = ""
    cl_absolute_url: str = ""
    cl_opinion_url: str = ""
    cl_docket_url: str = ""
    cl_n_candidates: int = 0
    cl_match_score: float = 0.0
    cl_n_opinions_saved: int = 0
    cl_n_docket_entries: int = 0
    cl_n_recap_pdfs: int = 0
    cl_has_token: bool = False
    cl_error: str = ""
    cl_folder: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Accept either env-var name; COURTLISTENER_TOKEN is the one in Sarnika's
    # shell history, CL_API_TOKEN was the original internal name.
    token = os.environ.get("CL_API_TOKEN") or os.environ.get("COURTLISTENER_TOKEN")
    if token:
        s.headers["Authorization"] = f"Token {token}"
    return s


class CLAuthRequired(Exception):
    """CourtListener returned 401/403 — detail endpoint needs an API token."""


def _get_json(session: requests.Session, url: str, **params) -> dict:
    resp = session.get(url, params=params or None, timeout=20)
    # Respect rate limits
    if resp.status_code == 429:
        time.sleep(2.0)
        resp = session.get(url, params=params or None, timeout=20)
    if resp.status_code in (401, 403):
        raise CLAuthRequired(f"CourtListener {resp.status_code} at {url}")
    resp.raise_for_status()
    return resp.json()


def _download_binary(session: requests.Session, url: str, path: Path) -> bool:
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Candidate matching
# ─────────────────────────────────────────────────────────────────────

def _normalize_caption(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _citation_matches(cand: dict, citation_raw: str) -> bool:
    """True when our LEXIS/reporter citation literally appears in the candidate's
    citations. CourtListener's citation list contains parallel cites, so we
    string-match any token."""
    if not citation_raw:
        return False
    target = re.sub(r"\s+", " ", citation_raw.strip().lower())
    for cite in (cand.get("citation") or []):
        if isinstance(cite, str) and target and target in cite.lower():
            return True
    return False


def _score_candidate(cand: dict, plaintiff: str, defendant: str,
                     year: Optional[int], court_hint: Optional[str],
                     citation_raw: str = "") -> float:
    """Score a candidate 0-100(+).

    Confounder we have to defeat: every state has a ``State v. Smith`` case,
    so identical caseName alone is not enough. We treat jurisdiction (court)
    and citation as authoritative tiebreakers.
    """
    case_name = cand.get("caseName", "") or cand.get("caseNameFull", "") or ""
    case_name_full = cand.get("caseNameFull", "") or ""
    caption_similarity = SequenceMatcher(None,
                            _normalize_caption(f"{plaintiff} v. {defendant}"),
                            _normalize_caption(case_name)).ratio()
    score = caption_similarity * 60

    # Defendant surname + (if given) first-name/initial check against caseNameFull,
    # which includes e.g. "STATE OF NEW JERSEY v. JOHN E. MCINNES".
    def_tokens = [t for t in re.split(r"\W+", defendant) if len(t) >= 3]
    full_lower = case_name_full.lower()
    for t in def_tokens:
        if t.lower() in full_lower:
            score += 4

    # Strong citation-match bonus: if our LEXIS cite literally appears in the
    # candidate's citation list, this is an exact match.
    citation_hit = _citation_matches(cand, citation_raw)
    if citation_hit:
        score += 40

    # Year discipline
    date_filed = cand.get("dateFiled", "") or ""
    if year and date_filed[:4].isdigit():
        df_year = int(date_filed[:4])
        if df_year == year:
            score += 25
        elif abs(df_year - year) == 1:
            score += 5
        elif abs(df_year - year) > 3:
            score -= 30     # strong penalty — different case
    elif year and not date_filed:
        score -= 10

    # Court jurisdiction discipline
    court_match = True  # unknown == not penalized
    if court_hint:
        if _court_hint_matches(cand.get("court_id", ""), court_hint):
            score += 20
        else:
            # the hint was explicit and the candidate is not it — penalize
            score -= 25
            court_match = False

    # Hard cap when jurisdiction hint explicitly disagrees AND we don't have a
    # citation match. Identical captions across different states are common;
    # without a citation hit we should not claim a match just because the
    # caseName string is identical.
    if court_hint and not court_match and not citation_hit:
        score = min(score, 45.0)

    return score


def _court_hint_from_citation(citation_raw: str):
    """Return the hint value for the first LEXIS abbreviation found in the
    citation, or None. Return type is ``None | str | tuple[str, ...]`` — see
    ``_court_hint_matches`` for how it's consumed.
    """
    for abbrev, court_id in LEXIS_ABBREV_TO_CL_COURT.items():
        if abbrev and abbrev in citation_raw:
            return court_id
    return None


def search_case(session: requests.Session, plaintiff: str, defendant: str,
                year: Optional[int], citation_raw: str) -> dict:
    """Return the best-matching opinion-cluster candidate and raw list."""
    q_caption = f'"{plaintiff} v. {defendant}"' if plaintiff and defendant \
                else f"{plaintiff} {defendant}"
    params = {"q": q_caption, "type": "o", "order_by": "score desc"}
    url = f"{API_BASE}/search/"
    data = _get_json(session, url, **params)
    candidates = data.get("results", [])
    court_hint = _court_hint_from_citation(citation_raw)
    scored = [
        (_score_candidate(c, plaintiff, defendant, year, court_hint, citation_raw), c)
        for c in candidates
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0] if scored else (0, None)
    return {
        "query": q_caption,
        "n_candidates": len(candidates),
        "best_score": best[0],
        "best": best[1],
        "all_candidates": candidates,
    }


# ─────────────────────────────────────────────────────────────────────
# Fetch full bundle for a matched cluster
# ─────────────────────────────────────────────────────────────────────

def fetch_cluster_bundle(session: requests.Session, cluster_id: int,
                         out_dir: Path, max_pdfs: int = 10) -> dict:
    """Download opinion text + PDFs and (if available) docket entries for a cluster."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_opinions_saved": 0, "n_docket_entries": 0, "n_recap_pdfs": 0}

    cluster = _get_json(session, f"{API_BASE}/clusters/{cluster_id}/")
    (out_dir / "cluster.json").write_text(json.dumps(cluster, indent=2))

    # Opinions (may be multiple: lead, concurrence, dissent, per curiam)
    opinion_urls = cluster.get("sub_opinions", [])
    for op_url in opinion_urls:
        op_id = op_url.rstrip("/").split("/")[-1]
        try:
            op = _get_json(session, op_url)
        except Exception:
            continue
        (out_dir / f"opinion_{op_id}.json").write_text(json.dumps(op, indent=2))
        plain = op.get("plain_text") or op.get("html_lawbox") or op.get("html") or ""
        if plain:
            (out_dir / f"opinion_{op_id}.txt").write_text(plain)
        dl = op.get("download_url") or ""
        if dl and dl.lower().endswith(".pdf"):
            _download_binary(session, dl, out_dir / f"opinion_{op_id}.pdf")
        summary["n_opinions_saved"] += 1

    # Docket + RECAP docs
    docket_url = cluster.get("docket")
    docket_id = None
    if docket_url:
        try:
            docket = _get_json(session, docket_url)
        except Exception:
            docket = None
        if docket:
            docket_id = docket.get("id")
            (out_dir / "docket.json").write_text(json.dumps(docket, indent=2))
            summary["docket_id"] = docket_id
            # RECAP documents for this docket (federal cases only)
            try:
                rd = _get_json(
                    session,
                    f"{API_BASE}/recap-documents/",
                    docket_entry__docket=docket_id,
                    page_size=50,
                )
            except Exception:
                rd = None
            if rd and rd.get("results"):
                (out_dir / "recap_documents.json").write_text(json.dumps(rd, indent=2))
                summary["n_docket_entries"] = len(rd["results"])
                pdf_dir = out_dir / "recap_pdfs"
                pdf_dir.mkdir(exist_ok=True)
                for i, entry in enumerate(rd["results"][:max_pdfs]):
                    fp = entry.get("filepath_local") or entry.get("filepath_ia")
                    if fp and fp.lower().endswith(".pdf"):
                        pdf_url = f"https://storage.courtlistener.com/{fp}" \
                                  if not fp.startswith("http") else fp
                        ok = _download_binary(session, pdf_url,
                                              pdf_dir / f"doc_{i:03d}.pdf")
                        if ok:
                            summary["n_recap_pdfs"] += 1
    return summary


# ─────────────────────────────────────────────────────────────────────
# Top-level: one case → CLResult
# ─────────────────────────────────────────────────────────────────────

def fetch_case(case_id: str, plaintiff: str, defendant: str,
               year: Optional[int], citation_raw: str,
               out_root: Path,
               match_threshold: float = 70.0,
               save_pdfs: bool = True) -> CLResult:
    """Search CourtListener for one case and, if a good match is found, fetch bundle.

    Degradation ladder:
    - No plaintiff/defendant -> not_attempted.
    - No candidate above threshold -> not_found (still writes first 5 raw candidates
      to the case folder for human review).
    - Match found but /clusters/ detail endpoint returns 401/403 (CourtListener
      requires auth for detail endpoints since late 2024) -> matched_metadata_only.
      The search-response metadata (case name, court, date, citations, URL) is
      still populated so downstream rows have useful provenance; a CL_API_TOKEN
      is needed to download opinion text and RECAP PDFs.
    - Match found and bundle download succeeds -> matched.
    """
    res = CLResult()
    res.cl_has_token = bool(os.environ.get("CL_API_TOKEN"))
    if not plaintiff or not defendant:
        res.cl_status = "not_attempted"
        res.cl_error = "missing plaintiff or defendant"
        return res

    case_folder = out_root / case_id
    res.cl_folder = str(case_folder)

    try:
        sess = _session()
        search_result = search_case(sess, plaintiff, defendant, year, citation_raw)
        res.cl_query = search_result["query"]
        res.cl_n_candidates = search_result["n_candidates"]
        best = search_result["best"]
        res.cl_match_score = round(search_result["best_score"], 2)
        if not best or search_result["best_score"] < match_threshold:
            res.cl_status = "not_found"
            # still save the search response for later inspection
            case_folder.mkdir(parents=True, exist_ok=True)
            (case_folder / "search_response.json").write_text(
                json.dumps(search_result["all_candidates"][:5], indent=2)
            )
            return res

        # Fill metadata from the best hit (everything here is from /search/,
        # which does not require auth — safe to populate even when the cluster
        # detail call later 401s).
        res.cl_cluster_id = best.get("cluster_id")
        res.cl_docket_id = best.get("docket_id")
        res.cl_court_id = best.get("court_id", "")
        res.cl_court_name = best.get("court", "")
        res.cl_case_name = best.get("caseName", "")
        res.cl_citations = "; ".join(best.get("citation", []) or [])
        res.cl_date_filed = best.get("dateFiled", "") or ""
        res.cl_absolute_url = "https://www.courtlistener.com" + best.get("absolute_url", "")

        # Persist the raw search hit so we always have a paper trail for the match.
        case_folder.mkdir(parents=True, exist_ok=True)
        (case_folder / "search_match.json").write_text(json.dumps(best, indent=2))

        if save_pdfs and res.cl_cluster_id:
            try:
                bundle = fetch_cluster_bundle(sess, res.cl_cluster_id, case_folder)
                res.cl_n_opinions_saved = bundle.get("n_opinions_saved", 0)
                res.cl_n_docket_entries = bundle.get("n_docket_entries", 0)
                res.cl_n_recap_pdfs = bundle.get("n_recap_pdfs", 0)
                res.cl_opinion_url = str(case_folder / "cluster.json")
                if "docket_id" in bundle:
                    res.cl_docket_url = str(case_folder / "docket.json")
                res.cl_status = "matched"
            except CLAuthRequired as e:
                res.cl_status = "matched_metadata_only"
                res.cl_error = (
                    "CourtListener requires an API token to download opinion "
                    "text / RECAP PDFs. Set CL_API_TOKEN to unlock bundle fetch. "
                    f"Detail: {e}"
                )
        else:
            res.cl_status = "matched_metadata_only"
        return res
    except CLAuthRequired as e:
        # Search itself shouldn't require auth, but if CL tightens further:
        res.cl_status = "auth_required"
        res.cl_error = str(e)
        return res
    except requests.HTTPError as e:
        res.cl_status = "error"
        res.cl_error = f"HTTP {e.response.status_code}"
        return res
    except Exception as e:
        res.cl_status = "error"
        res.cl_error = f"{type(e).__name__}: {e}"
        return res


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch CourtListener data for a single case file")
    ap.add_argument("path", type=Path, help="Case .docx file")
    ap.add_argument("out", type=Path, nargs="?", default=Path("./analysis_output/courtlistener"))
    args = ap.parse_args(argv)

    # Lazily import the main pipeline so party extraction logic is shared
    sys.path.insert(0, str(Path(__file__).parent))
    from legalbriefs_pipeline import (
        extract_case_id_from_filename, caption_from_filename, parse_caption,
        extract_citation_from_filename,
    )
    caption = caption_from_filename(args.path)
    parsed = parse_caption(caption)
    cite = extract_citation_from_filename(args.path)
    case_id = extract_case_id_from_filename(args.path)

    result = fetch_case(
        case_id=case_id,
        plaintiff=parsed["plaintiff"],
        defendant=parsed["defendant"],
        year=cite.get("year"),
        citation_raw=cite.get("citation_raw", ""),
        out_root=args.out,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _main()
