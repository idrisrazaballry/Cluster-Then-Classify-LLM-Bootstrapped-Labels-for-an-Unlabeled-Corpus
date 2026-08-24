"""Phase 0 -- load, clean, quarantine the labels.

The cleaning here is not boilerplate. Each rule below was added because it
changed the clustering result on this corpus; see the docstring on clean_field.
"""
import html
import re

import pandas as pd

import config as C

SOURCES = (
    r"AP|Reuters|AFP|Forbes\.com|USATODAY\.com|SPACE\.com|NewsFactor|Ziff Davis|"
    r"PC World|InfoWorld|TechWeb|CNET|washingtonpost\.com|The Motley Fool|"
    r"Investor's Business Daily|Canadian Press|MacCentral|CBS MarketWatch\.com|AP Online"
)

_RE_NUM_ENT = re.compile(r"(?<!&)#(\d+);")
_RE_NAMED_ENT = re.compile(r"(?<!&)\b(quot|amp|lt|gt|nbsp|apos);")
_RE_TAG = re.compile(r"<[^>]{0,400}>")
_RE_TICKER = re.compile(r"\b[A-Z]{1,5}\.[A-Z]{1,3}\b")
_RE_SRC_PREFIX = re.compile(rf"^\s*({SOURCES})\s*[-\u2013:]\s*", re.I)
_RE_DATELINE = re.compile(rf"^\s*[A-Z][A-Za-z .]{{2,25}}\s*\(({SOURCES})\)\s*[-\u2013]\s*")
_RE_SRC_PAREN = re.compile(rf"\s*\(({SOURCES})\)", re.I)
_RE_SPACE_PUNCT = re.compile(r"\s+([.,;:!?])")
_RE_WS = re.compile(r"\s+")


def clean_field(s: str) -> str:
    """Clean ONE field. Title and Description must be cleaned separately.

    Cleaning the joined string does not work: the source-prefix patterns anchor
    to the start of a field, and once joined the Description no longer starts
    the string.

    Rules, in the order they must run:
      1. Literal backslash escapes surviving the original export.
      2. Entities missing their leading '&' ("#39;", "quot;"). Repair before
         unescaping or html.unescape passes silently over them.
      3. Markup exposed by unescaping. Business stories embed Reuters quote
         links (<A HREF="...FullQuote.aspx?ticker=SPLS.O">); left in, the
         Business cluster forms around "fullquote"/"aspx"/"reuters" rather than
         around business content.
      4. Source markers in all three positions: "AP - x", "CITY (Reuters) - x",
         and "x (AP)". Left in, clusters form around the news agency.
    """
    s = s.replace("\\\\", " ").replace("\\", " ")

    s = _RE_NUM_ENT.sub(r"&#\1;", s)
    s = _RE_NAMED_ENT.sub(r"&\1;", s)
    s = html.unescape(html.unescape(s))          # double-encoded in places

    s = _RE_TAG.sub(" ", s)
    s = _RE_TICKER.sub(" ", s)

    s = _RE_SRC_PREFIX.sub("", s)
    s = _RE_DATELINE.sub("", s)
    s = _RE_SRC_PAREN.sub(" ", s)

    s = _RE_SPACE_PUNCT.sub(r"\1", s)
    return _RE_WS.sub(" ", s).strip()


ARTIFACT_CHECKS = {
    "entities": r"#\d+;|&[a-z]+;",
    "backslashes": r"\\",
    "html tags": r"<[^>]+>",
    "(AP)/(Reuters)": r"\((?:AP|Reuters|AFP)\)",
    "reuters boiler": r"(?i)fullquote|aspx",
}


def audit(texts: pd.Series) -> dict:
    """Residual-artifact counts. Every entry here caught a real bug."""
    return {k: int(texts.str.contains(v, regex=True).sum())
            for k, v in ARTIFACT_CHECKS.items()}


def load_and_clean(csv_path=None, quarantine=True):
    """Returns (work, y_true). `work` holds text only -- no labels.

    y_true is written to disk and returned for the harness, but Phases 1-4 must
    never receive it. Only Phase 5 reads it back.
    """
    raw = pd.read_csv(csv_path or C.RAW_CSV)

    title = raw[C.TEXT_COLS[0]].map(clean_field)
    desc = raw[C.TEXT_COLS[1]].map(clean_field)
    text = (title + ". " + desc).str.replace(r"\s+", " ", regex=True).str.strip()

    keep = (text.str.split().str.len() >= C.MIN_WORDS) & (~text.duplicated())
    dropped = (~keep).sum()

    work = pd.DataFrame({"text": text[keep].values})
    y_true = raw.loc[keep, C.LABEL_COL].values

    if quarantine:
        pd.DataFrame({C.LABEL_COL: y_true}).to_csv(C.P_QUARANTINE, index=False)
        work.to_csv(C.P_CLEAN, index=False)

    print(f"[phase0] {len(raw)} rows in, {dropped} dropped, {len(work)} remain")
    for k, v in audit(work["text"]).items():
        print(f"[phase0]   residual {k:16}: {v}")

    return work, y_true
