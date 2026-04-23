"""
Near-Duplicate Web Page Dataset Builder  v3.0
==============================================
Research Project: Boilerplate-Aware DOM Extraction Pipeline for LSH Stabilization
Authors: Aastha Malhotra, Raashi Sharma, Sonali Verma

WHAT'S NEW IN v3.0
-------------------
- CURATED URL MODE: BBC and Al Jazeera articles are loaded from a hardcoded
  list of manually verified URLs instead of RSS feed discovery.
  This is the correct approach when you have hand-picked URLs — no more
  dependency on RSS feeds or URL pattern matching for those two domains.
- WIKIPEDIA TOP-UP: After loading the curated BBC + AJE articles, the script
  automatically calculates how many Wikipedia articles are needed to reach
  the --n target and fetches exactly that many via the Wikipedia random API.
  Default target is 150 total articles (26 BBC + 20 AJE + 104 Wikipedia).
- --n now controls the TOTAL article target across all three sources.
- --wiki-only flag: skip BBC/AJE and only scrape Wikipedia (useful for testing).
- All variant generation and output format unchanged from v2.

INSTALLATION
------------
    pip install trafilatura requests beautifulsoup4 lxml tqdm

USAGE
-----
    # Standard run: 26 BBC + 20 AJE + 104 Wikipedia = 150 articles
    python build_dataset.py

    # Custom total (e.g. 200 = 26 BBC + 20 AJE + 154 Wikipedia)
    python build_dataset.py --n 200

    # Only Wikipedia articles (useful for quick testing)
    python build_dataset.py --wiki-only --n 50

    # Custom output directory
    python build_dataset.py --out ./my_dataset --seed 99

OUTPUT STRUCTURE
----------------
dataset/
  raw_html/                  original + all variant HTML files
  extracted_text/            trafilatura-extracted plain text files
  articles.jsonl             article metadata (no raw HTML)
  pairs.jsonl                pair records with file paths only
  pairs_with_content.jsonl   pairs WITH extracted_text inline
                             --> FEED THIS INTO MinHash / SimHash scripts
  stats.json                 summary statistics

pairs_with_content.jsonl SCHEMA (one JSON object per line)
----------------------------------------------------------
{
  "pair_id":        "bbc_001_boilerplate_only_diff",
  "article_id":     "bbc_001",
  "domain":         "bbc",
  "variant_type":   "boilerplate_only_diff",
  "label":          1,
  "label_reason":   "...",
  "page_a": {
      "url":            "https://...",
      "extracted_text": "Full article body...",
      "html_path":      "raw_html/bbc_001_original.html",
      "text_path":      "extracted_text/bbc_001_original.txt"
  },
  "page_b": { ... same fields ... }
}

VARIANT TYPES AND GROUND-TRUTH LABELS
--------------------------------------
  timestamp_swap        -> label 1  (true near-duplicate)
  ad_block_injection    -> label 1  (true near-duplicate)
  url_difference_only   -> label 1  (true near-duplicate)
  minor_content_edit    -> label 1  (true near-duplicate)
  boilerplate_only_diff -> label 1  (true near-duplicate)  *** KEY experiment ***
  full_content_swap     -> label 0  (true negative)
"""

import re
import json
import time
import random
import logging
import argparse
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
MISSING = []
try:
    import requests
except ImportError:
    MISSING.append("requests")
try:
    import trafilatura
    from trafilatura.settings import use_config
except ImportError:
    MISSING.append("trafilatura")
try:
    from bs4 import BeautifulSoup
except ImportError:
    MISSING.append("beautifulsoup4")
try:
    from tqdm import tqdm
except ImportError:
    MISSING.append("tqdm")

if MISSING:
    raise ImportError(
        f"Missing dependencies: {', '.join(MISSING)}\n"
        f"Run: pip install {' '.join(MISSING)}"
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ===========================================================================
# CURATED URL LISTS
# Hand-verified article URLs. These bypass RSS/feed discovery entirely.
# Add or remove URLs here as needed.
# ===========================================================================

CURATED_BBC_URLS = [
    "https://www.bbc.com/news/articles/cvg4jnn131qo",
    "https://www.bbc.com/news/articles/c0q9v1p2dd2o",
    "https://www.bbc.com/news/articles/c239500dx8ro",
    "https://www.bbc.com/news/articles/cge0grppe3po",
    "https://www.bbc.com/news/articles/c248m3z49j1o",
    "https://www.bbc.com/news/articles/cx2631x6nelo",
    "https://www.bbc.com/news/articles/c4g833128vlo",
    "https://www.bbc.com/news/articles/cpd557w952xo",
    "https://www.bbc.com/news/articles/cly6pj37wxgo",
    "https://www.bbc.com/news/articles/ce8lp2ny3rro",
    "https://www.bbc.com/news/articles/c87wqzzpzy9o",
    "https://www.bbc.com/news/articles/c07027e59d7o",
    "https://www.bbc.com/news/articles/c8ej818gz8xo",
    "https://www.bbc.com/news/articles/czd72dmpjpno",
    "https://www.bbc.com/news/articles/c4gxx9034wwo",
    "https://www.bbc.com/news/articles/c624vpxl5rqo",
    "https://www.bbc.com/news/articles/cx2r1qw8x45o",
    "https://www.bbc.com/news/articles/c78rzrymryyo",
    "https://www.bbc.com/news/articles/c5yjz819vmgo",
    "https://www.bbc.com/news/articles/c0rxdqeyjl8o",
    "https://www.bbc.com/news/articles/cy91r7ww3weo",
    "https://www.bbc.com/news/articles/cx24118yrpwo",
    "https://www.bbc.com/news/articles/c20dy511v3lo",
    "https://www.bbc.com/news/articles/cp86r29l45go",
    "https://www.bbc.com/news/articles/ce8jl1271pjo",
    "https://www.bbc.com/news/articles/cwyj8d49wzxo",
]

CURATED_AJE_URLS = [
    "https://www.aljazeera.com/features/2026/4/20/to-stay-or-to-go-no-good-options-for-lebanon-displaced",
    "https://www.aljazeera.com/news/2026/4/19/trump-says-us-seized-iran-flagged-ship-trying-to-get-past-hormuz-blockade",
    "https://www.aljazeera.com/economy/2026/4/20/oil-prices-surge-amid-mixed-signals-on-us-iran-peace-talks",
    "https://www.aljazeera.com/news/2026/4/20/whats-behind-the-us-armys-decision-to-raise-enlistment-age-to-42",
    "https://www.aljazeera.com/opinions/2026/4/8/did-america-lose-yet-another-war",
    "https://www.aljazeera.com/news/2026/4/20/pakistan-ready-for-multi-day-us-iran-talks-but-tehran-unsure-about-joining",
    "https://www.aljazeera.com/sports/2026/4/20/wembanyama-makes-history-as-spurs-defeat-blazers-in-game-1",
    "https://www.aljazeera.com/sports/2026/4/20/kane-scores-as-bayern-munich-claim-bundesliga-title-with-stuttgart-victory",
    "https://www.aljazeera.com/news/2026/4/20/as-barbed-wire-blocks-kids-from-class-palestinians-stage-freedom-school",
    "https://www.aljazeera.com/news/2026/4/19/outrage-after-photo-shows-israeli-soldier-smashing-jesus-statue-in-lebanon",
    "https://www.aljazeera.com/gallery/2026/4/19/displaced-lebanese-families-return-south-despite-risks-near-israeli-border",
    "https://www.aljazeera.com/news/2026/4/19/canadian-pm-says-close-economic-ties-with-us-have-become-a",
    "https://www.aljazeera.com/news/2026/4/19/eight-children-killed-in-mass-shooting-in-louisiana-us-media-reports",
    "https://www.aljazeera.com/news/2026/4/15/four-killed-in-turkiyes-second-school-shooting-in-two-days",
    "https://www.aljazeera.com/news/2026/4/19/drc-government-m23-rebels-commit-to-protect-civilians-aid-deliveries",
    "https://www.aljazeera.com/news/2026/4/19/pope-leo-tells-angola-during-huge-mass-to-build-hope",
    "https://www.aljazeera.com/news/2026/4/18/pope-leo-heads-to-angola-in-landmark-africa-visit-amid-trump-clash",
    "https://www.aljazeera.com/news/2026/4/16/pope-leo-decries-world-ruled-by-tyrants-after-trump-attacks",
    "https://www.aljazeera.com/news/2026/4/19/peru-says-presidential-election-results-due-by-mid-may-after-delayed-count",
    "https://www.aljazeera.com/news/2026/4/17/fifteen-south-american-people-deported-from-the-us-arrive-in-dr-congo",
]

# Deduplicate while preserving order (one AJE URL appeared twice in the input)
CURATED_BBC_URLS = list(dict.fromkeys(CURATED_BBC_URLS))
CURATED_AJE_URLS = list(dict.fromkeys(CURATED_AJE_URLS))

# ===========================================================================
# HTTP / extraction utilities
# ===========================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; "
        "Academic research - SNU; contact: am565@snu.edu.in)"
    )
}
REQUEST_DELAY = 1.5   # seconds between requests — be polite
MAX_RETRIES   = 3
FETCH_TIMEOUT = 20


def fetch_url(url: str) -> str | None:
    """Fetch a URL with retries. Returns raw HTML or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            log.warning(f"HTTP {resp.status_code}: {url[:80]} (attempt {attempt})")
        except requests.exceptions.ConnectionError:
            log.warning(f"Connection refused: {url[:80]} (attempt {attempt})")
        except requests.exceptions.Timeout:
            log.warning(f"Timeout: {url[:80]} (attempt {attempt})")
        except Exception as e:
            log.warning(f"Error fetching {url[:80]} (attempt {attempt}): {e}")
    return None


def extract_text(html: str) -> str | None:
    """Extract main article text using trafilatura."""
    if not html:
        return None
    try:
        cfg = use_config()
        cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        return trafilatura.extract(
            html,
            config=cfg,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
    except Exception as e:
        log.debug(f"trafilatura error: {e}")
        return None


def is_english(text: str, min_length: int = 200) -> bool:
    """
    Lightweight English detector — no external library needed.
    Checks ASCII character ratio and common English function word hits.
    Rejects CJK, Arabic, and Devanagari-heavy pages.
    """
    if not text or len(text.strip()) < min_length:
        return False
    sample = text[:1000]
    ascii_ratio = sum(1 for c in sample if ord(c) < 128) / len(sample)
    if ascii_ratio < 0.85:
        return False
    sample_lower = sample.lower()
    en_markers = [" the ", " and ", " of ", " in ", " to ", " a ", " is ", " that ", " for "]
    return sum(1 for m in en_markers if m in sample_lower) >= 4

# ===========================================================================
# Wikipedia URL discovery
# ===========================================================================

def discover_wikipedia_urls(n: int) -> list[str]:
    """
    Fetch n random English Wikipedia article URLs via the MediaWiki API.
    Makes multiple API calls (each returns up to 50 titles) until n is reached.
    """
    urls: list[str] = []
    api = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&list=random&rnnamespace=0&rnlimit=50&format=json"
    )
    max_calls = (n // 40) + 4
    for _ in range(max_calls):
        if len(urls) >= n:
            break
        data = fetch_url(api)
        if not data:
            log.warning("Wikipedia API call failed — check network access")
            break
        try:
            obj = json.loads(data)
            for page in obj["query"]["random"]:
                title = urllib.parse.quote(page["title"].replace(" ", "_"))
                urls.append(f"https://en.wikipedia.org/wiki/{title}")
        except Exception as e:
            log.warning(f"Wikipedia API parse error: {e}")
            break
    return list(dict.fromkeys(urls))[:n]

# ===========================================================================
# Article scraping
# ===========================================================================

def scrape_from_url_list(
    urls: list[str],
    domain: str,
    display_name: str,
    start_idx: int = 1,
) -> list[dict]:
    """
    Fetch and extract text from a fixed list of URLs.
    Skips any URL that returns no content or non-English text.
    Returns a list of article dicts.
    """
    log.info(f"\n{'─'*60}")
    log.info(f"Source: {display_name}  ({len(urls)} curated URLs)")
    log.info(f"{'─'*60}")

    articles = []
    idx = start_idx
    failed = 0

    for url in tqdm(urls, desc=f"Fetching {domain}", unit="page"):
        html = fetch_url(url)
        if not html:
            log.warning(f"  FAIL (no response): {url}")
            failed += 1
            continue

        text = extract_text(html)
        if not text or len(text) < 200:
            log.warning(f"  FAIL (text too short, {len(text or '')} chars): {url[:70]}")
            failed += 1
            continue

        if not is_english(text):
            log.warning(f"  SKIP (non-English): {url[:70]}")
            failed += 1
            continue

        article_id = f"{domain}_{idx:03d}"
        articles.append({
            "article_id":     article_id,
            "domain":         domain,
            "domain_display": display_name,
            "url":            url,
            "raw_html":       html,
            "extracted_text": text,
            "text_length":    len(text),
            "html_length":    len(html),
            "scraped_at":     datetime.utcnow().isoformat() + "Z",
            "source":         "curated",
        })
        idx += 1
        log.info(f"  OK {article_id}  ({len(text):,} chars)  {url[:65]}")

    log.info(f"  Collected: {len(articles)}  |  Failed/skipped: {failed}")
    return articles


def scrape_wikipedia(n: int, start_idx: int = 1) -> list[dict]:
    """
    Scrape n articles from English Wikipedia using the random-page API.
    Over-requests candidate URLs (3x) to account for short/non-English pages.
    """
    if n <= 0:
        return []

    log.info(f"\n{'─'*60}")
    log.info(f"Source: Wikipedia (English)  (target: {n} articles)")
    log.info(f"{'─'*60}")

    candidate_urls = discover_wikipedia_urls(n * 3)
    log.info(f"  Discovered {len(candidate_urls)} candidate Wikipedia URLs")

    if not candidate_urls:
        log.error(
            "  No Wikipedia URLs returned. Check network access.\n"
            "  Wikipedia API endpoint: https://en.wikipedia.org/w/api.php"
        )
        return []

    articles = []
    idx = start_idx
    failed = 0

    for url in tqdm(candidate_urls, desc="Fetching Wikipedia", unit="page"):
        if len(articles) >= n:
            break

        html = fetch_url(url)
        if not html:
            failed += 1
            continue

        text = extract_text(html)
        if not text or len(text) < 200:
            log.debug(f"  Skip (short/no text): {url[:70]}")
            failed += 1
            continue

        if not is_english(text):
            log.debug(f"  Skip (non-English): {url[:70]}")
            failed += 1
            continue

        article_id = f"wikipedia_{idx:03d}"
        articles.append({
            "article_id":     article_id,
            "domain":         "wikipedia",
            "domain_display": "Wikipedia (English)",
            "url":            url,
            "raw_html":       html,
            "extracted_text": text,
            "text_length":    len(text),
            "html_length":    len(html),
            "scraped_at":     datetime.utcnow().isoformat() + "Z",
            "source":         "wikipedia_random_api",
        })
        idx += 1
        log.info(f"  OK wikipedia_{idx-1:03d}  ({len(text):,} chars)  {url[:60]}")

    log.info(f"  Collected: {len(articles)}  |  Failed/skipped: {failed}")
    return articles

# ===========================================================================
# Variant constants
# ===========================================================================

VARIANT_TYPES = {
    "timestamp_swap": {
        "label": 1,
        "label_reason": (
            "Main content identical; only the publication date/timestamp "
            "string changed. Matches Henzinger's timestamp-only category."
        ),
    },
    "ad_block_injection": {
        "label": 1,
        "label_reason": (
            "Main content identical; a synthetic ad block injected. "
            "Simulates dynamic advertisement slots."
        ),
    },
    "url_difference_only": {
        "label": 1,
        "label_reason": (
            "Main content identical; domain in canonical/og:url tags replaced "
            "with a synthetic mirror domain. Matches Henzinger's parked-domain "
            "/ URL-only category."
        ),
    },
    "minor_content_edit": {
        "label": 1,
        "label_reason": (
            "Main content nearly identical; 5-10 words substituted with "
            "synonyms. Tests the near-duplicate sensitivity threshold."
        ),
    },
    "boilerplate_only_diff": {
        "label": 1,
        "label_reason": (
            "Main content completely identical; nav/header/footer boilerplate "
            "replaced with a different synthetic template. KEY experimental "
            "case: Pipeline A (raw DOM) may fail; Pipeline B (extraction-based) "
            "should correctly detect as near-duplicate."
        ),
    },
    "full_content_swap": {
        "label": 0,
        "label_reason": (
            "A completely different article body placed inside the same site "
            "template. True negative — must NOT be flagged as near-duplicate."
        ),
    },
}

SYNONYM_MAP = {
    "said": ["stated", "noted", "remarked", "indicated"],
    "told": ["informed", "notified", "advised"],
    "big": ["large", "significant", "major", "substantial"],
    "small": ["minor", "limited", "modest", "slight"],
    "new": ["recent", "latest", "fresh", "novel"],
    "show": ["reveal", "demonstrate", "indicate", "suggest"],
    "help": ["assist", "support", "aid", "facilitate"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "make": ["create", "produce", "generate", "form"],
    "start": ["begin", "initiate", "launch", "commence"],
    "end": ["conclude", "finish", "complete", "terminate"],
    "find": ["discover", "identify", "locate", "detect"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "give": ["provide", "offer", "supply", "deliver"],
    "take": ["acquire", "obtain", "receive", "collect"],
    "see": ["observe", "notice", "view", "witness"],
    "know": ["understand", "recognize", "realize", "acknowledge"],
    "think": ["believe", "consider", "feel", "view"],
    "come": ["arrive", "emerge", "appear", "surface"],
    "go": ["move", "proceed", "travel", "advance"],
    "first": ["initial", "primary", "leading", "foremost"],
    "last": ["final", "latest", "most recent", "concluding"],
    "long": ["extended", "lengthy", "prolonged", "sustained"],
    "high": ["elevated", "substantial", "significant", "considerable"],
    "low": ["minimal", "reduced", "limited", "modest"],
    "old": ["former", "previous", "prior", "earlier"],
    "part": ["portion", "section", "segment", "component"],
    "work": ["function", "operate", "perform", "serve"],
    "place": ["location", "site", "area", "region"],
    "people": ["individuals", "persons", "residents", "citizens"],
    "including": ["comprising", "encompassing", "involving", "covering"],
    "after": ["following", "subsequent to", "once", "in the wake of"],
    "before": ["prior to", "ahead of", "preceding", "in advance of"],
    "because": ["since", "as", "given that", "due to the fact that"],
    "however": ["nevertheless", "nonetheless", "yet", "that said"],
    "also": ["additionally", "furthermore", "moreover", "as well"],
    "many": ["numerous", "several", "multiple", "various"],
    "more": ["additional", "further", "greater", "increased"],
    "less": ["fewer", "reduced", "lower", "diminished"],
}

BOILERPLATE_TEMPLATES = [
    """<header id="site-header" role="banner">
  <div class="logo-wrap"><a href="/">MediaGroup International</a></div>
  <nav aria-label="Primary navigation">
    <a href="/news">News</a> <a href="/opinion">Opinion</a>
    <a href="/business">Business</a> <a href="/tech">Tech</a>
  </nav>
  <div class="header-actions">
    <button class="btn-subscribe">Subscribe</button>
    <input type="search" placeholder="Search articles..." aria-label="Site search"/>
  </div>
</header>
<div class="breaking-bar">Breaking: Markets react to latest economic data</div>
<footer id="site-footer">
  <div class="footer-grid">
    <div class="footer-col"><h4>Company</h4><ul><li>About</li><li>Careers</li></ul></div>
    <div class="footer-col"><h4>Legal</h4><ul><li>Privacy Policy</li><li>Terms of Use</li></ul></div>
    <div class="footer-col"><h4>Follow</h4><ul><li>Twitter</li><li>Facebook</li><li>RSS</li></ul></div>
  </div>
  <p class="footer-copy">&copy; 2024 MediaGroup International Ltd. All rights reserved.</p>
</footer>""",

    """<div class="masthead">
  <span class="brand-name">GlobalPress</span>
  <span class="edition">International Edition</span>
  <time class="masthead-date">Monday, April 14, 2025</time>
</div>
<nav class="primary-nav" role="navigation" aria-label="Main menu">
  <ul>
    <li><a href="/breaking">Breaking</a></li><li><a href="/world">World</a></li>
    <li><a href="/analysis">Analysis</a></li><li><a href="/podcasts">Podcasts</a></li>
  </ul>
</nav>
<div class="ad-leaderboard" aria-hidden="true"><span class="ad-label">Advertisement</span></div>
<aside class="sidebar" aria-label="Trending stories">
  <h2>Most Read</h2>
  <ol>
    <li>Global summit reaches landmark deal</li>
    <li>Tech giant announces layoffs</li>
    <li>Climate report: record temperatures logged</li>
  </ol>
</aside>
<footer class="site-footer" role="contentinfo">
  <nav class="footer-nav" aria-label="Footer links">
    <a href="/rss">RSS</a> | <a href="/newsletters">Newsletters</a> |
    <a href="/corrections">Corrections</a> | <a href="/contact">Contact Us</a>
  </nav>
  <p>Registered in England &amp; Wales. Company No. 04135891.</p>
</footer>""",

    """<header class="site-top">
  <a class="skip-link" href="#main-content">Skip to content</a>
  <div class="header-main">
    <a href="/" class="site-logo" aria-label="Homepage">NewsPortal</a>
    <nav class="main-nav">
      <a href="/world">World</a><a href="/science">Science</a>
      <a href="/health">Health</a><a href="/culture">Culture</a>
    </nav>
  </div>
</header>
<div class="cookie-banner" role="dialog" aria-label="Cookie consent">
  <p>We use cookies to improve your experience.</p>
  <button class="btn-accept">Accept all</button>
  <button class="btn-manage">Manage preferences</button>
</div>
<div class="newsletter-signup">
  <p>Get our daily briefing in your inbox</p>
  <form><input type="email" placeholder="your@email.com"/><button>Sign up</button></form>
</div>
<footer>
  <div class="footer-primary">
    <ul class="footer-sections">
      <li><a href="/world">World</a></li><li><a href="/business">Business</a></li>
      <li><a href="/travel">Travel</a></li><li><a href="/future">Future</a></li>
    </ul>
  </div>
  <p class="footer-legal">Copyright &copy; 2024 NewsPortal. All rights reserved.</p>
</footer>""",
]

FAKE_AD_BLOCKS = [
    (
        '<div class="ad-unit sponsored" data-ad-slot="banner-top" aria-hidden="true">'
        '<p class="ad-label">Sponsored</p>'
        '<p>Buy Premium Insurance Today! Free quote in 60 seconds. Limited-time offer.</p>'
        '<a href="#" class="ad-cta">Get Free Quote</a></div>'
    ),
    (
        '<div class="advertisement" data-slot="mid-article" aria-label="Advertisement">'
        '<p class="ad-tag">Advertisement</p>'
        '<p>Flash Sale: 70% off electronics. Free delivery on orders over £50.</p>'
        '<button>Shop Now</button></div>'
    ),
    (
        '<aside class="ad-sidebar" aria-label="Sponsored content">'
        '<span class="label">Paid partnership</span>'
        '<p>Learn a language in 15 minutes a day. 2 million users. Free trial — no card needed.</p>'
        '</aside>'
    ),
    (
        '<div class="promo-block" data-type="native-ad">'
        '<p class="promo-label">Promoted</p>'
        '<p>Best travel deals — flights from £99. Compare and book instantly.</p>'
        '</div>'
    ),
]

# ===========================================================================
# Variant generators
# ===========================================================================

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def variant_timestamp_swap(html: str) -> str:
    """Replace date-like strings with a plausibly shifted date."""
    MONTHS = (
        "January|February|March|April|May|June|"
        "July|August|September|October|November|December"
    )
    patterns = [
        (
            r'\b(\d{4})-(\d{2})-(\d{2})(?:T\d{2}:\d{2}:\d{2}Z?)?\b',
            lambda m: f"{int(m.group(1))-1}-{m.group(2)}-{m.group(3)}"
        ),
        (
            rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\s+(\d{{4}})\b',
            lambda m: f"{(int(m.group(1))%28)+1} {m.group(2)} {int(m.group(3))-1}"
        ),
        (
            rf'\b({MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})\b',
            lambda m: f"{m.group(1)} {(int(m.group(2))%28)+1}, {int(m.group(3))-1}"
        ),
    ]
    text = html
    changed = False
    for pat, repl in patterns:
        new_text, n = re.subn(pat, repl, text)
        if n > 0:
            text = new_text
            changed = True
    if not changed:
        soup = _soup(html)
        head = soup.find("head")
        if head:
            meta = soup.new_tag("meta", attrs={
                "property": "article:published_time",
                "content":  "2019-01-01T00:00:00Z"
            })
            head.append(meta)
        text = str(soup)
    return text


def variant_ad_block_injection(html: str, rng: random.Random) -> str:
    """Inject a realistic ad block after the 2nd paragraph."""
    soup = _soup(html)
    ad_html = rng.choice(FAKE_AD_BLOCKS)
    ad_node = BeautifulSoup(ad_html, "lxml")

    body = soup.find("body")
    if not body:
        return html

    paragraphs = body.find_all("p")
    insert_after = (
        paragraphs[1] if len(paragraphs) > 1 else
        paragraphs[0] if paragraphs else None
    )

    if ad_node.body:
        for child in list(ad_node.body.children):
            child_copy = BeautifulSoup(str(child), "lxml")
            if child_copy.body and child_copy.body.contents:
                node = child_copy.body.contents[0]
                if insert_after:
                    insert_after.insert_after(node)
                else:
                    body.insert(0, node)

    return str(soup)


def variant_url_difference_only(html: str, original_url: str) -> str:
    """Replace the domain in canonical/og:url/base tags with a mirror domain."""
    MIRROR = "mirror-archive.newscdn.net"
    try:
        original_domain = urllib.parse.urlparse(original_url).netloc
    except Exception:
        original_domain = ""

    soup = _soup(html)

    for tag in soup.find_all("link", rel="canonical"):
        href = tag.get("href", "")
        if original_domain and original_domain in href:
            tag["href"] = href.replace(original_domain, MIRROR)

    for tag in soup.find_all("meta", property="og:url"):
        content = tag.get("content", "")
        if original_domain and original_domain in content:
            tag["content"] = content.replace(original_domain, MIRROR)

    base = soup.find("base")
    if base and base.get("href") and original_domain:
        if original_domain in base["href"]:
            base["href"] = base["href"].replace(original_domain, MIRROR)

    return str(soup)


def variant_minor_content_edit(html: str, rng: random.Random, n_swaps: int = 8) -> str:
    """Substitute n_swaps words in <p> tags via synonym map."""
    soup = _soup(html)
    paragraphs = soup.find_all("p")
    if not paragraphs:
        return html

    swaps_done = 0
    shuffled = paragraphs[:]
    rng.shuffle(shuffled)

    for p in shuffled:
        if swaps_done >= n_swaps:
            break
        words = p.get_text().split()
        modified = False
        for i, word in enumerate(words):
            clean = word.lower().strip(".,!?;:\"'()-")
            if clean in SYNONYM_MAP and swaps_done < n_swaps:
                replacement = rng.choice(SYNONYM_MAP[clean])
                if word and word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement
                swaps_done += 1
                modified = True
        if modified:
            p.clear()
            p.append(" ".join(words))

    return str(soup)


def variant_boilerplate_only_diff(html: str, rng: random.Random) -> str:
    """
    KEY EXPERIMENTAL VARIANT.

    Strips the site's structural boilerplate (header/nav/footer/sidebar/
    cookie banners/newsletter sign-ups) and replaces it with a completely
    different synthetic boilerplate. The article body (<article>/<main>) is
    left completely unchanged.

    This is the core comparison your research measures:
      Pipeline A (raw DOM shingling/SimHash): will see large token differences
        due to the boilerplate change -> may fail to detect near-duplicate.
      Pipeline B (trafilatura extraction first): sees identical extracted text
        -> should correctly detect as near-duplicate.
    """
    soup = _soup(html)

    # Find and protect the main content node
    main_content = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id=re.compile(
            r"(^|\b)(content|article|main|body|story)(\b|$)", re.I)) or
        soup.find(class_=re.compile(
            r"(^|\b)(article|story|content|post)[-_]?(body|text|content)?(\b|$)", re.I))
    )

    # Remove structural boilerplate elements
    for tag_name in ["header", "footer", "nav", "aside", "dialog"]:
        for tag in soup.find_all(tag_name):
            if main_content and tag in list(main_content.descendants):
                continue
            tag.decompose()

    # Remove elements with boilerplate class or id names
    bp_re = re.compile(
        r"(navbar|nav-bar|site-header|site-footer|top-bar|bottom-bar|"
        r"sidebar|breadcrumb|cookie|newsletter|related|recommend|"
        r"ad-banner|promo-bar|social-share|share-bar|trending|"
        r"most-read|popular|tag-list|author-bio|comments|skip-link)",
        re.I
    )
    for tag in soup.find_all(class_=bp_re):
        if main_content and tag in list(main_content.descendants):
            continue
        tag.decompose()
    for tag in soup.find_all(id=bp_re):
        if main_content and tag in list(main_content.descendants):
            continue
        tag.decompose()

    # Inject a different synthetic boilerplate at the top of body
    new_bp = BeautifulSoup(rng.choice(BOILERPLATE_TEMPLATES), "lxml")
    body = soup.find("body")
    if body and new_bp.body:
        for child in reversed(list(new_bp.body.children)):
            child_html = BeautifulSoup(str(child), "lxml")
            if child_html.body and child_html.body.contents:
                body.insert(0, child_html.body.contents[0])

    return str(soup)


def variant_full_content_swap(
    original_html: str,
    donor_text: str,
    donor_url: str,
) -> str:
    """
    TRUE NEGATIVE. Replaces the article body with a completely different
    article's content while keeping the original site template intact.
    Both pipelines should correctly return label=0.
    """
    soup = _soup(original_html)
    container = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id=re.compile(r"(content|article|main|body|story)", re.I)) or
        soup.find(class_=re.compile(r"(article|story|content|post)", re.I))
    )
    if container and donor_text:
        container.clear()
        sentences = [s.strip() + "." for s in donor_text.split(".") if len(s.strip()) > 20]
        for chunk in [sentences[i:i+4] for i in range(0, len(sentences), 4)]:
            p = soup.new_tag("p")
            p.string = " ".join(chunk)
            container.append(p)
        title_tag = soup.find("title")
        if title_tag:
            title_tag.string = f"[SWAPPED from {donor_url[:50]}]"
    return str(soup)

# ===========================================================================
# Pair generation
# ===========================================================================

def generate_pairs(articles: list[dict], rng: random.Random) -> list[dict]:
    """
    Generate all six variant pairs for every article.
    full_content_swap uses a random other article from the SAME domain as donor
    to keep the template consistent.
    """
    domain_index: dict[str, list[dict]] = {}
    for a in articles:
        domain_index.setdefault(a["domain"], []).append(a)

    pairs = []
    for art in tqdm(articles, desc="Generating pairs", unit="article"):
        aid  = art["article_id"]
        dom  = art["domain"]
        html = art["raw_html"]
        url  = art["url"]
        text = art["extracted_text"]

        # 1. timestamp_swap
        html_b = variant_timestamp_swap(html)
        pairs.append(_build_pair(art, "timestamp_swap",
                                 html_b, extract_text(html_b) or text, url))

        # 2. ad_block_injection
        html_b = variant_ad_block_injection(html, rng)
        pairs.append(_build_pair(art, "ad_block_injection",
                                 html_b, extract_text(html_b) or text, url))

        # 3. url_difference_only
        html_b = variant_url_difference_only(html, url)
        mirror = re.sub(r"https?://[^/]+", "https://mirror-archive.newscdn.net", url)
        pairs.append(_build_pair(art, "url_difference_only",
                                 html_b, extract_text(html_b) or text, mirror))

        # 4. minor_content_edit
        html_b = variant_minor_content_edit(html, rng, n_swaps=8)
        pairs.append(_build_pair(art, "minor_content_edit",
                                 html_b, extract_text(html_b) or text, url))

        # 5. boilerplate_only_diff  *** KEY experimental case ***
        html_b = variant_boilerplate_only_diff(html, rng)
        pairs.append(_build_pair(art, "boilerplate_only_diff",
                                 html_b, extract_text(html_b) or text, url))

        # 6. full_content_swap (true negative)
        candidates = [a for a in domain_index[dom] if a["article_id"] != aid]
        if candidates:
            donor = rng.choice(candidates)
            html_b = variant_full_content_swap(html, donor["extracted_text"], donor["url"])
            text_b = extract_text(html_b) or donor["extracted_text"]
            pairs.append(_build_pair(art, "full_content_swap",
                                     html_b, text_b, url,
                                     donor_id=donor["article_id"]))
        else:
            log.warning(f"  {aid}: skipping full_content_swap (only 1 article in domain '{dom}')")

    return pairs


def _build_pair(
    art: dict,
    variant_type: str,
    html_b: str,
    text_b: str,
    url_b: str,
    donor_id: str = None,
) -> dict:
    meta = VARIANT_TYPES[variant_type]
    record = {
        "pair_id":      f"{art['article_id']}_{variant_type}",
        "article_id":   art["article_id"],
        "domain":       art["domain"],
        "variant_type": variant_type,
        "label":        meta["label"],
        "label_reason": meta["label_reason"],
        "_html_b":      html_b,   # temp field; stripped before writing JSONL
    }
    if donor_id:
        record["donor_article_id"] = donor_id
    record["page_a"] = {
        "url":            art["url"],
        "extracted_text": art["extracted_text"],
        "html_path":      f"raw_html/{art['article_id']}_original.html",
        "text_path":      f"extracted_text/{art['article_id']}_original.txt",
    }
    record["page_b"] = {
        "url":            url_b,
        "extracted_text": text_b or "",
        "html_path":      f"raw_html/{art['article_id']}_{variant_type}.html",
        "text_path":      f"extracted_text/{art['article_id']}_{variant_type}.txt",
    }
    return record

# ===========================================================================
# Saving
# ===========================================================================

def save_dataset(articles: list[dict], pairs: list[dict], out_dir: Path) -> None:
    (out_dir / "raw_html").mkdir(parents=True, exist_ok=True)
    (out_dir / "extracted_text").mkdir(parents=True, exist_ok=True)

    log.info("Saving original HTML and text files...")
    for art in tqdm(articles, desc="Saving originals", unit="article"):
        (out_dir / "raw_html" / f"{art['article_id']}_original.html").write_text(
            art["raw_html"], encoding="utf-8"
        )
        (out_dir / "extracted_text" / f"{art['article_id']}_original.txt").write_text(
            art["extracted_text"], encoding="utf-8"
        )

    log.info("Saving variant HTML and text files...")
    for pair in tqdm(pairs, desc="Saving variants", unit="pair"):
        html_b = pair.pop("_html_b", "")
        vtype  = pair["variant_type"]
        aid    = pair["article_id"]
        if html_b:
            (out_dir / "raw_html" / f"{aid}_{vtype}.html").write_text(
                html_b, encoding="utf-8"
            )
        (out_dir / "extracted_text" / f"{aid}_{vtype}.txt").write_text(
            pair["page_b"]["extracted_text"], encoding="utf-8"
        )

    # articles.jsonl — metadata only
    log.info("Writing articles.jsonl...")
    with (out_dir / "articles.jsonl").open("w", encoding="utf-8") as f:
        for art in articles:
            rec = {k: v for k, v in art.items() if k != "raw_html"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # pairs.jsonl — file paths only (lightweight index)
    log.info("Writing pairs.jsonl...")
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for pair in pairs:
            slim = {k: v for k, v in pair.items()
                    if k not in ("page_a", "page_b", "_html_b")}
            slim["page_a"] = {k: v for k, v in pair["page_a"].items()
                              if k != "extracted_text"}
            slim["page_b"] = {k: v for k, v in pair["page_b"].items()
                              if k != "extracted_text"}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    # pairs_with_content.jsonl — includes extracted_text for algorithm ingestion
    log.info("Writing pairs_with_content.jsonl...")
    with (out_dir / "pairs_with_content.jsonl").open("w", encoding="utf-8") as f:
        for pair in pairs:
            rec = {k: v for k, v in pair.items() if k != "_html_b"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # stats.json
    label_counts   = Counter(p["label"] for p in pairs)
    variant_counts = Counter(p["variant_type"] for p in pairs)
    domain_counts  = Counter(a["domain"] for a in articles)
    source_counts  = Counter(a.get("source", "unknown") for a in articles)

    stats = {
        "generated_at":         datetime.utcnow().isoformat() + "Z",
        "total_articles":        len(articles),
        "total_pairs":           len(pairs),
        "true_near_duplicates":  label_counts[1],
        "true_negatives":        label_counts[0],
        "articles_per_domain":   dict(domain_counts),
        "articles_per_source":   dict(source_counts),
        "pairs_per_variant":     dict(variant_counts),
        "variant_label_map":     {k: v["label"] for k, v in VARIANT_TYPES.items()},
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(f"\n{'='*60}")
    log.info(f"Dataset saved to: {out_dir.resolve()}")
    log.info(f"  Total articles:        {stats['total_articles']}")
    log.info(f"  Total pairs:           {stats['total_pairs']}")
    log.info(f"  True near-duplicates:  {stats['true_near_duplicates']}")
    log.info(f"  True negatives:        {stats['true_negatives']}")
    log.info(f"{'─'*60}")
    for dom, cnt in domain_counts.items():
        log.info(f"  {dom:<20} {cnt:>4} articles")
    log.info(f"{'─'*60}")
    for v, c in variant_counts.items():
        log.info(f"  {v:<30} {c:>4} pairs  label={VARIANT_TYPES[v]['label']}")
    log.info(f"{'='*60}")
    log.info(f"  Algorithm input file: {out_dir / 'pairs_with_content.jsonl'}")

# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a labelled near-duplicate web page dataset.\n\n"
            "Article sources:\n"
            "  BBC        26 curated URLs (hardcoded)\n"
            "  Al Jazeera 20 curated URLs (hardcoded)\n"
            "  Wikipedia  remainder up to --n total (via random-page API)\n\n"
            "Default: --n 150  =>  26 BBC + 20 AJE + 104 Wikipedia"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, default=150,
        help="Total article target across all sources (default: 150)"
    )
    p.add_argument(
        "--out", default="./dataset",
        help="Output directory (default: ./dataset)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    p.add_argument(
        "--wiki-only", action="store_true",
        help="Skip BBC and AJE; scrape only Wikipedia (useful for testing)"
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    rng    = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []

    if not args.wiki_only:
        # --- BBC (curated) ---
        bbc_articles = scrape_from_url_list(
            CURATED_BBC_URLS,
            domain="bbc",
            display_name="BBC News",
            start_idx=1,
        )
        all_articles.extend(bbc_articles)

        # --- Al Jazeera (curated) ---
        aje_articles = scrape_from_url_list(
            CURATED_AJE_URLS,
            domain="aljazeera",
            display_name="Al Jazeera English",
            start_idx=1,
        )
        all_articles.extend(aje_articles)

    # --- Wikipedia (top-up to reach --n total) ---
    wiki_needed = max(0, args.n - len(all_articles))
    if wiki_needed > 0:
        log.info(
            f"\nCurated articles collected so far: {len(all_articles)}"
            f"\nWikipedia articles needed to reach target {args.n}: {wiki_needed}"
        )
        wiki_articles = scrape_wikipedia(
            n=wiki_needed,
            start_idx=1,
        )
        all_articles.extend(wiki_articles)
    else:
        log.info(
            f"\nTarget of {args.n} already met by curated sources "
            f"({len(all_articles)} articles). Skipping Wikipedia."
        )

    if not all_articles:
        log.error(
            "No articles collected at all.\n"
            "Check network access — both news sites and Wikipedia must be reachable."
        )
        return

    log.info(f"\nTotal articles collected: {len(all_articles)}")
    log.info(f"  BBC:        {sum(1 for a in all_articles if a['domain']=='bbc')}")
    log.info(f"  Al Jazeera: {sum(1 for a in all_articles if a['domain']=='aljazeera')}")
    log.info(f"  Wikipedia:  {sum(1 for a in all_articles if a['domain']=='wikipedia')}")

    log.info("\nGenerating variant pairs (6 per article)...")
    pairs = generate_pairs(all_articles, rng)

    save_dataset(all_articles, pairs, out_dir)


if __name__ == "__main__":
    main()