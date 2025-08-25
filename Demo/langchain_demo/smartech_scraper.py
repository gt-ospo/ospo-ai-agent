#!/usr/bin/env python3
"""
smartech_scraper.py
────────────────────────────────────────────────────────────
Download open‑access Georgia Tech PhD theses exactly as they
appear in the SMARTech UI (20 items per page).

Usage:  python smartech_scraper.py OUT_DIR [MAX_PAGES]
"""

import sys, time, pathlib, requests

# ─── configuration ─────────────────────────────────────────
BASE       = "https://repository.gatech.edu"
COLL_ID    = "3b203ae7-3ac9-4107-aae7-d4320ca8e1e0"
PAGE_SIZE  = 20          # UI shows 20 per page
PAUSE      = 0.3         # politeness delay (seconds)
DEBUG      = True        # flip False when happy

# ─── shared HTTP session ───────────────────────────────────
session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent":
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36"
})

def fetch_json(url: str, **params):
    """GET a JSON endpoint and return the parsed body."""
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ─── step 1: list items exactly like the UI ────────────────
def list_items(page: int) -> list[dict]:
    """Return the 20 records visible on UI page <page> (0‑based)."""
    url = f"{BASE}/server/api/discover/search/objects"
    params = {
        "scope": COLL_ID,
        "dsoType": "item",
        "page": page,
        "size": PAGE_SIZE,
        "sort": "dc.date.accessioned,desc"
    }
    data = fetch_json(url, **params)
    return data["_embedded"]["searchResult"]["_embedded"]["objects"]

# ─── step 2: find PDFs inside an item ──────────────────────
def pdf_bitstreams(item_id: str) -> list[dict]:
    """
    Walk every bundle under /items/{id}/bundles and collect bitstreams
    that look like PDFs (mimeType OR filename test).
    """
    bundles_url = f"{BASE}/server/api/core/items/{item_id}/bundles"
    bundles = fetch_json(bundles_url).get("_embedded", {}).get("bundles", [])

    pdfs = []
    for bun in bundles:
        bits_url = f"{BASE}/server/api/core/bundles/{bun['uuid']}/bitstreams"
        bits = fetch_json(bits_url, size=50).get("_embedded", {}).get("bitstreams", [])
        for bs in bits:
            mime = (bs.get("mimeType") or bs.get("format") or "").lower()
            name = bs.get("name", "").lower()
            if mime.startswith("application/pdf") or name.endswith(".pdf"):
                pdfs.append(bs)
                if DEBUG:
                    print(f"      • {bs['name']}  [{mime or '??'}]")
    if DEBUG:
        print(f"      → {len(pdfs)} PDF(s) found")
    return pdfs

# ─── step 3: stream a PDF to disk ──────────────────────────
def save_pdf(bs: dict, out_dir: pathlib.Path):
    target = out_dir / bs["name"]
    if target.exists():
        if DEBUG: print("        ✓", bs["name"], "(already)")
        return
    url = f"{BASE}/server/api/core/bitstreams/{bs['uuid']}/content"
    with session.get(url, stream=True, timeout=90) as r, open(target, "wb") as fh:
        r.raise_for_status()
        for chunk in r.iter_content(8192):
            fh.write(chunk)
    size = int(bs.get("sizeBytes", 0)) / (1024*1024)
    print(f"        ✓ {bs['name']} ({size:.2f} MB)")

# ─── main crawl loop ───────────────────────────────────────
def crawl(dest: pathlib.Path, max_pages: int | None):
    downloaded = skipped = page = 0
    while max_pages is None or page < max_pages:
        items = list_items(page)
        if not items:
            break
        if DEBUG:
            print(f"\n[page {page+1}] {len(items)} items")

        for obj in items:
            rec   = obj["_embedded"]["indexableObject"]
            title = rec["name"][:90]
            if DEBUG:
                print("  ↳", title)

            pdfs = pdf_bitstreams(rec["uuid"])
            if not pdfs:
                skipped += 1
                if DEBUG: print("      → No open PDF")
                continue

            for bs in pdfs:
                save_pdf(bs, dest)
                downloaded += 1
                time.sleep(PAUSE)

        page += 1
        time.sleep(PAUSE)

    print(f"\nFinished.  Downloaded={downloaded}  Skipped={skipped}  Pages={page}")

# ─── CLI entry point ───────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python smartech_scraper.py OUT_DIR [MAX_PAGES]")
    outdir = pathlib.Path(sys.argv[1]).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    limit  = int(sys.argv[2]) if len(sys.argv) == 3 else None
    crawl(outdir, limit)
