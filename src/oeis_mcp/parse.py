"""Parse oeisdata internal-format .seq files into a single table.

Usage: python -m oeis_mcp.parse [oeisdata dir] [seqs.parquet|.pkl]   (defaults: DATA_DIR/oeisdata, DATA_DIR/seqs.pkl)
"""
import os
import re
import sys
import time
from collections import defaultdict

import pandas as pd

LINE_RE = re.compile(r"^%(\w) (A\d{6})\s?(.*)$")
YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")


def parse_file(path):
    fields = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line.rstrip("\n"))
            if m:
                fields[m.group(1)].append(m.group(3))
    return fields


def to_record(anum, fields):
    terms_str = "".join(fields.get("S", []) + fields.get("T", []) + fields.get("U", []))
    terms = [t for t in terms_str.split(",") if t.strip()]
    keywords = ",".join(fields.get("K", [])).split(",")
    keywords = [k.strip() for k in keywords if k.strip()]
    ext = fields.get("E", [])
    # Years mentioned in extension/author lines approximate when terms were added.
    ext_years = [int(y) for e in ext for y in YEAR_RE.findall(e)]
    author = " ".join(fields.get("A", []))
    author_years = [int(y) for y in YEAR_RE.findall(author)]
    last_term = terms[-1].strip() if terms else ""
    offset = fields.get("O", ["0"])[0].split(",")[0]
    return {
        "anum": anum,
        "name": " ".join(fields.get("N", [])),
        "keywords": keywords,
        "n_terms": len(terms),
        "offset": int(offset) if offset.lstrip("-").isdigit() else 0,
        "last_term": last_term,
        "last_digits": len(last_term.lstrip("-")),
        "terms": ",".join(t.strip() for t in terms),
        "comments": "\n".join(fields.get("C", [])),
        "formula": "\n".join(fields.get("F", [])),
        "links": "\n".join(fields.get("H", [])),
        "has_bfile": any("/b" in h and ".txt" in h for h in fields.get("H", [])),
        "programs": "\n".join(fields.get("p", []) + fields.get("t", []) + fields.get("o", [])),
        "xrefs": "\n".join(fields.get("Y", [])),
        "extensions": "\n".join(ext),
        "author": author,
        "created_year": min(author_years) if author_years else None,
        "last_ext_year": max(ext_years) if ext_years else None,
    }


def progress(i, total, start, width=40):
    elapsed = time.time() - start
    eta = elapsed / i * (total - i)
    filled = width * i // total
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[{bar}] {i}/{total} {100 * i / total:5.1f}%  {elapsed:4.0f}s elapsed, ETA {eta:4.0f}s")
    sys.stderr.flush()


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # also runnable as a script
    from oeis_mcp.search import OEISDATA, SEQS
    root = sys.argv[1] if len(sys.argv) > 1 else OEISDATA
    out = sys.argv[2] if len(sys.argv) > 2 else SEQS
    seqdir = os.path.join(root, "seq")
    paths = []
    for sub in sorted(os.listdir(seqdir)):
        d = os.path.join(seqdir, sub)
        if os.path.isdir(d):
            paths += [os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".seq")]
    recs = []
    total = len(paths)
    start = time.time()
    for i, path in enumerate(paths, 1):
        recs.append(to_record(os.path.basename(path)[:-4], parse_file(path)))
        if i % 2000 == 0 or i == total:
            progress(i, total, start)
    print(file=sys.stderr)
    df = pd.DataFrame(recs).sort_values("anum").reset_index(drop=True)
    df["last_year"] = df[["created_year", "last_ext_year"]].max(axis=1)
    df.to_pickle(out)
    print(f"parsed {len(df)} sequences -> {out}")


if __name__ == "__main__":
    main()
