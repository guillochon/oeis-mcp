"""MCP server over the local OEIS snapshot (oeis_mcp.search): term, text and entry lookups.

    python -m oeis_mcp.server          # stdio transport

Terms go in and come out as comma-separated strings: OEIS terms overflow JSON numbers (2^53) all the time.
The term index is loaded on first use and kept; the large files are memory-mapped, so the process stays small.
"""
import os
import re

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from oeis_mcp import search as S


def _snapshot():
    try:
        return open(os.path.join(S.OEISDATA, "time.txt")).read().strip()
    except OSError:
        return "unknown"


mcp = MCPServer("oeis", instructions=(
    f"Local snapshot of the whole OEIS (oeisdata of {_snapshot()}), searched in well under a second. "
    "Use these tools instead of fetching oeis.org; go to the web only for changes made after the snapshot. "
    "Terms are passed and returned as comma-separated strings. ext=True searches DATA extended with real "
    "b-file terms (up to 256 terms per sequence) instead of DATA alone."))

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_ix = {}


def ix(ext=False):
    if ext not in _ix:
        _ix[ext] = S.TermIndex(ext=ext)
    return _ix[ext]


def _anum(a):
    m = re.fullmatch(r"\s*[Aa]?0*(\d{1,6})\s*", str(a))
    if not m:
        raise ToolError(f"not an A-number: {a!r}")
    return f"A{int(m.group(1)):06d}"


def _row(I, a):
    i = I.index_of(_anum(a))
    if i is None:
        raise ToolError(f"{_anum(a)} is not in the snapshot")
    return i


def _parse(terms):
    try:
        q = [int(x) for x in re.split(r"[,\s]+", terms.strip()) if x]
    except ValueError:
        raise ToolError(f"terms must be comma-separated integers: {terms[:80]!r}") from None
    if len(q) < S.K:
        raise ToolError(f"need at least {S.K} terms")
    return q


def _fields(fields):
    fs = S.TEXT_FIELDS if fields in (None, [], ["all"], "all") else ([fields] if isinstance(fields, str) else fields)
    bad = [f for f in fs if f not in S.TEXT_FIELDS]
    if bad:
        raise ToolError(f"unknown fields {bad}; choose from {S.TEXT_FIELDS} or 'all'")
    return list(fs)


def _summary(I, i):
    return {"anum": I.anums[i], "name": S.field_line("name", i), "keywords": S.field_line("keywords", i),
            "offset": int(I.offsets[i]), "n_terms": I.n_terms(i)}


def _clip(s, n):
    return s if n is None or len(s) <= n else s[:n] + f" …[{len(s) - n} more chars]"


@mcp.tool(annotations=READ_ONLY)
def search_terms(terms: str, ext: bool = False, sub: bool = False, min_run: int = 5, limit: int = 20) -> dict:
    """Find sequences containing the given terms consecutively, like the OEIS search box.

    terms: comma-separated integers, e.g. "1, 2, 5, 14, 42, 132". At least 4 consecutive terms with some
      |term| >= 10 are needed (shorter or tinier queries are too generic to index).
    ext: search DATA plus b-file terms instead of DATA only.
    sub: instead of the whole query, report every maximal common run of >= min_run terms between the query
      and a sequence (useful when the query has a wrong term or a different start), longest first.
    Each hit gives the A-number, name, keywords, offset, and where the match starts: `at` is the index into
    the sequence's terms (0-based), `n` the corresponding n (offset + at)."""
    I, q = ix(ext), _parse(terms)
    if sub:
        res = sorted(((e, sid, b, qa) for (sid, _), (qa, b, e) in I.runs(q, min_run).items()), reverse=True)
        hits = [dict(_summary(I, sid), run=e, query_at=qa, at=b, n=int(I.offsets[sid]) + b)
                for e, sid, b, qa in res[:limit]]
    else:
        if not any(S.informative(q[i:i + S.K]) for i in range(len(q) - S.K + 1)):
            raise ToolError(f"need {S.K} consecutive terms including one with |term| >= 10")
        res = I.contains(q)
        hits = [dict(_summary(I, sid), at=b, n=int(I.offsets[sid]) + b) for sid, b in res[:limit]]
    return {"total": len(res), "shown": len(hits), "terms_source": "DATA+b-file" if ext else "DATA", "hits": hits}


@mcp.tool(annotations=READ_ONLY)
def search_terms_batch(queries: list[str], ext: bool = False, limit: int = 5) -> list[dict]:
    """Run many exact term searches at once (each as in search_terms with sub=False).

    queries: list of comma-separated term strings. Returns, per query, the total number of sequences
    containing it and up to `limit` of their A-numbers with the match position. Queries too generic to index
    come back with an error instead of hits."""
    I, out = ix(ext), []
    for t in queries:
        try:
            q = _parse(t)
            if not any(S.informative(q[i:i + S.K]) for i in range(len(q) - S.K + 1)):
                raise ToolError(f"need {S.K} consecutive terms including one with |term| >= 10")
            res = I.contains(q)
            out.append({"query": _clip(t, 60), "total": len(res),
                        "hits": [{"anum": I.anums[sid], "at": b} for sid, b in res[:limit]]})
        except ToolError as e:
            out.append({"query": _clip(t, 60), "error": str(e)})
    return out


@mcp.tool(annotations=READ_ONLY)
def search_text(pattern: str, fields: list[str] | None = None, limit: int = 30, snippet_chars: int = 160) -> dict:
    """Find sequences whose text fields match a word, phrase or regex (case-insensitive).

    pattern: a plain word/phrase (fast literal scan) or a Python regex, e.g. "Catalan", "plane partitions",
      r"\\bLyndon words?\\b", "Sloane, \\d{4}".
    fields: any of name, comments, formula, xrefs, links, author, programs, keywords; or ["all"].
      Default: ["name"].
    Hits are in A-number order. Each gives the name and keywords, and, when the match is outside the
    name, a snippet around the first match."""
    fs = _fields(fields or ["name"])
    I = ix()
    rows = S.text_search(pattern, fs)
    literal = not (set(pattern) & S.REGEX_META)
    rx = re.compile(re.escape(pattern) if literal else pattern, re.I)
    hits = []
    for i in rows[:limit]:
        h = _summary(I, i)
        for f in fs:
            if f == "name":
                if rx.search(h["name"]):
                    break
                continue
            line = S.field_line(f, i)
            m = rx.search(line)
            if m:
                w = max(0, (snippet_chars - (m.end() - m.start())) // 2)
                a, b = max(0, m.start() - w), min(len(line), m.end() + w)
                h["match"] = {"field": f, "snippet": ("…" if a else "") + line[a:b].replace(" ¶ ", " | ")
                              + ("…" if b < len(line) else "")}
                break
        hits.append(h)
    return {"total": len(rows), "shown": len(hits), "fields": fs, "hits": hits}


@mcp.tool(annotations=READ_ONLY)
def get_entry(anum: str, fields: list[str] | None = None, ext: bool = False, max_terms: int | None = None,
              max_chars: int = 4000) -> dict:
    """One OEIS entry from the snapshot.

    anum: "A000108" (also accepts "108", "a108").
    fields: which text fields to include (name, comments, formula, xrefs, links, author, programs, keywords,
      or ["all"]); default name, keywords, formula, comments, xrefs, author. Terms and offset always come back.
    ext: return DATA extended with the real b-file terms (up to 256) instead of DATA.
    max_terms: cut the terms to this many. max_chars: cut each text field (links and programs can be long).
    For the exact internal-format record (%S/%N/%E... lines, incl. extensions), use get_raw_entry."""
    I = ix(ext)
    i = _row(I, anum)
    fs = _fields(fields) if fields else ["name", "keywords", "formula", "comments", "xrefs", "author"]
    t = I.terms(i)
    out = {"anum": I.anums[i], "offset": int(I.offsets[i]), "n_terms": len(t),
           "terms": ",".join(map(str, t[:max_terms] if max_terms else t)),
           "terms_source": "DATA+b-file" if ext else "DATA", "has_bfile": bool(I.has_bfile[i])}
    for f in fs:
        v = S.field_line(f, i).replace(" ¶ ", "\n")
        if v:
            out[f] = _clip(v, max_chars)
    return out


@mcp.tool(annotations=READ_ONLY)
def get_raw_entry(anum: str, max_chars: int = 20000) -> str:
    """The entry's record exactly as stored in oeisdata (internal format: %I %S %T %U %N %C %D %H %F %e %p %t
    %o %Y %K %O %A %E lines), including extension (%E) lines that get_entry does not carry."""
    a = _anum(anum)
    p = os.path.join(S.OEISDATA, "seq", a[:4], f"{a}.seq")
    if not os.path.exists(p):
        raise ToolError(f"{a} is not in the snapshot")
    return _clip(open(p, encoding="utf-8", errors="replace").read(), max_chars)


@mcp.tool(annotations=READ_ONLY)
def get_summaries(anums: list[str], n_terms: int = 12) -> list[dict]:
    """Short summaries of many entries at once: name, keywords, offset, number of terms and the first n_terms
    terms. Unknown A-numbers come back with an error."""
    I, out = ix(), []
    for a in anums:
        try:
            i = _row(I, a)
            out.append(dict(_summary(I, i), terms=",".join(map(str, I.terms(i)[:n_terms]))))
        except ToolError as e:
            out.append({"anum": str(a), "error": str(e)})
    return out


@mcp.tool(annotations=READ_ONLY)
def get_bfile(anum: str, start: int = 0, count: int = 200) -> dict:
    """Lines of the entry's b-file from the local oeisdata clone (n, a(n) pairs), when it was fetched with
    git lfs. start/count select lines of (n, a(n)) data, comments skipped."""
    a = _anum(anum)
    p = os.path.join(S.BFILES, a[:4], f"b{a[1:]}.txt")
    if not os.path.exists(p):
        return {"anum": a, "available": False, "reason": "no b-file for this entry in the clone"}
    with open(p, encoding="utf-8", errors="replace") as f:
        head = f.readline()
        if head.startswith("version https://git-lfs"):
            return {"anum": a, "available": False, "reason": "b-file not fetched (git lfs pointer)"}
        rows, total = [], 0
        for line in [head] + f.readlines():
            parts = line.split()
            if len(parts) < 2 or parts[0].startswith("#"):
                continue
            if start <= total < start + count:
                rows.append(f"{parts[0]} {parts[1]}")
            total += 1
    return {"anum": a, "available": True, "total_lines": total, "start": start, "lines": "\n".join(rows)}


@mcp.tool(annotations=READ_ONLY)
def index_info() -> dict:
    """Which snapshot is loaded and what is indexed."""
    import json
    meta = json.load(open(os.path.join(S.CACHE, "meta.json")))
    ext = os.path.join(S.CACHE, "ext_meta.json")
    return {"oeisdata_time": _snapshot(), "data_dir": S.DATA_DIR, "sequences": meta["n"],
            "term_windows": meta["windows"], "ext_index": os.path.exists(ext),
            "ext_sequences_extended": json.load(open(ext))["longer"] if os.path.exists(ext) else None,
            "text_fields": S.TEXT_FIELDS}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
