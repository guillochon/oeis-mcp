"""Fast local OEIS search: terms, text (names/comments/formulas/...), and single entries.

    python -m oeis_mcp.search 1,2,3,8,15,52,126          # sequences containing these consecutive terms
    python -m oeis_mcp.search 1,2,3,8,15,52 --sub        # any common run of >= --min terms
    python -m oeis_mcp.search -t "alternating sign"      # regex in names (case-insensitive)
    python -m oeis_mcp.search -t "Fischer" -f all        # ... in name, comments, formula, xrefs, links, author, programs
    python -m oeis_mcp.search A005162                    # show one entry
    python -m oeis_mcp.search --build                    # rebuild the cache (automatic when seqs.pkl is newer)
    python -m oeis_mcp.search --build-ext                # DATA + real b-file terms; select with env OEIS_TERMS=ext

Data lives in DATA_DIR (env OEIS_DATA, default <repo>/data): oeisdata/ (git clone), seqs.pkl
(python -m oeis_mcp.parse) and oeis_cache/.
Cache (oeis_cache/, built from seqs.pkl in ~30 s, memory-mapped afterwards, so a query is ~0.2 s):
  - terms.bin + terms_idx.npy: DATA terms of every sequence, one line each;
  - keys/sid/pos.npy: every window of K consecutive terms with some |term| >= 10, hashed to uint64 and sorted.
    A term query hashes all its informative windows and looks them up in one vectorised searchsorted;
  - text_<field>.txt + text_<field>_idx.npy: one line per sequence, scanned with a single compiled regex.
From Python: `from oeis_mcp.search import TermIndex, text_search, entry`.
"""
import argparse
import json
import mmap
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("OEIS_DATA") or os.path.join(ROOT, "data")
OEISDATA = os.path.join(DATA_DIR, "oeisdata")
SEQS = os.path.join(DATA_DIR, "seqs.pkl")
CACHE = os.path.join(DATA_DIR, "oeis_cache")
K = 4
MASK = (1 << 64) - 1
TEXT_FIELDS = ["name", "comments", "formula", "xrefs", "links", "author", "programs", "keywords"]


def informative(w):
    return max(abs(x) for x in w) >= 10


def whash(w):
    return hash(tuple(w)) & MASK  # int and tuple hashes are deterministic in CPython (no PYTHONHASHSEED effect)


def parse_terms(s):
    out = []
    for t in s.split(","):
        t = t.strip()
        if t:
            try:
                out.append(int(t))
            except ValueError:
                break
    return out


def _p(name):
    return os.path.join(CACHE, name)


_staged = []


def _w(name):
    """Where to write cache file `name`: a staged copy (_new_<name>), put in place by _commit()."""
    _staged.append(name)
    return _p("_new_" + name)


def _commit():
    """Swap the staged files in. Processes using the index (an oeis MCP server, a search script) hold its files
    open, and Windows will not replace those: then nothing is swapped, the old index stays whole, and the new files
    wait as _new_* until a rerun."""
    names, moved = list(_staged), []
    _staged.clear()
    for n in names:
        if os.path.exists(_p(n)):
            try:
                os.replace(_p(n), _p("_old_" + n))
            except OSError as e:
                for m in moved:
                    os.replace(_p("_old_" + m), _p(m))
                raise SystemExit(f"{_p(n)} is in use ({e.strerror}): stop whatever has the index open (oeis MCP "
                                 "servers, search scripts) and rerun; nothing was replaced") from None
            moved.append(n)
    for n in names:
        os.replace(_p("_new_" + n), _p(n))
        if n in moved:
            os.remove(_p("_old_" + n))


def stale():
    m = _p("meta.json")
    return not os.path.exists(m) or json.load(open(m))["seqs_mtime"] != os.path.getmtime(SEQS)


def build():
    import pandas as pd
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    s = pd.read_pickle(SEQS).sort_values("anum").reset_index(drop=True)
    np.save(_w("anum.npy"), s["anum"].str[1:].astype(np.uint32).to_numpy())
    np.save(_w("offset.npy"), s["offset"].astype(np.int32).to_numpy())
    np.save(_w("has_bfile.npy"), s["has_bfile"].astype(bool).to_numpy())

    def lines(name, values):
        """One UTF-8 line per sequence; returns the byte offset of each line start (plus the end)."""
        starts, pos, parts = [0], 0, []
        for v in values:
            v = ((v if isinstance(v, str) else "").replace("\n", " ¶ ") + "\n").encode("utf-8")
            parts.append(v)
            pos += len(v)
            starts.append(pos)
        data = b"".join(parts)
        with open(_w(name + ".txt"), "wb") as f:
            f.write(data)
        if name.startswith("text_"):  # ASCII-lowercased copy (same byte offsets) for fast literal search
            with open(_w(name + ".lower.txt"), "wb") as f:
                f.write(data.lower())
        return np.array(starts, dtype=np.int64)

    kw = s["keywords"].map(lambda k: ",".join(k) if isinstance(k, (list, tuple)) else "")
    for f in TEXT_FIELDS:
        np.save(_w(f"text_{f}_idx.npy"), lines(f"text_{f}", kw if f == "keywords" else s[f] if f in s else [""] * len(s)))
    # Terms: byte offsets (ASCII, so chars == bytes).
    np.save(_w("terms_idx.npy"), lines("terms", s["terms"]))

    keys, sid, pos = [], [], []
    for i, ts in enumerate(s["terms"]):
        t = parse_terms(ts) if isinstance(ts, str) else []
        for j in range(len(t) - K + 1):
            w = t[j:j + K]
            if informative(w):
                keys.append(whash(w))
                sid.append(i)
                pos.append(j)
    keys = np.array(keys, dtype=np.uint64)
    order = np.argsort(keys, kind="stable")
    np.save(_w("keys.npy"), keys[order])
    np.save(_w("sid.npy"), np.array(sid, dtype=np.uint32)[order])
    np.save(_w("pos.npy"), np.array(pos, dtype=np.uint16)[order])
    json.dump({"seqs_mtime": os.path.getmtime(SEQS), "n": len(s), "windows": len(keys)}, open(_w("meta.json"), "w"))
    _commit()
    print(f"built oeis_cache: {len(s)} sequences, {len(keys)} windows in {time.time() - t0:.0f}s", file=sys.stderr)


EXT_TERMS, EXT_DIGITS = 256, 400
BFILES = os.path.join(OEISDATA, "files")


def _bfile_tail(anum, offset, data):
    """DATA extended with the real b-file (git lfs), up to EXT_TERMS terms / EXT_DIGITS digits; DATA if the
    b-file is missing, a pointer stub, has a gap before passing DATA, or disagrees with DATA."""
    p = os.path.join(BFILES, anum[:4], f"b{anum[1:]}.txt")
    if not os.path.exists(p) or len(data) >= EXT_TERMS:
        return data
    out = list(data)
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            prev = None
            for line in f:
                parts = line.split()
                if not parts or parts[0].startswith("#") or len(parts) < 2:
                    if line.startswith("version https://git-lfs"):
                        return data
                    continue
                try:
                    n, v = int(parts[0]), int(parts[1])
                except ValueError:
                    break
                if prev is not None and n != prev + 1:
                    break
                prev = n
                j = n - offset
                if j < 0:
                    continue
                if j < len(data):
                    if data[j] != v:
                        return data
                    continue
                if j != len(out) or len(parts[1]) > EXT_DIGITS:
                    break
                out.append(v)
                if len(out) >= EXT_TERMS:
                    break
    except OSError:
        return data
    return out


def build_ext():
    """Extended cache (oeis_cache/ext_*): DATA plus b-file terms; used when OEIS_TERMS=ext."""
    t0 = time.time()
    base = TermIndex(ext=False)
    n = len(base.anum_int)
    starts, pos, parts = [0], 0, []
    keys, sid, wpos = [], [], []
    kb, sb, pb = [], [], []
    longer = 0
    for i in range(n):
        d = base.terms(i)
        t = _bfile_tail(base.anums[i], int(base.offsets[i]), d)
        longer += len(t) > len(d)
        v = (",".join(map(str, t)) + "\n").encode()
        parts.append(v)
        pos += len(v)
        starts.append(pos)
        for j in range(len(t) - K + 1):
            w = t[j:j + K]
            if informative(w):
                kb.append(whash(w))
                sb.append(i)
                pb.append(j)
        if len(kb) > 2_000_000 or i == n - 1:
            keys.append(np.array(kb, dtype=np.uint64))
            sid.append(np.array(sb, dtype=np.uint32))
            wpos.append(np.array(pb, dtype=np.uint16))
            kb, sb, pb = [], [], []
    with open(_w("ext_terms.txt"), "wb") as f:
        f.write(b"".join(parts))
    np.save(_w("ext_terms_idx.npy"), np.array(starts, dtype=np.int64))
    keys, sid, wpos = np.concatenate(keys), np.concatenate(sid), np.concatenate(wpos)
    order = np.argsort(keys, kind="stable")
    np.save(_w("ext_keys.npy"), keys[order])
    np.save(_w("ext_sid.npy"), sid[order])
    np.save(_w("ext_pos.npy"), wpos[order])
    json.dump({"seqs_mtime": os.path.getmtime(SEQS), "longer": longer, "windows": len(keys)},
              open(_w("ext_meta.json"), "w"))
    _commit()
    print(f"built ext cache: {longer} sequences extended from b-files, {len(keys)} windows in {time.time() - t0:.0f}s",
          file=sys.stderr)


def ensure():
    if stale():
        build()


class TermIndex:
    def __init__(self, ext=None):
        """ext=True (or env OEIS_TERMS=ext): terms are DATA extended with real b-file terms (build_ext)."""
        ensure()
        self.ext = os.environ.get("OEIS_TERMS") == "ext" if ext is None else ext
        x = "ext_" if self.ext else ""
        if self.ext and not os.path.exists(_p("ext_meta.json")):
            raise SystemExit("no ext cache: run python -m oeis_mcp.search --build-ext")
        self.anum_int = np.load(_p("anum.npy"))
        self._anums = None
        self.offsets = np.load(_p("offset.npy"))
        self.has_bfile = np.load(_p("has_bfile.npy"))
        self.keys = np.load(_p(x + "keys.npy"), mmap_mode="r")
        self.sid = np.load(_p(x + "sid.npy"), mmap_mode="r")
        self.pos = np.load(_p(x + "pos.npy"), mmap_mode="r")
        self.tidx = np.load(_p(x + "terms_idx.npy"))
        self._fh = open(_p(x + "terms.txt"), "rb")
        self.blob = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._row = None
        self._cache = {}

    @property
    def anums(self):
        if self._anums is None:
            self._anums = np.array([f"A{x:06d}" for x in self.anum_int.tolist()])
        return self._anums

    @property
    def row(self):
        if self._row is None:
            self._row = {a: i for i, a in enumerate(self.anums.tolist())}
        return self._row

    def index_of(self, anum):
        i = int(np.searchsorted(self.anum_int, int(anum[1:])))
        return i if i < len(self.anum_int) and self.anum_int[i] == int(anum[1:]) else None

    def terms(self, i):
        t = self._cache.get(i)
        if t is None:
            t = self._cache[i] = parse_terms(self.blob[int(self.tidx[i]):int(self.tidx[i + 1]) - 1].decode())
        return t

    def n_terms(self, i):
        return len(self.terms(i))

    def runs(self, q, min_len=6, max_per_window=None):
        """All maximal common runs between q and any sequence: {(sid, shift): (q_start, s_start, length)},
        shift = s_start - q_start, length >= min_len. Windows hitting more than max_per_window sequences
        (generic runs such as factorials) are skipped."""
        wins = [i for i in range(len(q) - K + 1) if informative(q[i:i + K])]
        if not wins:
            return {}
        h = np.fromiter((whash(q[i:i + K]) for i in wins), dtype=np.uint64, count=len(wins))
        lo = np.searchsorted(self.keys, h, "left")
        hi = np.searchsorted(self.keys, h, "right")
        found = {}
        for i, a0, a1 in zip(wins, lo.tolist(), hi.tolist()):
            if a1 == a0 or (max_per_window and a1 - a0 > max_per_window):
                continue
            w = q[i:i + K]
            for sid, j in zip(self.sid[a0:a1].tolist(), self.pos[a0:a1].tolist()):
                key = (sid, j - i)
                f = found.get(key)
                if f is not None and f[0] <= i < f[0] + f[2]:
                    continue
                t = self.terms(sid)
                if t[j:j + K] != w:
                    continue  # hash collision
                a, b = i, j
                while a > 0 and b > 0 and q[a - 1] == t[b - 1]:
                    a, b = a - 1, b - 1
                e = 0
                while a + e < len(q) and b + e < len(t) and q[a + e] == t[b + e]:
                    e += 1
                if e >= min_len and (f is None or f[2] < e):
                    found[key] = (a, b, e)
        return found

    def contains(self, q):
        """Sequences containing q as a consecutive block: [(row, position in DATA)]. Needs an informative window."""
        need = len(q)
        return sorted((sid, b) for (sid, _), (a, b, e) in self.runs(q, min_len=need).items() if a == 0 and e == need)


_text = {}


def _mmap(path):
    fh = open(path, "rb")
    return mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ), fh


def _field(f, lower=False):
    """(memory-mapped UTF-8 bytes, line-start byte offsets) of one text field, or of its lowercased copy."""
    key = (f, lower)
    if key not in _text:
        ensure()
        buf, fh = _mmap(_p(f"text_{f}{'.lower' if lower else ''}.txt"))
        _text[key] = (buf, np.load(_p(f"text_{f}_idx.npy"), mmap_mode="r"), fh)
    return _text[key][:2]


REGEX_META = set(r".^$*+?{}[]\|()")


def text_search(pattern, fields=("name",)):
    """Rows (indices into the sorted A-number list) whose fields match. Case-insensitive for ASCII.
    A plain word or phrase is found with mmap.find on the lowercased field files (~1 GB/s); anything else
    is a regex run on the UTF-8 bytes of the memory-mapped files. Nothing is decoded."""
    pat = pattern.encode("utf-8")
    rows = set()
    if not (set(pattern) & REGEX_META):
        needle = pat.lower()
        for f in fields:
            buf, idx = _field(f, lower=True)
            starts, k = [], buf.find(needle)
            while k >= 0:
                starts.append(k)
                # jump to the next line: one hit per sequence is enough
                line = int(np.searchsorted(idx, k, "right"))
                k = buf.find(needle, int(idx[line]) if line < len(idx) else len(buf))
            if starts:
                rows.update((np.searchsorted(idx, np.array(starts), "right") - 1).tolist())
        return sorted(rows)
    rx = re.compile(pat, re.I | re.M)
    for f in fields:
        buf, idx = _field(f)
        starts = [m.start() for m in rx.finditer(buf)]
        if starts:
            rows.update((np.searchsorted(idx, np.array(starts), "right") - 1).tolist())
    return sorted(rows)


def field_line(f, i):
    buf, idx = _field(f)
    return buf[int(idx[i]):int(idx[i + 1]) - 1].decode("utf-8")


def show(ix, i, extra=""):
    print(f"{ix.anums[i]}  [{ix.n_terms(i)} terms; {field_line('keywords', i)}]{extra}  {field_line('name', i)[:150]}")


def entry(ix, anum):
    i = ix.index_of(anum)
    if i is None:
        return None
    out = {f: field_line(f, i).replace(" ¶ ", "\n") for f in TEXT_FIELDS}
    out.update(anum=anum, offset=int(ix.offsets[i]), terms=ix.terms(i))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="comma-separated terms, or an A-number")
    ap.add_argument("-t", "--text", help="regex to search in text fields")
    ap.add_argument("-f", "--fields", default="name", help="comma list of fields for -t, or 'all' (" + ",".join(TEXT_FIELDS) + ")")
    ap.add_argument("--sub", action="store_true", help="term search: report any common run of >= --min terms")
    ap.add_argument("--min", type=int, default=5)
    ap.add_argument("-n", type=int, default=40, help="max results")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--build-ext", action="store_true", help="build the DATA + b-file term cache (OEIS_TERMS=ext)")
    a = ap.parse_args()
    if a.build:
        build()
    if a.build_ext:
        build_ext()
        return
    ix = TermIndex()
    if a.text:
        fields = TEXT_FIELDS if a.fields == "all" else a.fields.split(",")
        hits = text_search(a.text, fields)
        print(f"{len(hits)} matches in {', '.join(fields)}")
        for i in hits[:a.n]:
            show(ix, i)
    elif a.query and re.fullmatch(r"A\d{6}", a.query.strip()):
        e = entry(ix, a.query.strip())
        if e is None:
            sys.exit("not found")
        for col in ["name", "keywords", "offset", "terms", "formula", "comments", "xrefs"]:
            v = e[col]
            if v not in ("", [], None):
                print(f"--- {col}\n{','.join(map(str, v)) if col == 'terms' else v}")
    elif a.query:
        q = [int(x) for x in re.split(r"[,\s]+", a.query.strip()) if x]
        if a.sub:
            res = sorted(((e, sid, b, qa) for (sid, _), (qa, b, e) in ix.runs(q, a.min).items()), reverse=True)
            print(f"{len(res)} runs of >= {a.min} terms")
            for e, sid, b, qa in res[:a.n]:
                show(ix, sid, f" run {e} (query[{qa}:] = DATA[{b}:])")
        else:
            if not any(informative(q[i:i + K]) for i in range(len(q) - K + 1)):
                sys.exit(f"need {K} consecutive terms including one with |term| >= 10")
            res = ix.contains(q)
            print(f"{len(res)} sequences contain these {len(q)} terms")
            for sid, b in res[:a.n]:
                show(ix, sid, f" at DATA[{b}]")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
