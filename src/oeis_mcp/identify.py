"""Transform identification: is a sequence a simple transform of some OEIS sequence? (a local "superseeker")

Apply chains of up to `depth` primitive transforms (oeis_mcp.transform_lib: multiply/divide by n!, (2n)!, 2^n,
2^C(n,2), Catalan, Bell, n^n, ...; binomial, Euler, weigh, EGF exp/log, INVERT, Stirling and Moebius transforms
and their inverses; partial sums, differences, ratios, bisections, signs, +-1) and look each distinct result up in
the term index. Depth 3 uses a smaller core set for the last step. A hit S is kept when the common run is long and
carries real information, and runs to the end of T(B)'s data or of S's data (so the two agree everywhere they
overlap from the run on).

  extend_target: S's DATA continues past the end of T(B)  ->  S may give new terms of B
  extend_linked: T(B) continues past the end of S's DATA  ->  B's terms may give new terms of S
"""
import time

from oeis_mcp.transform_lib import CORE, NOCHAIN, all_transforms, apply

MAX_MATCHES = 8       # a transformed query matching more sequences than this is generic
MAX_PER_WINDOW = 40   # a 4-term window found in more sequences than this is generic (factorials, powers, ...)

_T = None


def transforms():
    global _T
    if _T is None:
        _T = all_transforms()
    return _T


def digits(xs):
    return sum(len(str(abs(x))) for x in xs)


def accept(run, depth):
    """A run must be long and informative; deeper chains (many more of them) need more evidence."""
    big = sum(abs(x) >= 100 for x in run)
    # Slowly varying runs (1, 2, 4, 5, 7, ...) match hundreds of unrelated "numbers such that" sequences.
    steep = sum(abs(y - x) > 3 for x, y in zip(run, run[1:]))
    need_len, need_digits = {0: (6, 20), 1: (6, 20), 2: (8, 30)}.get(depth, (10, 40))
    return len(run) >= need_len and big >= 4 and steep >= 4 and digits(run) >= need_digits


def identify(ix, t, o, depth=2, exclude=None, deadline=None):
    """Hits of T(t) in the index for every chain T of up to `depth` primitives (t has offset o), and the number of
    distinct transformed sequences looked up. `exclude`: a row never reported (the target itself). `deadline`:
    a time.time() value after which no further chains are tried (the result then says complete=False).
    Returns (hits, n_queries, complete); each hit is a dict with transform, depth, sid, run, run_digits,
    extend_target, extend_linked, extra_terms."""
    T = transforms()
    names_all = list(T)
    seen = {tuple(t[:16])}
    hits, n_queries = [], 0

    def query(q, chain):
        nonlocal n_queries
        n_queries += 1
        d = len(chain)
        runs = ix.runs(q, 6, MAX_PER_WINDOW)
        if not runs or len({sid for sid, _ in runs}) > MAX_MATCHES:
            return
        for (sid, _), (qa, sb, e) in runs.items():
            if sid == exclude or not accept(q[qa:qa + e], d):
                continue
            n_s = ix.n_terms(sid)
            q_end, s_end = qa + e == len(q), sb + e == n_s
            if not (q_end or s_end):
                continue  # they disagree right after the run
            hits.append({"transform": " then ".join(chain) or "id", "depth": d, "sid": sid, "run": e,
                         "run_digits": digits(q[qa:qa + e]), "extend_target": q_end and not s_end,
                         "extend_linked": s_end and not q_end,
                         "extra_terms": (n_s - sb - e) if q_end else (len(q) - qa - e)})

    query(t, ())
    level = [((), t, o)]
    for d in range(1, depth + 1):
        names = names_all if d <= 2 else CORE
        nxt = []
        for chain, q0, o0 in level:
            if deadline is not None and time.time() > deadline:
                return hits, n_queries, False
            for name in names:
                r = apply(T[name], q0, o0)
                if r is None:
                    continue
                key = tuple(r[0][:16])
                if key in seen:  # same sequence via a shorter or earlier chain
                    continue
                seen.add(key)
                ch = chain + (name,)
                query(r[0], ch)
                if d < depth and name not in NOCHAIN:
                    nxt.append((ch, r[0], r[1]))
        level = nxt
    return hits, n_queries, True
