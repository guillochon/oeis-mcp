"""Guess P-recursive (holonomic) recurrences: sum_{i=0..r} p_i(n) a(n-i) = 0, deg p_i <= d.

Try (r, d) in order of the number of unknowns (r+1)(d+1), r <= MAX_R, d <= MAX_D, and keep the first recurrence
whose kernel (mod a 61-bit prime, then exact over Z with python-flint) is one-dimensional with at least SPARE more
equations than unknowns. Such a recurrence is a guess, not a proof, and it predicts further terms (integer-checked).
"""
from math import gcd

import flint

P = (1 << 61) - 1
SPARE = 6
MAX_R, MAX_D = 6, 6
PREDICT = 20


def system(a, o, r, d):
    """Rows: for n = o+r .. o+len(a)-1, the coefficients of c_{i,j} = n^j a(n-i)."""
    rows = []
    for k in range(r, len(a)):
        n = o + k
        rows.append([n ** j * a[k - i] for i in range(r + 1) for j in range(d + 1)])
    return rows


def guess(a, o, max_r=MAX_R, max_d=MAX_D, spare=SPARE):
    """(r, d, c) for the first recurrence found, c the coefficients c_{i,j} in row-major (i, j) order; or None."""
    combos = sorted(((r, d) for r in range(1, max_r + 1) for d in range(max_d + 1)),
                    key=lambda x: ((x[0] + 1) * (x[1] + 1), x[0]))
    for r, d in combos:
        u = (r + 1) * (d + 1)
        if len(a) - r < u + spare:
            continue
        rows = system(a, o, r, d)
        M = flint.nmod_mat([[x % P for x in row] for row in rows], P)
        X, nul = M.nullspace()
        if nul != 1:
            continue
        Mz = flint.fmpz_mat(rows)
        Xz, nz = Mz.nullspace()
        if nz != 1:
            continue
        c = [int(Xz[i, 0]) for i in range(Xz.nrows())]
        g = 0
        for x in c:
            g = gcd(g, x)
        c = [x // g for x in c]
        p0 = c[:d + 1]
        if not any(p0) or all(x == 0 for x in c[d + 1:]):
            continue  # needs a(n) and some earlier term
        return r, d, c
    return None


def predict(a, o, r, d, c, count=PREDICT):
    """The next `count` terms from the recurrence, and whether every one came out an integer."""
    out = list(a)
    for _ in range(count):
        n = o + len(out)
        lead = sum(c[j] * n ** j for j in range(d + 1))
        if lead == 0:
            break
        rest = sum(c[i * (d + 1) + j] * n ** j * out[len(out) - i] for i in range(1, r + 1) for j in range(d + 1))
        if rest % lead:
            return out[len(a):], False
        out.append(-rest // lead)
    return out[len(a):], True


def poly_str(cs):
    t = [f"{x:+d}" + (f"*n^{j}" if j > 1 else "*n" if j == 1 else "") for j, x in enumerate(cs) if x]
    return "(" + " ".join(t) + ")" if t else "0"


def recurrence_str(r, d, c):
    return " + ".join(f"{poly_str(c[i * (d + 1):(i + 1) * (d + 1)])}*a(n-{i})".replace("a(n-0)", "a(n)")
                      for i in range(r + 1) if any(c[i * (d + 1):(i + 1) * (d + 1)])) + " = 0"
