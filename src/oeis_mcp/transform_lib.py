"""Sequence transforms (110 primitives) for oeis_mcp.identify and oeis_mcp.formulas.

Every transform maps (terms, offset) -> (terms, offset), or None when it does not apply (an inexact division,
a missing leading 1, ...). `offset` is the index of terms[0]. Pointwise transforms use the true index n;
sequence transforms (binomial, Euler, ...) follow the OEIS conventions noted on each.

    python -m oeis_mcp.transform_lib      # self-test against known sequences
"""
import math
from functools import lru_cache
from itertools import accumulate

MAX_DIGITS = 400  # truncate a transformed sequence at the first term longer than this
LIMIT = 10 ** MAX_DIGITS
MAX_TERMS = 300  # and at this many terms (run-length transforms grow exponentially; chains must stay cheap)


def fact(n):
    return math.factorial(n) if n >= 0 else None


def dfact(n):  # double factorial, (-1)!! = 1
    if n < -1:
        return None
    r = 1
    while n > 1:
        r, n = r * n, n - 2
    return r


@lru_cache(maxsize=None)
def bell(n):
    if n < 0:
        return None
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def catalan(n):
    return math.comb(2 * n, n) // (n + 1) if n >= 0 else None


def pw(n, e):
    return n ** e if e >= 0 and not (n == 0 and e == 0) else (1 if e == 0 else None)


# g(n) for pointwise multiply / exact divide.
G = {
    "n!": fact, "(n-1)!": lambda n: fact(n - 1), "(n+1)!": lambda n: fact(n + 1), "(n+2)!": lambda n: fact(n + 2),
    "(2n)!": lambda n: fact(2 * n), "(2n-1)!": lambda n: fact(2 * n - 1), "(2n+1)!": lambda n: fact(2 * n + 1),
    "(2n-1)!!": lambda n: dfact(2 * n - 1), "2^n*n!": lambda n: 2 ** n * fact(n) if n >= 0 else None,
    "n!^2": lambda n: fact(n) ** 2 if n >= 0 else None, "n!(n+1)!": lambda n: fact(n) * fact(n + 1) if n >= 0 else None,
    "C(2n,n)": lambda n: math.comb(2 * n, n) if n >= 0 else None, "Cat(n)": catalan, "Cat(n+1)": lambda n: catalan(n + 1),
    "Bell(n)": bell,
    "2^n": lambda n: pw(2, n), "2^(n-1)": lambda n: pw(2, n - 1), "2^(n+1)": lambda n: pw(2, n + 1),
    "3^n": lambda n: pw(3, n), "4^n": lambda n: pw(4, n), "2^C(n,2)": lambda n: 2 ** (n * (n - 1) // 2) if n >= 0 else None,
    "2^(n^2)": lambda n: 2 ** (n * n) if n >= 0 else None,
    "n": lambda n: n, "n+1": lambda n: n + 1, "n-1": lambda n: n - 1, "n+2": lambda n: n + 2, "2n+1": lambda n: 2 * n + 1,
    "2n-1": lambda n: 2 * n - 1, "n^2": lambda n: n * n, "C(n+1,2)": lambda n: n * (n + 1) // 2, "C(n,2)": lambda n: n * (n - 1) // 2,
    "n^n": lambda n: pw(n, n), "n^(n-1)": lambda n: pw(n, n - 1), "n^(n-2)": lambda n: pw(n, n - 2),
    "(n+1)^(n-1)": lambda n: pw(n + 1, n - 1),
    "2": lambda n: 2, "3": lambda n: 3, "4": lambda n: 4, "6": lambda n: 6, "8": lambda n: 8, "12": lambda n: 12,
    "24": lambda n: 24, "48": lambda n: 48,
}


def pointwise(g, divide):
    def f(t, o):
        out, start = [], None
        for k, x in enumerate(t):
            n = o + k
            try:
                v = g(n)
            except (ValueError, OverflowError, ZeroDivisionError):
                v = None
            if not v:
                if out:
                    break  # keep one contiguous block
                continue
            if start is None:
                start = n
            if divide:
                if x % v:
                    return None
                out.append(x // v)
            else:
                out.append(x * v)
        return (out, start) if out else None
    return f


# ---- sequence transforms (positional: terms[0] is index 0 unless noted) ------------------------------------
def from1(t, o):
    """Terms indexed from 1 (a(1), a(2), ...), dropping a(0) and padding zeros below the offset."""
    if o <= 0:
        return t[1 - o:]
    return [0] * (o - 1) + t


def from0(t, o):
    """Terms indexed from 0, padding zeros below the offset."""
    return t[-o:] if o < 0 else [0] * o + t


def psum(t, o):
    return list(accumulate(t)), o


def diff(t, o):
    return [b - a for a, b in zip(t, t[1:])], o + 1


def binom(t, o):
    return [sum(math.comb(n, k) * t[k] for k in range(n + 1)) for n in range(len(t))], 0


def invbinom(t, o):
    return [sum((-1) ** (n - k) * math.comb(n, k) * t[k] for k in range(n + 1)) for n in range(len(t))], 0


def alt(t, o):
    return [x if (o + k) % 2 == 0 else -x for k, x in enumerate(t)], o


def absval(t, o):
    return ([abs(x) for x in t], o) if any(x < 0 for x in t) else None


def _sigma_c(a):
    m = len(a)
    return [sum(d * a[d - 1] for d in range(1, n + 1) if n % d == 0) for n in range(1, m + 1)]


def euler(t, o):
    """Multiset transform: prod_k (1-x^k)^(-a(k)), input from a(1)."""
    a = from1(t, o)
    c = _sigma_c(a)
    b = [1]
    for n in range(1, len(a) + 1):
        s = sum(c[k - 1] * b[n - k] for k in range(1, n + 1))
        if s % n:
            return None
        b.append(s // n)
    return b, 0


def mobius(n):
    r, p = 1, 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0:
                return 0
            r = -r
        p += 1
    return -r if n > 1 else r


def inveuler(t, o):
    b = from0(t, o)
    if not b or b[0] != 1:
        return None
    m = len(b) - 1
    c = []
    for n in range(1, m + 1):
        c.append(n * b[n] - sum(c[k - 1] * b[n - k] for k in range(1, n)))
    a = []
    for n in range(1, m + 1):
        s = sum(mobius(n // d) * c[d - 1] for d in range(1, n + 1) if n % d == 0)
        if s % n:
            return None
        a.append(s // n)
    return a, 1


def weigh(t, o):
    """Distinct-parts transform: prod_k (1+x^k)^a(k), input from a(1)."""
    a = from1(t, o)
    m = len(a)
    b = [1] + [0] * m
    for k in range(1, m + 1):
        ak = a[k - 1]
        if ak < 0:
            return None
        nb = b[:]  # multiply by (1+x^k)^ak = sum_j C(ak, j) x^(kj)
        for j in range(1, m // k + 1):
            cj = math.comb(ak, j)
            if not cj:
                break
            for n in range(k * j, m + 1):
                nb[n] += cj * b[n - k * j]
        b = nb
    return b, 0


def exp_egf(t, o):
    """Labeled sets: EGF exp of sum a(n) x^n/n!, input from a(1)."""
    a = [0] + from1(t, o)
    m = len(a) - 1
    b = [1] + [0] * m
    for n in range(1, m + 1):
        b[n] = sum(math.comb(n - 1, k - 1) * a[k] * b[n - k] for k in range(1, n + 1))
    return b, 0


def log_egf(t, o):
    b = from0(t, o)
    if not b or b[0] != 1:
        return None
    m = len(b) - 1
    a = [0] * (m + 1)
    for n in range(1, m + 1):
        a[n] = b[n] - sum(math.comb(n - 1, k - 1) * a[k] * b[n - k] for k in range(1, n))
    return a[1:], 1


def invert(t, o):
    """1/(1 - x A(x)) with A(x) = sum a(n+1) x^n: sequences of A-objects (INVERT), input from a(1)."""
    a = from1(t, o)
    b = [1]
    for n in range(1, len(a) + 1):
        b.append(sum(a[k - 1] * b[n - k] for k in range(1, n + 1)))
    return b, 0


def invinvert(t, o):
    b = from0(t, o)
    if not b or b[0] != 1:
        return None
    a = []
    for n in range(1, len(b)):
        a.append(b[n] - sum(a[k - 1] * b[n - k] for k in range(1, n)))
    return a, 1


@lru_cache(maxsize=None)
def stirling2_row(n):
    if n == 0:
        return (1,)
    prev = stirling2_row(n - 1)
    return tuple((k * (prev[k] if k < len(prev) else 0) + (prev[k - 1] if k >= 1 else 0)) for k in range(n + 1))


@lru_cache(maxsize=None)
def stirling1_row(n):  # unsigned
    if n == 0:
        return (1,)
    prev = stirling1_row(n - 1)
    return tuple(((n - 1) * (prev[k] if k < len(prev) else 0) + (prev[k - 1] if k >= 1 else 0)) for k in range(n + 1))


def stirling2(t, o):
    a = from0(t, o)
    return [sum(stirling2_row(n)[k] * a[k] for k in range(n + 1)) for n in range(len(a))], 0


def invstirling2(t, o):
    a = from0(t, o)
    return [sum((-1) ** (n - k) * stirling1_row(n)[k] * a[k] for k in range(n + 1)) for n in range(len(a))], 0


def stirling1(t, o):
    a = from0(t, o)
    return [sum(stirling1_row(n)[k] * a[k] for k in range(n + 1)) for n in range(len(a))], 0


def dirichlet_one(t, o):
    """b(n) = sum_{d | n} a(d) (inverse Moebius), input from a(1)."""
    a = from1(t, o)
    return [sum(a[d - 1] for d in range(1, n + 1) if n % d == 0) for n in range(1, len(a) + 1)], 1


def moebius_tr(t, o):
    a = from1(t, o)
    return [sum(mobius(n // d) * a[d - 1] for d in range(1, n + 1) if n % d == 0) for n in range(1, len(a) + 1)], 1


def ratio(t, o):
    out = []
    for x, y in zip(t, t[1:]):
        if x == 0 or y % x:
            return None
        out.append(y // x)
    return out, o


def selfconv(t, o):
    return [sum(t[k] * t[n - k] for k in range(n + 1)) for n in range(len(t))], 2 * o



# ---- more named transforms (Oct 2026; names as the OEIS uses them) -----------------------------------------
def kbinom(k):
    """k-th binomial transform: b(n) = sum C(n,j) k^(n-j) a(j) (binomial transform applied k times); k < 0 inverts."""
    def f(t, o):
        a = from0(t, o)
        return [sum(math.comb(n, j) * k ** (n - j) * a[j] for j in range(n + 1)) for n in range(len(a))], 0
    return f


@lru_cache(maxsize=None)
def zigzag(n):
    """Euler zigzag numbers A000111 (Seidel-Entringer-Arnold triangle)."""
    row, out = [1], [1]
    for m in range(1, n + 1):
        new = [0]
        for x in reversed(row):
            new.append(new[-1] + x)
        row = new
        out.append(row[-1])
    return tuple(out)


def boustrophedon(t, o):
    """b(n) = sum C(n,k) a(k) E(n-k), E = A000111 (Millar-Sloane-Young)."""
    a = from0(t, o)
    e = zigzag(len(a))
    return [sum(math.comb(n, k) * a[k] * e[n - k] for k in range(n + 1)) for n in range(len(a))], 0


def invboustrophedon(t, o):
    a = from0(t, o)
    e = zigzag(len(a))
    return [sum(math.comb(n, k) * a[k] * e[n - k] * (-1) ** (n - k) for k in range(n + 1)) for n in range(len(a))], 0


def hankel(t, o):
    """Hankel transform: b(n) = det [a(i+j)]_{i,j=0..n-1}, b(0) = 1 (fraction-free Bareiss)."""
    a = from0(t, o)
    out = [1]
    for n in range(1, (len(a) + 1) // 2 + 1):
        M = [[a[i + j] for j in range(n)] for i in range(n)]
        sign, prev = 1, 1
        for k in range(n - 1):
            if M[k][k] == 0:
                sw = next((r for r in range(k + 1, n) if M[r][k]), None)
                if sw is None:
                    M = None
                    break
                M[k], M[sw] = M[sw], M[k]
                sign = -sign
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
            prev = M[k][k]
        out.append(0 if M is None else sign * M[n - 1][n - 1])
    return out, 0


def _compose(a, u, N):
    """A(u(x)) mod x^N for a power series u with u(0) = 0."""
    out, pw = [0] * N, [1] + [0] * (N - 1)
    for k in range(N):
        if k:
            nxt = [0] * N
            for i, x in enumerate(pw):
                if x:
                    for j in range(1, N - i):
                        if u[j]:
                            nxt[i + j] += x * u[j]
            pw = nxt
        if k < len(a) and a[k]:
            out = [p + a[k] * q for p, q in zip(out, pw)]
    return out


def catalan_tr(t, o):
    """Catalan transform (Barry): B(x) = A(x c(x)), c = Catalan g.f."""
    a = from0(t, o)
    N = len(a)
    u = [0] + [math.comb(2 * k, k) // (k + 1) for k in range(N - 1)]
    return _compose(a, u, N), 0


def invcatalan_tr(t, o):
    a = from0(t, o)
    N = len(a)
    return _compose(a, [0, 1, -1] + [0] * max(0, N - 3), N), 0


def chebyshev_tr(t, o):
    """Chebyshev transform: B(x) = (1-x^2)/(1+x^2) A(x/(1+x^2))."""
    a = from0(t, o)
    N = len(a)
    u = [0] * N  # x/(1+x^2) = x - x^3 + x^5 - ...
    for k in range(1, N, 2):
        u[k] = (-1) ** (k // 2)
    c = _compose(a, u, N)
    f = [0] * N  # (1-x^2)/(1+x^2) = 1 - 2x^2 + 2x^4 - ...
    for k in range(0, N, 2):
        f[k] = 1 if k == 0 else 2 * (-1) ** (k // 2)
    return [sum(f[i] * c[n - i] for i in range(n + 1)) for n in range(N)], 0


def hyperbinomial(t, o):
    """b(n) = sum C(n,k) (n-k+1)^(n-k-1) a(k)."""
    a = from0(t, o)
    out = []
    for n in range(len(a)):
        s = 0
        for k in range(n + 1):
            m = n - k
            w = 1 if m == 0 else (m + 1) ** (m - 1)
            s += math.comb(n, k) * w * a[k]
        out.append(s)
    return out, 0


def runlength(t, o):
    """Run-length transform: b(n) = prod over runs of 1's in binary n of a(run length), a from a(1); b(0) = 1."""
    a = from1(t, o)
    m = len(a)
    out = []
    for n in range(1 << min(m + 1, 14)):
        p = 1
        for run in bin(n)[2:].split("0"):
            if run:
                if len(run) > m:
                    return (out, 0) if len(out) >= 6 else None
                p *= a[len(run) - 1]
        out.append(p)
    return out, 0


def ordinal_tr(t, o):
    """Ordinal transform: b(n) = number of times a(n) has occurred among a(o..n)."""
    seen, out = {}, []
    for x in t:
        seen[x] = seen.get(x, 0) + 1
        out.append(seen[x])
    return (out, o) if 2 * len(seen) <= len(t) else None  # mostly distinct values: ~all 1's, no information


def rgs_tr(t, o):
    """Restricted growth sequence transform: replace each value by the rank of its first occurrence (1, 2, ...)."""
    rank, out = {}, []
    for x in t:
        if x not in rank:
            rank[x] = len(rank) + 1
        out.append(rank[x])
    return (out, o) if 2 * len(rank) <= len(t) else None  # mostly distinct values: ~1, 2, 3, ...


def records_tr(t, o):
    """Record values (RECORDS transform)."""
    out = []
    for x in t:
        if not out or x > out[-1]:
            out.append(x)
    return (out, 1) if len(out) < len(t) else None  # increasing: records = the sequence itself


def xorbinom(t, o):
    """b(n) = XOR of a(k) over k with C(n,k) odd (k a submask of n)."""
    a = from0(t, o)
    if any(x < 0 for x in a):
        return None
    out = []
    for n in range(len(a)):
        v, k = 0, n
        while True:
            v ^= a[k]
            if k == 0:
                break
            k = (k - 1) & n
        out.append(v)
    return out, 0


def xormoebius(t, o):
    """b(n) = XOR of a(d) over d | n with n/d squarefree, a from a(1)."""
    a = from1(t, o)
    if any(x < 0 for x in a):
        return None
    return [_xor(a[d - 1] for d in range(1, n + 1) if n % d == 0 and mobius(n // d)) for n in range(1, len(a) + 1)], 1


def _xor(it):
    v = 0
    for x in it:
        v ^= x
    return v


def _gcomb(a, j):
    """Generalized binomial C(a, j) for any integer a."""
    r = 1
    for i in range(j):
        r = r * (a - i)
    return r // math.factorial(j)


def invweigh(t, o):
    """Inverse of weigh: a with prod_k (1+x^k)^a(k) = B(x)."""
    b = from0(t, o)
    if not b or b[0] != 1:
        return None
    m = len(b) - 1
    cur = [1] + [0] * m  # product over k < n
    a = []
    for n in range(1, m + 1):
        an = b[n] - cur[n]
        a.append(an)
        nb = cur[:]
        for j in range(1, m // n + 1):
            cj = _gcomb(an, j)
            if not cj:
                continue
            for i in range(n * j, m + 1):
                nb[i] += cj * cur[i - n * j]
        cur = nb
    return a, 1


def invstirling1(t, o):
    a = from0(t, o)
    return [sum((-1) ** (n - k) * stirling2_row(n)[k] * a[k] for k in range(n + 1)) for n in range(len(a))], 0


def invhyperbinomial(t, o):
    """Inverse of hyperbinomial (unitriangular, solved row by row)."""
    b = from0(t, o)
    a = []
    for n in range(len(b)):
        s = b[n]
        for k in range(n):
            m = n - k
            s -= math.comb(n, k) * (m + 1) ** (m - 1) * a[k]
        a.append(s)
    return a, 0


def invrunlength(t, o):
    """a(k) = b(2^k - 1) (the run-length transform evaluated at a single run of k ones)."""
    b = from0(t, o)
    a, k = [], 1
    while (1 << k) - 1 < len(b):
        a.append(b[(1 << k) - 1])
        k += 1
    return a, 1


def isqrt_seq(t, o):
    out = []
    for x in t:
        if x < 0:
            return None
        r = math.isqrt(x)
        if r * r != x:
            return None
        out.append(r)
    return out, o


def capped(f, n):
    """Use at most the first n terms: the transform is cubic or worse in the length (with growing integers)."""
    return lambda t, o: f(t[:n], o)


MORE = {
    "hankel": capped(hankel, 40), "boustrophedon": boustrophedon, "invboustrophedon": invboustrophedon,
    "catalan": capped(catalan_tr, 80), "invcatalan": capped(invcatalan_tr, 80), "chebyshev": capped(chebyshev_tr, 80),
    "hyperbinomial": capped(hyperbinomial, 120),
    "runlength": runlength, "ordinal": ordinal_tr, "rgs": rgs_tr, "records": records_tr,
    "xorbinom": xorbinom, "xormoebius": capped(xormoebius, 200), "invweigh": capped(invweigh, 120),
    "invstirling1": invstirling1, "invhyperbinomial": capped(invhyperbinomial, 120), "invrunlength": invrunlength,
    "sqrt": isqrt_seq,
    "trisect0": lambda t, o: (t[(-o) % 3::3], 0), "trisect1": lambda t, o: (t[(1 - o) % 3::3], 1),
    "trisect2": lambda t, o: (t[(2 - o) % 3::3], 2), "square": lambda t, o: ([x * x for x in t], o),
}
for _k in (2, 3, 4, 5):
    MORE[f"binom{_k}"] = kbinom(_k)
    MORE[f"invbinom{_k}"] = kbinom(-_k)

SEQ = {
    "psum": psum, "diff": diff, "binom": binom, "invbinom": invbinom, "alt": alt, "abs": absval,
    "euler": euler, "inveuler": inveuler, "weigh": weigh, "exp": exp_egf, "log": log_egf,
    "invert": invert, "invinvert": invinvert, "stirling2": stirling2, "invstirling2": invstirling2,
    "stirling1": stirling1, "sumdiv": dirichlet_one, "moebius": moebius_tr, "ratio": ratio, "conv2": selfconv,
    "bisect0": lambda t, o: (t[(o % 2)::2], 0), "bisect1": lambda t, o: (t[1 - (o % 2)::2], 1),
    "+1": lambda t, o: ([x + 1 for x in t], o), "-1": lambda t, o: ([x - 1 for x in t], o),
}

# Index-type transforms: meaningful as a whole relation (S = ordinal(B)), not as a step inside a chain.
NOCHAIN = {"ordinal", "rgs", "records", "runlength"}

# A smaller set used for the third step of depth-3 chains.
CORE = ["*n!", "/n!", "*2^n", "/2^n", "*n", "/n", "*2", "/2", "psum", "diff", "binom", "invbinom", "euler",
        "inveuler", "exp", "log", "invert", "invinvert", "bisect0", "bisect1", "+1", "-1", "alt"]


def all_transforms():
    ts = {}
    for g, fn in G.items():
        ts["*" + g] = pointwise(fn, False)
        ts["/" + g] = pointwise(fn, True)
    ts.update(SEQ)
    ts.update(MORE)
    return ts


def apply(f, t, o):
    try:
        r = f(t, o)
    except (ZeroDivisionError, OverflowError, IndexError, ValueError, TypeError):
        return None
    if r is None:
        return None
    t2, o2 = r
    t2 = t2[:MAX_TERMS]
    for k, x in enumerate(t2):
        if abs(x) > LIMIT:
            t2 = t2[:k]
            break
    return (t2, o2) if len(t2) >= 6 else None


def _selftest():
    T = all_transforms()
    ones = [1] * 12
    checks = [
        ("euler", ones, 1, [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77]),      # partitions
        ("exp", ones, 1, [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597]),  # Bell
        ("stirling2", [1] * 10, 0, [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147]),  # Bell
        ("invert", ones, 1, [1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        ("weigh", ones, 1, [1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 15]),         # distinct parts
        ("sumdiv", ones, 1, [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6]),               # tau
        ("binom", ones, 0, [2 ** n for n in range(12)]),
    ]
    for name, t, o, want in checks:
        got = T[name](t, o)[0]
        assert got == want, (name, got)
    # inverses
    p = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56]
    assert T["inveuler"](p, 0)[0] == [1] * 11
    assert T["log"]([1, 1, 2, 5, 15, 52, 203, 877], 0)[0] == [1] * 7
    assert T["invinvert"]([1, 1, 2, 4, 8, 16, 32], 0)[0] == [1] * 6
    assert T["invstirling2"]([1, 1, 2, 5, 15, 52, 203], 0)[0] == [1] * 7
    assert T["moebius"]([1, 2, 2, 3, 2, 4, 2, 4], 1)[0] == [1] * 8
    assert T["/n!"]([1, 1, 2, 6, 24, 120], 0) == ([1] * 6, 0)
    print(f"transform_lib: {len(T)} primitives, self-test passed")


if __name__ == "__main__":
    _selftest()
