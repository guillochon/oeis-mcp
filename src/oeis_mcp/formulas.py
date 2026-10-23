"""Evaluate the relations an OEIS entry states on the parts' current terms (mined from FORMULA / NAME / COMMENTS).

Many FORMULA / COMMENT lines define a sequence through others ("a(n) = A001931(n) - 3*A001168(n) + 3",
"Euler transform of A046091", "partial sums of A080601"). When a part got more terms later, the target can lag.

1. Expressions: every "a(n) = <expr>" line (FORMULA), split at further " = ", cut at the author signature or a
   range clause. <expr> is parsed by a small grammar: integers, n, + - * / ^, parentheses, !, binomial(x, y),
   A######(<expr>) (another sequence, any integer argument, nesting allowed) and a(<expr>) (the sequence itself,
   so recurrences through other sequences work). Lines with other syntax (sums, floor, T(n,k), ...) are skipped.
2. Transform phrases in NAME / COMMENTS / FORMULA: "[inverse] Euler / binomial / Moebius / INVERT / exponential /
   logarithmic / Stirling2 transform of A######", "partial sums of A######", "first differences of A######".
   The transform of the part's terms is aligned with S (shift -3..3).

Lines that say "appears", "conjecture", "seems", "empirical", "apparently", "probably", "?" ... are skipped.
A candidate must agree with every known term of S except at most the first 3 (range conditions such as
"for n > 2"), on at least 6 terms; new terms are computed while all parts have terms and the value is an integer.

    from oeis_mcp import formulas; formulas.init(TermIndex(ext=True)); formulas.extend(row)
"""
import re
import sys
from fractions import Fraction
from math import comb, factorial

from oeis_mcp.search import field_line
from oeis_mcp.transform_lib import all_transforms, apply

MAX_NEW = 400
BIG = 10 ** 1000  # stop extending past 1000-digit terms
MIN_AGREE = 6
MAX_HEAD_SKIP = 3
DOUBT = re.compile(r"appear|conjectur|seem|empiric|apparent|probabl|likely|guess|observ|numerical|\?|presumab|"
                   r"heuristic|if and only if|unproved|unproven", re.I)
CUT = re.compile(r"\s+-\s*_|\s+-\s+[A-Z][a-z]+ [A-Z]|\[|;|\.\s|\s+for\s|,\s*for\s|,\s*n\s*[<>=]|,\s*if\s|\s+if\s|"
                 r"\s+where\s|\(End\)|\s+with\s|,\s*with\s|\s+when\s|\s+otherwise|,\s*a\(|\s+and\s+a\(|\(n\s*[<>]=?")
PHRASES = [
    (r"inverse\s+euler\s+transform\s+of\s+(A\d{6})", "inveuler"),
    (r"euler\s+transform\s+of\s+(A\d{6})", "euler"),
    (r"inverse\s+binomial\s+transform\s+of\s+(A\d{6})", "invbinom"),
    (r"binomial\s+transform\s+of\s+(A\d{6})", "binom"),
    (r"inverse\s+m(?:o|ö|oe)bius\s+transform\s+of\s+(A\d{6})", "sumdiv"),
    (r"m(?:o|ö|oe)bius\s+transform\s+of\s+(A\d{6})", "moebius"),
    (r"inverse\s+invert\s+transform\s+of\s+(A\d{6})", "invinvert"),
    (r"invert\s+transform\s+of\s+(A\d{6})", "invert"),
    (r"exponential\s+transform\s+of\s+(A\d{6})", "exp"),
    (r"logarithmic\s+transform\s+of\s+(A\d{6})", "log"),
    (r"inverse\s+stirling2?\s+transform\s+of\s+(A\d{6})", "invstirling2"),
    (r"stirling2\s+transform\s+of\s+(A\d{6})", "stirling2"),
    (r"partial\s+sums\s+of\s+(A\d{6})", "psum"),
    (r"first\s+differences\s+of\s+(A\d{6})", "diff"),
    (r"hankel\s+transform\s+of\s+(A\d{6})", "hankel"),
    (r"inverse\s+boustrophedon\s+transform\s+of\s+(A\d{6})", "invboustrophedon"),
    (r"boustrophedon\s+transform\s+of\s+(A\d{6})", "boustrophedon"),
    (r"inverse\s+catalan\s+transform\s+of\s+(A\d{6})", "invcatalan"),
    (r"catalan\s+transform\s+of\s+(A\d{6})", "catalan"),
    (r"chebyshev\s+transform\s+of\s+(A\d{6})", "chebyshev"),
    (r"hyperbinomial\s+transform\s+of\s+(A\d{6})", "hyperbinomial"),
    (r"run[\s-]+length\s+transform\s+of\s+(A\d{6})", "runlength"),
    (r"ordinal\s+transform\s+of\s+(A\d{6})", "ordinal"),
    (r"restricted\s+growth\s+sequence\s+transform\s+of\s+(A\d{6})", "rgs"),
    (r"records?\s+transform\s+of\s+(A\d{6})", "records"),
    (r"xor[\s-]+binomial\s+transform\s+of\s+(A\d{6})", "xorbinom"),
    (r"xor[\s-]+m(?:o|ö|oe)bius\s+transform\s+of\s+(A\d{6})", "xormoebius"),
    (r"inverse\s+weigh\s+transform\s+of\s+(A\d{6})", "invweigh"),
    (r"weigh\s+transform\s+of\s+(A\d{6})", "weigh"),
    (r"inverse\s+stirling1\s+transform\s+of\s+(A\d{6})", "invstirling1"),
    (r"stirling1\s+transform\s+of\s+(A\d{6})", "stirling1"),
    (r"inverse\s+invert(?:i)?\s+transform\s+of\s+(A\d{6})", "invinvert"),
    (r"inverti\s+transform\s+of\s+(A\d{6})", "invinvert"),
    (r"inverse\s+exponential\s+transform\s+of\s+(A\d{6})", "log"),
    (r"multiset\s+transform\s+of\s+(A\d{6})", "euler"),
    (r"self[\s-]+convolution\s+of\s+(A\d{6})", "conv2"),
    (r"squares?\s+of\s+(A\d{6})", "square"),
    (r"(?:first\s+)?trisection\s+of\s+(A\d{6})", "trisect0"),
    (r"(?:even\s+)?bisection\s+of\s+(A\d{6})", "bisect0"),
    (r"second\s+differences\s+of\s+(A\d{6})", "diff2"),
    (r"lambert\s+transform\s+of\s+(A\d{6})", "sumdiv"),
]
for _k, _w in ((2, "second|2nd"), (3, "third|3rd"), (4, "fourth|4th"), (5, "fifth|5th")):
    PHRASES.insert(0, (rf"(?:{_w})\s+inverse\s+binomial\s+transform\s+of\s+(A\d{{6}})", f"invbinom{_k}"))
    PHRASES.insert(0, (rf"inverse\s+(?:{_w})\s+binomial\s+transform\s+of\s+(A\d{{6}})", f"invbinom{_k}"))
    PHRASES.insert(0, (rf"(?:{_w})\s+binomial\s+transform\s+of\s+(A\d{{6}})", f"binom{_k}"))
PHRASES = [(re.compile(p, re.I), t) for p, t in PHRASES]


# ---- expression parser ------------------------------------------------------------------------------
TOK = re.compile(r"\s*(A\d{6}|binomial|\d+|n|a|[-+*/^()!,])")


class Unsupported(Exception):
    pass


def tokenize(s):
    out, i = [], 0
    s = s.replace("−", "-").replace("·", "*").replace("×", "*").replace("**", "^").strip().rstrip(".").strip()
    while i < len(s):
        m = TOK.match(s, i)
        if not m or m.start(1) != i + len(s[i:m.start(1)]) - len(s[i:m.start(1)].lstrip()) and not m:
            raise Unsupported(s[i:])
        if m is None:
            raise Unsupported(s[i:])
        gap = s[i:m.start(1)]
        if gap.strip():
            raise Unsupported(gap)
        out.append(m.group(1))
        i = m.end()
    return out


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        tok = self.peek()
        if tok is None or (x is not None and tok != x):
            raise Unsupported(f"expected {x}, got {tok}")
        self.i += 1
        return tok

    def parse(self):
        e = self.expr()
        if self.peek() is not None:
            raise Unsupported(f"trailing {self.peek()}")
        return e

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.eat()
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while True:
            p = self.peek()
            if p in ("*", "/"):
                op = self.eat()
                node = (op, node, self.unary())
            elif p is not None and (p == "(" or p.startswith("A") or p in ("n", "a", "binomial")):
                node = ("*", node, self.unary())  # implicit multiplication: 2A000290(n), 3(n+1)
            else:
                return node

    def unary(self):
        if self.peek() == "-":
            self.eat()
            return ("neg", self.unary())
        if self.peek() == "+":
            self.eat()
            return self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.peek() == "^":
            self.eat()
            return ("^", base, self.unary())
        return base

    def postfix(self):
        node = self.primary()
        while self.peek() == "!":
            self.eat()
            node = ("!", node)
        return node

    def primary(self):
        tok = self.eat()
        if tok.isdigit():
            return ("num", int(tok))
        if tok == "n":
            return ("n",)
        if tok == "(":
            e = self.expr()
            self.eat(")")
            return e
        if tok.startswith("A") or tok == "a":
            self.eat("(")
            arg = self.expr()
            self.eat(")")
            return ("seq", tok, arg)
        if tok == "binomial":
            self.eat("(")
            x = self.expr()
            self.eat(",")
            y = self.expr()
            self.eat(")")
            return ("binom", x, y)
        raise Unsupported(tok)


def refs(node, out):
    if node[0] == "seq":
        out.add(node[1])
        refs(node[2], out)
    else:
        for c in node[1:]:
            if isinstance(c, tuple):
                refs(c, out)
    return out


class Undefined(Exception):
    pass


def evaluate(node, n, look):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "n":
        return n
    if k == "neg":
        return -evaluate(node[1], n, look)
    if k in "+-*/":
        x, y = evaluate(node[1], n, look), evaluate(node[2], n, look)
        if k == "+":
            return x + y
        if k == "-":
            return x - y
        if k == "*":
            return x * y
        if y == 0:
            raise Undefined
        return Fraction(x) / y
    if k == "^":
        x, y = evaluate(node[1], n, look), evaluate(node[2], n, look)
        if Fraction(y).denominator != 1 or y < 0 or y > 5000:
            raise Undefined
        return Fraction(x) ** int(y)
    if k == "!":
        x = evaluate(node[1], n, look)
        if Fraction(x).denominator != 1 or x < 0 or x > 3000:
            raise Undefined
        return factorial(int(x))
    if k == "binom":
        x, y = evaluate(node[1], n, look), evaluate(node[2], n, look)
        if Fraction(x).denominator != 1 or Fraction(y).denominator != 1:
            raise Undefined
        x, y = int(x), int(y)
        return comb(x, y) if 0 <= y <= x else 0
    if k == "seq":
        i = evaluate(node[2], n, look)
        if Fraction(i).denominator != 1:
            raise Undefined
        return look(node[1], int(i))
    raise Undefined


# ---- evaluation on the index ------------------------------------------------------------------------
IX = T = None


def init(ix):
    """Use the term index ix (a TermIndex; DATA + b-file terms when built with ext=True) for the parts' terms."""
    global IX, T
    sys.set_int_max_str_digits(0)
    IX = ix
    if T is None:
        T = all_transforms()


def terms_of(anum):
    j = IX.index_of(anum)
    if j is None:
        return None
    return IX.terms(j), int(IX.offsets[j])


def rhs_pieces(line):
    m = re.search(r"\ba\(n\)\s*=", line)
    if not m:
        return []
    body = line[m.end():]
    c = CUT.search(body)
    if c:
        body = body[:c.start()]
    return [p.strip().rstrip(".,").strip() for p in body.split("=") if p.strip()]


def check_expr(s_anum, S, so, ast):
    cache = {}
    known = {so + i: x for i, x in enumerate(S)}
    computed = {}

    def look(anum, i):
        if anum == "a":
            if i in known:
                return known[i]
            if i in computed:
                return computed[i]
            raise Undefined
        if anum not in cache:
            cache[anum] = terms_of(anum)
        t = cache[anum]
        if t is None or not 0 <= i - t[1] < len(t[0]):
            raise Undefined
        return t[0][i - t[1]]

    agree, bad = 0, []
    for i, x in enumerate(S):
        try:
            y = evaluate(ast, so + i, look)
        except (Undefined, ZeroDivisionError, OverflowError, ValueError):
            continue
        if y == x:
            agree += 1
        else:
            bad.append(i)
    if agree < MIN_AGREE or any(b >= MAX_HEAD_SKIP for b in bad):
        return None
    new = []
    n = so + len(S)
    while len(new) < MAX_NEW:
        try:
            y = evaluate(ast, n, look)
        except (Undefined, ZeroDivisionError, OverflowError, ValueError):
            break
        if Fraction(y).denominator != 1 or abs(y) > BIG:
            break
        computed[n] = int(y)
        new.append(int(y))
        n += 1
    return agree, len(bad), new


def check_transform(S, so, part, tname):
    t = terms_of(part)
    if t is None:
        return None
    if tname == "diff2":
        r1 = apply(T["diff"], t[0], t[1])
        r = apply(T["diff"], *r1) if r1 else None
    else:
        r = apply(T[tname], t[0], t[1])
    if r is None:
        return None
    q, qo = r
    for s in range(-3, 4):
        # q[n + s - qo] == S[n - so] for all known n (except up to MAX_HEAD_SKIP leading)
        agree, bad, ok = 0, [], True
        for i, x in enumerate(S):
            j = so + i + s - qo
            if not 0 <= j < len(q):
                continue
            if q[j] == x:
                agree += 1
            else:
                bad.append(i)
        if agree >= MIN_AGREE and not any(b >= MAX_HEAD_SKIP for b in bad):
            start = so + len(S) + s - qo
            return agree, len(bad), q[start:start + MAX_NEW] if start >= 0 else [], s
    return None


def extend(row):
    """Every stated relation of row's entry that matches its known terms and gives new ones (call init first)."""
    s_anum = IX.anums[row]
    S, so = IX.terms(row), int(IX.offsets[row])
    if len(S) < MIN_AGREE:
        return []
    hits, seen = [], set()
    for line in field_line("formula", row).split(" ¶ "):
        if DOUBT.search(line) or "A" not in line:
            continue
        for piece in rhs_pieces(line):
            if piece in seen:
                continue
            seen.add(piece)
            try:
                ast = Parser(tokenize(piece)).parse()
            except (Unsupported, RecursionError):
                continue
            parts = refs(ast, set()) - {"a"}
            if not parts or parts == {s_anum}:
                continue
            res = check_expr(s_anum, S, so, ast)
            if res and res[2]:
                hits.append({"S": s_anum, "kind": "formula", "relation": "a(n) = " + piece, "parts": " ".join(sorted(parts)),
                             "agree": res[0], "head_mismatch": res[1], "n_new": len(res[2]),
                             "new_terms": " ".join(map(str, res[2])), "source": line[:300]})
    text = " ¶ ".join(field_line(f, row) for f in ("name", "comments", "formula"))
    for line in text.split(" ¶ "):
        if DOUBT.search(line):
            continue
        for rx, tname in PHRASES:
            for m in rx.finditer(line):
                part = m.group(1)
                if part == s_anum or (part, tname) in seen:
                    continue
                # "A is the Euler transform of a" style is excluded by requiring the phrase not to be preceded by "is the"
                seen.add((part, tname))
                res = check_transform(S, so, part, tname)
                if res and res[2]:
                    hits.append({"S": s_anum, "kind": "transform", "relation": f"{tname}({part}) shift {res[3]}",
                                 "parts": part, "agree": res[0], "head_mismatch": res[1], "n_new": len(res[2]),
                                 "new_terms": " ".join(map(str, res[2])), "source": line[:300]})
    return hits
