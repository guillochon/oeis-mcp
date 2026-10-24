# oeis-mcp

A fast index over a local snapshot of the [OEIS](https://oeis.org), and an [MCP](https://modelcontextprotocol.io)
server that lets Claude Code (or any MCP client) search it without going to oeis.org. It also
identifies a sequence as a transform of a known one, guesses recurrences, and evaluates the relations entries state.

Lookups take well under a second. The index files are memory-mapped, so the server process stays small (tens of MB).
The pages it touches sit in the OS file cache, and the OS takes them back when it needs the memory.

## Tools

| Tool | What it does |
|---|---|
| `search_terms` | Sequences containing the given terms consecutively; `sub=True` finds partial overlaps (runs of ≥ `min_run` terms) |
| `search_terms_batch` | Many exact term searches in one call |
| `search_text` | Word, phrase or regex in names, comments, formulas, xrefs, links, authors, programs, keywords, with snippets |
| `get_entry` | One entry: terms, offset and chosen text fields (long fields are clipped) |
| `get_raw_entry` | The entry's exact internal-format record (`%S %N %C ... %E`), including extensions |
| `get_summaries` | Name, keywords and first terms of many entries at once |
| `get_bfile` | Lines of the entry's b-file, when the clone has it (git lfs) |
| `identify_transform` | Is this sequence (terms, or an A-number) a chain of up to 3 of ~140 transforms of a known one? A local superseeker (~2 s at depth 2) |
| `guess_recurrence` | Guess a P-recursive recurrence (polynomial coefficients, order and degree ≤ 6) and predict terms |
| `extend_from_formulas` | Evaluate an entry's own stated relations (`a(n) = ...A######(n)...`, "Euler transform of A######", ...) on the other sequences' current terms, giving new terms |
| `index_info` | Snapshot date and what is indexed |

Terms are passed and returned as comma-separated strings, because OEIS terms routinely exceed what JSON numbers
hold exactly. `ext=True` searches DATA extended with real b-file terms (up to 256 per sequence) instead of DATA alone.

## Data

Everything lives in `data/`, which is not tracked. The location can be changed with the `OEIS_DATA` env var.

```
data/oeisdata/     shallow clone of github.com/oeis/oeisdata (~680k files); b-files (~245k, ~35 GB) via git lfs
data/seqs.pkl      one row per sequence (python -m oeis_mcp.parse, ~400 MB)
data/oeis_cache/   the search index (~3.3 GB; python -m oeis_mcp.search --build; --build-ext for b-file terms)
```

To set it up from scratch (the clone's `.lfsconfig` fetches no LFS content by default, so the b-files are asked
for explicitly; skip those lines to work from DATA alone):

```bash
git clone --depth 1 https://github.com/oeis/oeisdata.git data/oeisdata
git -C data/oeisdata lfs fetch origin main -I "files/*/b*.txt" -X ""
git -C data/oeisdata lfs checkout "files/*/b*.txt"
```

To refresh it (the OEIS pushes a new snapshot daily):

```bash
git -C data/oeisdata fetch --depth 1 origin main
git -C data/oeisdata lfs fetch origin origin/main -I "files/*/b*.txt" -X ""   # new b-files first, so the
git -C data/oeisdata reset --hard origin/main                                 # checkout fills them in
python -m oeis_mcp.parse                 # -> data/seqs.pkl
python -m oeis_mcp.search --build        # also automatic when seqs.pkl is newer than the cache
python -m oeis_mcp.search --build-ext    # DATA + b-file terms
```

The cache is rebuilt into staged `_new_*` files and swapped in at the end. Windows will not replace a file another
process has open, so stop running oeis MCP servers (and search scripts) first; if one is still running, the swap is
refused and the old index stays as it was.

## Install and register

```bash
git clone https://github.com/guillochon/oeis-mcp.git && cd oeis-mcp
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e .     # .venv/bin/python on Linux/macOS
claude mcp add --scope user oeis -- "$PWD/.venv/Scripts/python.exe" -m oeis_mcp.server
```

Then set up `data/` as described above. The OEIS data itself is not part of this repo; it is the
[OEIS](https://oeis.org)'s, under its own [license](https://oeis.org/LICENSE) (CC BY-SA 4.0).

## Command line and Python

The same index from a shell:

```bash
python -m oeis_mcp.search 1,2,5,14,42,132        # sequences containing these terms
python -m oeis_mcp.search 1,2,5,14,42 --sub      # partial overlaps
python -m oeis_mcp.search -t "plane partitions" -f all
python -m oeis_mcp.search A000108
```

And from Python: `from oeis_mcp.search import TermIndex, text_search, field_line, entry`, plus `oeis_mcp.identify`
(`identify`), `oeis_mcp.recur` (`guess`, `predict`), `oeis_mcp.formulas` (`init`, `extend`) and
`oeis_mcp.transform_lib` (`all_transforms`, `apply`; `python -m oeis_mcp.transform_lib` self-tests). Only numpy is
needed for search, pandas for building, python-flint for recurrences, and `mcp` for the server.

## How the index works

- Every window of 4 consecutive terms that has some |term| ≥ 10 is hashed to a uint64. The hashes are sorted,
  so a query looks up all its windows with one vectorised `searchsorted`. Hits are then extended to maximal runs
  against the stored terms, which also weeds out hash collisions.
- Each text field is one UTF-8 line per sequence, with line-start offsets, plus an ASCII-lowercased copy. A plain
  word or phrase is found with `mmap.find` (~1 GB/s); a regex runs on the mapped bytes. Nothing is decoded in bulk.
