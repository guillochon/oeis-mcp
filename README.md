# oeis-mcp

A fast index over a local snapshot of the [OEIS](https://oeis.org), and an [MCP](https://modelcontextprotocol.io)
server that lets Claude Code (or any MCP client) search it without going to oeis.org.

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
| `index_info` | Snapshot date and what is indexed |

Terms are passed and returned as comma-separated strings, because OEIS terms routinely exceed what JSON numbers
hold exactly. `ext=True` searches DATA extended with real b-file terms (up to 256 per sequence) instead of DATA alone.

## Data

Everything lives in `data/`, which is not tracked. The location can be changed with the `OEIS_DATA` env var.

```
data/oeisdata/     git clone of github.com/oeis/oeisdata (~680k files); b-files fetched with git lfs
data/seqs.pkl      one row per sequence (python -m oeis_mcp.parse, ~400 MB)
data/oeis_cache/   the search index (python -m oeis_mcp.search --build, ~30 s; --build-ext for b-file terms)
```

To set it up from scratch, or refresh it:

```bash
git clone --depth 1 https://github.com/oeis/oeisdata.git data/oeisdata   # or: git -C data/oeisdata pull
python -m oeis_mcp.parse                 # -> data/seqs.pkl
python -m oeis_mcp.search --build        # automatic anyway when seqs.pkl is newer than the cache
python -m oeis_mcp.search --build-ext    # optional: DATA + b-file terms (needs b-files via git lfs)
```

## Install and register

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e .
claude mcp add --scope user oeis -- C:/Users/guill/oeis-mcp/.venv/Scripts/python.exe -m oeis_mcp.server
```

## Command line and Python

The same index from a shell:

```bash
python -m oeis_mcp.search 1,2,5,14,42,132        # sequences containing these terms
python -m oeis_mcp.search 1,2,5,14,42 --sub      # partial overlaps
python -m oeis_mcp.search -t "plane partitions" -f all
python -m oeis_mcp.search A000108
```

And from Python: `from oeis_mcp.search import TermIndex, text_search, field_line, entry`. Only numpy is needed for
search, and pandas for building. The `mcp` package is only needed for the server.

## How the index works

- Every window of 4 consecutive terms that has some |term| ≥ 10 is hashed to a uint64. The hashes are sorted,
  so a query looks up all its windows with one vectorised `searchsorted`. Hits are then extended to maximal runs
  against the stored terms, which also weeds out hash collisions.
- Each text field is one UTF-8 line per sequence, with line-start offsets, plus an ASCII-lowercased copy. A plain
  word or phrase is found with `mmap.find` (~1 GB/s); a regex runs on the mapped bytes. Nothing is decoded in bulk.
