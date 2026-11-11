---
name: code-docs
description: Generate verified, visual documentation of a codebase and of what Claude changed in it, as a static tabbed web page (changes, folder map, Level 0/1 DFD, sequence diagrams, per-file and per-function summaries, tests with real results). Every file and function name is checked against a parsed inventory, so nothing is invented. Use this whenever the user asks to document code, explain or walk through what Claude changed, make a codebase map, diagrams, DFDs or sequence diagrams of a repo, onboarding docs, a "what did you do" summary after a coding task, or to update existing code-docs, even if they don't say "documentation".
---

# code-docs

Builds `docs/code-docs/index.html`: a static page a human can open and understand in minutes. The reader's main question is **"what did Claude do to this code, and how does it fit together?"** Optimise for that: visuals first, short text, every claim traceable to a line of code.

## The one rule: nothing invented

Wrong docs are worse than no docs, because the reader trusts them. So identifiers never come from your memory. `scripts/codedocs.py` parses the code into `inventory.json` (files, folders, functions, classes, params, types, return types, line numbers, tests). You write **only prose** into `docs.json`, under keys the script created. `verify` then rejects any name in prose keys, diagrams, changes or tests that is not in the inventory.

- Read the actual code before summarising a function. Describe what it does, not what its name suggests.
- If the purpose genuinely can't be determined from the code, write exactly `Unclear from code`. That is a useful signal to the reader, a guess is not.
- Never hand-write test results. They come only from a real run imported with `import-tests`.

## Files

| Path | What it is |
|---|---|
| `scripts/codedocs.py` | All deterministic work. `python <skill>/scripts/codedocs.py --root REPO <cmd>` |
| `assets/index.html` | Fixed viewer template. Copied by `build`; never rewrite it per project. |
| `references/docs-schema.md` | Exact `docs.json` shape, Mermaid rules `verify` enforces, writing rules. **Read before writing docs.json.** |
| `references/example-docs.json` | A complete, verified example (small todo app). |

Output goes to `REPO/docs/code-docs/` (override with `--out`). `index.html` + `data.js` open by double-click, no server or build step.

## Workflow

Track these as tasks; the user may not be watching.

1. **Inventory**: `codedocs.py --root REPO extract`, then `skeleton`. Skeleton adds an empty entry for every folder, source file and symbol, keeps text you already wrote, and moves entries for deleted code to `_orphaned`. If `extract` reports parse errors, mention those files in the final report.
2. **Changes**: `diff` (auto base = merge-base with main/master; pass `--base REF` if the user named one, or if the auto base is wrong for how the work was done, e.g. Claude committed straight to main: then use the commit before Claude's first). The script compares the base with the working tree, so uncommitted and untracked work is included, and fills status and added/modified/removed symbols per file. You add a one-line `reason` per file: why it changed, from the user's request and the code, not a restatement of the diff.
3. **Tests**: run the project's test command with JUnit output, then `import-tests REPORT.xml --command "<cmd>"`.
   pytest `--junitxml=docs/code-docs/junit.xml` | jest `jest-junit` reporter | go `go test -v 2>&1 | go-junit-report` | maven/gradle surefire XML. If tests can't run (missing deps, needs services), don't import anything; they show as "not run" and you say why in the report.
4. **Write prose** in `docs.json` (see schema reference): project summary, folder purposes, file summaries (2-3 lines), function summaries (2-3 lines; inputs/outputs are rendered from the inventory, so don't repeat types), externals, DFD level 0 and 1, sequence diagrams, test groups. Work file by file: open the file, write its entries, move on. For large repos see "Scale" below.
5. **Diagrams** (Mermaid text only, inside docs.json):
   - DFD Level 0: the system as one process, external entities, data stores, labelled data flows. Level 1: the main processes, each a real file/module (label like `store.py<br/>TodoStore`).
   - One sequence diagram per entry point or user-facing flow (CLI command, API route, `main`, event handler, job). Participants are real files/classes; solid arrows carry real function names; dashed arrows carry returned data. List flows you skipped in `skipped_flows` with a reason (same shape as another, trivial, no caller).
   - If the `diagram-craft` skill is available, follow its layout and correctness rules. Keep each diagram readable: about 12 nodes or 8 participants max; split rather than cram.
6. **Stamp and verify**: `stamp` records each file's hash next to its summary (this is what makes later updates incremental), then `verify`. Fix every error and rerun until it prints `0 errors`. Don't silence an error by deleting a true statement; fix the name or the claim.
7. **Build and look**: `build --single docs/code-docs/single.html`, then `render-check` (headless Chrome via Playwright; `--chrome PATH` to use an installed browser). It reports Mermaid syntax errors per tab and saves screenshots to `docs/code-docs/screenshots/`. Look at the screenshots; fix overlapping or unreadable diagrams. If Playwright is missing, `pip install playwright && playwright install chromium`; if impossible, say the render was not checked. Mermaid loads from the jsDelivr CDN, falling back to `mermaid.min.js` next to `index.html` (copy it from `npm i mermaid` for offline use).
8. **Deliver**: if an artifact/publish tool is available, publish `single.html` as a page. Otherwise point to `docs/code-docs/index.html`. Report briefly: counts (files, functions, diagrams), verify result, test results, anything marked `Unclear from code`, parse errors, flows skipped.

## Updating existing docs

When `docs/code-docs/docs.json` already exists, don't start over. Run `extract`, `skeleton`, `stale`. `stale` lists files whose code changed since their summary was stamped; reread only those, update their file and function entries and any diagram they appear in, fill new empty entries, delete `_orphaned`, then `diff`, tests, `stamp`, `verify`, `build`. This keeps reruns cheap.

## Scale

For repos over ~150 source files, writing every summary is slow. Ask the user (or, if unattended, decide and state it): full coverage, or `verify --changed-only`, which requires summaries only for changed files while folders, diagrams, changes and tests stay complete. Summaries for vendored, generated or fixture code can be one short line.

## Writing style

The reader is skimming. Short declarative sentences, concrete nouns, actual behaviour (what's read, written, returned, raised). No marketing words, no filler, no em dashes. Prefer "Returns None if the row is missing" over "Handles retrieval logic gracefully".
