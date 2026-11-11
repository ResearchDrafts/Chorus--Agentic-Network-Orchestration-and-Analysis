# docs.json reference

`skeleton` creates this file; `diff` and `import-tests` fill their sections. You edit only the prose fields marked ✍. IDs are always copied from `inventory.json` (symbol ID = `path::Qualname`, e.g. `app/store.py::TodoStore.add`; duplicates get `@L<line>`).

```jsonc
{
  "project": { "name": "repo", "summary": "✍ one line", "repo_url": "✍ optional, e.g. https://github.com/o/r (adds line links)", "base_ref": "set by diff" },
  "externals": [               // ✍ everything in diagrams that is NOT code in this repo
    { "name": "User", "kind": "entity", "note": "runs the CLI" },
    { "name": "todos.db", "kind": "store", "note": "SQLite file" }
  ],
  "folders":   { "app": "✍ 1-line purpose" },                 // key "." = repo root
  "files":     { "app/store.py": { "summary": "✍ 2-3 lines", "hash": "set by stamp" } },
  "functions": { "app/store.py::TodoStore.add": "✍ 2-3 lines" },  // classes and tests too
  "changes": {                  // set by diff; you only write each "reason"
    "base_ref": "c5259b4", "base_note": "merge-base with main",
    "files": [ { "path": "app/store.py", "status": "modified", "reason": "✍ why, 1 line",
                 "added": ["ids"], "modified": ["ids"], "removed": ["qualnames"] } ],
    "commits": [ { "sha": "6bdf2fc", "subject": "...", "files": ["..."] } ]
  },
  "dfd": { "level0": "✍ mermaid flowchart", "level1": "✍ mermaid flowchart", "note": "✍ optional; required if no DFD" },
  "sequences": [ { "title": "✍ cli add", "entry": "app/cli.py::main", "summary": "✍ optional 1 line", "mermaid": "✍ sequenceDiagram" } ],
  "skipped_flows": [ { "name": "✍ cli list", "reason": "✍ same shape as add" } ],
  "tests": {
    "command": "set by import-tests --command",
    "groups": [ { "file": "tests/test_store.py", "summary": "✍ 1-2 lines",
                  "tests": [ { "id": "tests/test_store.py::TestStore.test_add_and_list",
                               "summary": "✍ 1 line", "covers": ["app/store.py::TodoStore.add"] } ] } ],
    "results": {}, "ran_at": null, "report": null      // set by import-tests only
  }
}
```

Tests section: include only test files Claude added or changed (those in `changes.files`). Group by file; one row per test function (inventory ID). Parametrized cases are merged into that one row by `import-tests`; any failing case marks it failed, so say "(3 cases)" in the summary. `covers` = the functions the test actually calls or asserts on (read the test body).

## What verify accepts in diagrams

The checker reads Mermaid line by line. Write diagrams in this plain style so it can check them.

**Sequence diagrams**
- Declare every participant: `participant S as TodoStore`. The label (after `as`) must be a file path, file name, module stem or class from the inventory.
- External things use `actor U as User`, or `participant DB as todos.db`, and must be listed in `externals`.
- Solid arrows (`->>`, `->`, `-x`, `-)`) into a code participant: the text must start with a function name defined in that participant's file/class, e.g. `C->>S: add(title)`. Dotted access is fine: `store.add(...)`.
- Solid arrows into an external: free text (`S->>DB: INSERT todos`).
- Dashed arrows (`-->>`) = returned data: free text.
- `loop/alt/opt/par/else/end/Note/autonumber/activate` are fine.

**Flowcharts (DFDs)**
- Every node label must contain a real file/folder/module/function name (`P1[cli.py<br/>main]`) or be listed in `externals` (`DB[(todos.db)]`, `U[User]`).
- Edge labels are data flows, free text: `P1 -- title --> P2` or `P1 -->|title| P2`.
- Define each node once with its label; reuse the bare ID afterwards.
- Subgraph titles follow the same rule as node labels: `subgraph api [api/]`.

DFD notation: rectangles = external entities, `((...))` circles or boxes = processes, `[(...)]` cylinders = data stores. Level 0 has exactly one process (the system); label it with the top-level package/folder name (e.g. `S((app))`), which verify accepts.

## Writing rules

- File summary: what the file is responsible for, then the key things it contains or calls. 2-3 lines.
- Function summary: what it does with its inputs, side effects (DB writes, network, files, globals), what it returns or raises, notable edge cases. Don't restate the signature; the page shows In/Out from the inventory.
- Class summary: what it represents and what state it owns.
- Test summary: the behaviour checked, not "tests X".
- `Unclear from code` (exact text) when the code doesn't reveal the purpose.
- Plain, short sentences. No em dashes.
