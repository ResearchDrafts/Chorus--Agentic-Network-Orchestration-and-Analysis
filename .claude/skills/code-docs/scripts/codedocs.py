#!/usr/bin/env python3
"""code-docs: deterministic inventory, diff, test import, verification and build.

Every identifier that appears in the generated documentation must come from
inventory.json, which this script derives from the source code. The LLM only
writes prose into docs.json. `verify` enforces that.

Subcommands (run with --root REPO; output dir defaults to REPO/docs/code-docs):
  extract        parse the repo -> inventory.json
  skeleton       create/refresh docs.json keys from inventory (keeps written text)
  diff           compute Claude's changes vs a base ref -> docs.json["changes"]
  import-tests   read a JUnit XML report -> docs.json["tests"]["results"]
  stale          list entries whose source changed since they were written
  stamp          record current file hashes for the summaries you just reviewed
  verify         fail on any unknown identifier, missing summary or stale entry
  build          write data.js (+ index.html), optional --single file for sharing
  render-check   headless-render every tab, report Mermaid errors, screenshot
"""
import argparse, ast, datetime as dt, hashlib, json, os, re, shutil, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
UNCLEAR = "Unclear from code"

LANGS = {".py": "python", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
         ".ts": "js", ".tsx": "js", ".java": "java", ".kt": "java", ".cs": "java",
         ".go": "go", ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c", ".hpp": "c",
         ".cxx": "c", ".rs": "rust"}
SKIP_DIRS = {".git", ".claude", "node_modules", "venv", ".venv", "env", "__pycache__",
             "dist", "build", ".mypy_cache", ".pytest_cache", "target", "vendor",
             ".next", ".idea", ".vscode", "coverage", ".tox", "site-packages"}
KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "else",
            "do", "try", "new", "typeof", "sizeof", "with", "elif", "await", "yield"}


# ---------------------------------------------------------------- helpers
def sh(args, cwd, check=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout

def is_git(root):
    return subprocess.run(["git", "rev-parse", "--git-dir"], cwd=root,
                          capture_output=True).returncode == 0

def load(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))

def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def sha(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]

def is_test_path(rel):
    n = Path(rel).name.lower()
    parts = {p.lower() for p in Path(rel).parts[:-1]}
    return (n.startswith("test_") or n.endswith("_test.py") or n.endswith("_test.go")
            or re.search(r"\.(test|spec)\.[jt]sx?$", n) is not None
            or re.search(r"tests?\.java$", n) is not None
            or bool(parts & {"tests", "test", "__tests__", "spec"}))


# ---------------------------------------------------------------- python parser
def parse_python(src, rel):
    tree = ast.parse(src)
    out, test_file = [], is_test_path(rel)

    def params(fn):
        a, res = fn.args, []
        pos = a.posonlyargs + a.args
        defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        for arg, d in zip(pos, defaults):
            res.append(_p(arg, d))
        if a.vararg:
            res.append(_p(a.vararg, None, "*"))
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            res.append(_p(arg, d))
        if a.kwarg:
            res.append(_p(a.kwarg, None, "**"))
        return res

    def _p(arg, d, prefix=""):
        return {"name": prefix + arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else None,
                "default": ast.unparse(d) if d is not None else None}

    def walk(node, prefix, in_class):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                q = prefix + ch.name
                out.append(dict(name=ch.name, qualname=q, kind="class", line=ch.lineno,
                                end=ch.end_lineno, params=[], returns=None,
                                bases=[ast.unparse(b) for b in ch.bases],
                                doc=_doc(ch), is_test=test_file and ch.name.startswith("Test")))
                walk(ch, q + ".", True)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
                is_t = test_file and ch.name.startswith("test")
                out.append(dict(name=ch.name, qualname=q,
                                kind="test" if is_t else ("method" if in_class else "function"),
                                line=ch.lineno, end=ch.end_lineno, params=params(ch),
                                returns=ast.unparse(ch.returns) if ch.returns else None,
                                is_async=isinstance(ch, ast.AsyncFunctionDef),
                                decorators=[ast.unparse(d) for d in ch.decorator_list],
                                doc=_doc(ch), is_test=is_t))
                walk(ch, q + ".", False)  # nested defs
    walk(tree, "", False)
    return out

def _doc(node):
    d = ast.get_docstring(node)
    return d.strip().splitlines()[0][:200] if d else None


# ---------------------------------------------------------------- brace-language parser
def blank_strings_comments(src, lang):
    """Replace string/comment bodies with spaces, keeping newlines, so braces can be matched."""
    out, i, n = list(src), 0, len(src)
    quotes = "'\"`" if lang == "js" else ("\"`" if lang == "go" else "\"'")
    while i < n:
        c, nx = src[i], src[i + 1] if i + 1 < n else ""
        if c == "/" and nx == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j): out[k] = " "
            i = j
        elif c == "/" and nx == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if src[k] != "\n": out[k] = " "
            i = j
        elif c in quotes:
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\": j += 1
                elif src[j] == "\n" and c != "`": break
                j += 1
            for k in range(i + 1, min(j, n)):
                if src[k] != "\n": out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)

def split_params(s):
    s = s.strip()
    if not s:
        return []
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<([{": depth += 1
        if ch in ">)]}": depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]

def param_obj(p, lang):
    default = None
    if "=" in p and lang in ("js", "c"):
        p, default = [x.strip() for x in p.split("=", 1)]
    if lang == "js":
        if ":" in p:
            nm, ty = [x.strip() for x in p.split(":", 1)]
            return {"name": nm, "type": ty, "default": default}
        return {"name": p, "type": None, "default": default}
    if lang == "go":
        bits = p.split(None, 1)
        return {"name": bits[0], "type": bits[1] if len(bits) > 1 else None, "default": None}
    if lang == "rust":
        if ":" in p:
            nm, ty = [x.strip() for x in p.split(":", 1)]
            return {"name": nm, "type": ty, "default": None}
        return {"name": p, "type": None, "default": None}
    m = re.match(r"(.*?)([A-Za-z_]\w*)\s*(\[\])?$", p)  # java / c: "Type name"
    if m and m.group(1).strip():
        return {"name": m.group(2), "type": (m.group(1) + (m.group(3) or "")).strip(), "default": default}
    return {"name": p, "type": None, "default": default}

PATTERNS = {
    "js": [
        ("class", re.compile(r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", re.M)),
        ("function", re.compile(r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?::\s*([^{=]+?))?\s*\{", re.M)),
        ("function", re.compile(r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\s*)?\(([^)]*)\)\s*(?::\s*([^=]+?))?\s*(?:=>)?\s*\{", re.M)),
        ("method", re.compile(r"^[ \t]+(?:public\s+|private\s+|protected\s+|static\s+|async\s+|readonly\s+|override\s+|get\s+|set\s+)*([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?::\s*([^{]+?))?\s*\{", re.M)),
    ],
    "java": [
        ("class", re.compile(r"^[ \t]*(?:public\s+|private\s+|protected\s+|abstract\s+|final\s+|static\s+|sealed\s+|data\s+|open\s+)*(?:class|interface|enum|record|object)\s+([A-Za-z_]\w*)", re.M)),
        ("method", re.compile(r"^[ \t]*(?:@\w+(?:\([^)]*\))?\s+)*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|abstract\s+|synchronized\s+|override\s+|async\s+|virtual\s+)*([\w<>\[\],.? ]+?)\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:throws\s+[\w.,\s]+)?\s*\{", re.M)),
    ],
    "go": [
        ("class", re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b", re.M)),
        ("function", re.compile(r"^func\s+(?:\(\s*\w*\s*\*?\s*([A-Za-z_]\w*)[^)]*\)\s*)?([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\(([^)]*)\)\s*([^{]*)\{", re.M)),
    ],
    "c": [
        ("class", re.compile(r"^[ \t]*(?:class|struct)\s+([A-Za-z_]\w*)[^;{]*\{", re.M)),
        ("function", re.compile(r"^[ \t]*(?:template\s*<[^>]*>\s*)?((?:[\w:*&<>,]+[ \t]+)+?[*&]*)\s*((?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{", re.M)),
    ],
    "rust": [
        ("class", re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.M)),
        ("impl", re.compile(r"^[ \t]*impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?([A-Za-z_]\w*)", re.M)),
        ("function", re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^{]+?))?\s*(?:where[^{]*)?\{", re.M)),
    ],
}

def match_brace(clean, start):
    i = clean.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(clean)):
        if clean[j] == "{": depth += 1
        elif clean[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return len(clean) - 1

def parse_braced(src, rel, lang):
    clean = blank_strings_comments(src, lang)
    line_of = lambda pos: clean.count("\n", 0, pos) + 1
    test_file = is_test_path(rel)
    raw = []
    for kind, rx in PATTERNS[lang]:
        for m in rx.finditer(clean):
            g = m.groups()
            if lang == "java" and kind == "method":
                ret, name, ps = g[0].strip(), g[1], g[2]
                if not ret or ret.split()[-1] in KEYWORDS or ret.split()[0] in ("new", "return", "else", "throw"):
                    continue
            elif lang == "go" and kind == "function":
                recv, name, ps, ret = g
                ret = ret.strip() or None
            elif lang == "c" and kind == "function":
                ret, name, ps = g[0].strip(), g[1], g[2]
            elif kind in ("class", "impl"):
                name, ps, ret = g[0], "", None
            else:
                name, ps = g[0], g[1]
                ret = (g[2] or "").strip() or None if len(g) > 2 else None
            if name in KEYWORDS:
                continue
            end_pos = match_brace(clean, m.end() - 1 if clean[m.end() - 1] == "{" else m.start())
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
            item = dict(name=name.split("::")[-1], kind=kind, line=line_of(start),
                        end=line_of(end_pos) if end_pos else line_of(start),
                        params=[param_obj(p, lang) for p in split_params(ps)] if kind not in ("class", "impl") else [],
                        returns=ret if kind not in ("class", "impl") else None, doc=None)
            if lang == "go" and kind == "function" and g[0]:
                item["receiver"] = g[0]
            if lang == "c" and "::" in name:
                item["receiver"] = name.split("::")[-2]
            raw.append(item)
    # de-duplicate (same line), method pattern also matches top-level functions
    by_line = {}
    for it in raw:
        prev = by_line.get(it["line"])
        if prev is None or (prev["kind"] == "method" and it["kind"] != "method"):
            by_line[it["line"]] = it
    items = sorted(by_line.values(), key=lambda x: x["line"])
    classes = [c for c in items if c["kind"] in ("class", "impl")]
    out = []
    for it in items:
        if it["kind"] == "impl":
            continue  # Rust impl blocks only give methods their owner
        owner = it.get("receiver")
        if not owner and it["kind"] not in ("class",):
            enclosing = [c for c in classes if c["line"] < it["line"] <= c["end"]]
            if enclosing:
                owner = max(enclosing, key=lambda c: c["line"])["name"]
        if it["kind"] == "method" and not owner:
            continue  # "method" regex outside any class is almost always a call, skip
        if it["kind"] != "class":
            it["kind"] = "method" if owner else "function"
        it["qualname"] = f"{owner}.{it['name']}" if owner and it["kind"] != "class" else it["name"]
        is_t = test_file and it["kind"] != "class" and (
            it["name"].startswith("test") or it["name"].startswith("Test"))
        it["is_test"] = is_t
        if is_t: it["kind"] = "test"
        it.pop("receiver", None)
        out.append(it)
    if lang == "js" and test_file:  # it("...") / test("...") blocks
        for m in re.finditer(r"^[ \t]*(?:it|test)(?:\.only|\.skip)?\s*\(\s*(['\"`])(.+?)\1", src, re.M):
            ln = src.count("\n", 0, m.start()) + 1
            end_pos = match_brace(clean, m.end())
            out.append(dict(name=m.group(2), qualname=m.group(2), kind="test", line=ln,
                            end=line_of(end_pos) if end_pos else ln, params=[], returns=None,
                            doc=None, is_test=True))
    if lang == "java" and test_file:  # @Test annotation marks tests regardless of name
        for it in out:
            if it["kind"] in ("method", "function"):
                head = src.splitlines()[max(0, it["line"] - 3): it["line"]]
                if any("@Test" in h for h in head):
                    it["kind"], it["is_test"] = "test", True
    return out


def parse_file(src, rel, lang):
    if lang == "python":
        return parse_python(src, rel), "ast"
    return parse_braced(src, rel, lang), "regex"


# ---------------------------------------------------------------- extract
def list_files(root, out_dir):
    root = Path(root)
    if is_git(root):
        names = sh(["git", "ls-files", "--cached", "--others", "--exclude-standard"], root).splitlines()
        files = [n for n in names if (root / n).is_file()]
    else:
        files = []
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for f in fns:
                files.append(str(Path(dp, f).relative_to(root)))
    outrel = None
    try:
        outrel = Path(out_dir).resolve().relative_to(root.resolve())
    except ValueError:
        pass
    keep = []
    for f in files:
        p = Path(f)
        if any(part in SKIP_DIRS for part in p.parts[:-1]):
            continue
        if outrel and str(outrel) != "." and (p == outrel or outrel in p.parents):
            continue
        keep.append(p.as_posix())
    return sorted(keep)

def cmd_extract(a):
    root, out = Path(a.root).resolve(), Path(a.out)
    inv = {"root": str(root), "generated": dt.datetime.now().isoformat(timespec="seconds"),
           "git_head": sh(["git", "rev-parse", "--short", "HEAD"], root, False).strip() if is_git(root) else None,
           "folders": {}, "files": {}, "symbols": {}, "parse_errors": {}}
    for rel in list_files(root, out):
        p = root / rel
        try:
            src = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary
        lang = LANGS.get(p.suffix.lower())
        folder = Path(rel).parent.as_posix()
        for f in [folder] + [x.as_posix() for x in Path(folder).parents]:
            inv["folders"].setdefault(f, {"files": 0})
        inv["folders"][folder]["files"] += 1
        inv["files"][rel] = {"lang": lang or "other", "lines": src.count("\n") + 1,
                             "hash": sha(src), "source": bool(lang),
                             "is_test": bool(lang) and is_test_path(rel), "parser": None}
        if not lang:
            continue
        try:
            syms, parser = parse_file(src, rel, lang)
        except Exception as e:  # never let one odd file stop the run
            inv["parse_errors"][rel] = f"{type(e).__name__}: {e}"
            continue
        inv["files"][rel]["parser"] = parser
        for s in syms:
            sid = f"{rel}::{s['qualname']}"
            if sid in inv["symbols"]:
                sid = f"{sid}@L{s['line']}"  # overloads / redefinitions
            s["file"] = rel
            s["public"] = not s["name"].startswith("_")
            inv["symbols"][sid] = s
    save(out / "inventory.json", inv)
    n_src = sum(1 for f in inv["files"].values() if f["source"])
    print(f"inventory: {len(inv['folders'])} folders, {len(inv['files'])} files "
          f"({n_src} source), {len(inv['symbols'])} symbols -> {out/'inventory.json'}")
    if inv["parse_errors"]:
        print("parse errors (document these files by hand, carefully):")
        for k, v in inv["parse_errors"].items(): print(f"  {k}: {v}")


# ---------------------------------------------------------------- skeleton
EMPTY_DOCS = {
    "project": {"name": "", "summary": "", "repo_url": "", "base_ref": ""},
    "externals": [],
    "folders": {}, "files": {}, "functions": {},
    "changes": {"base_ref": "", "files": [], "commits": []},
    "dfd": {"level0": "", "level1": "", "note": ""},
    "sequences": [], "skipped_flows": [],
    "tests": {"command": "", "groups": [], "results": {}, "ran_at": None, "report": None},
}

def cmd_skeleton(a):
    out = Path(a.out)
    inv = load(out / "inventory.json")
    docs = load(out / "docs.json", json.loads(json.dumps(EMPTY_DOCS)))
    for k, v in EMPTY_DOCS.items():
        docs.setdefault(k, json.loads(json.dumps(v)))
    if not docs["project"]["name"]:
        docs["project"]["name"] = Path(inv["root"]).name
    orphans = docs.setdefault("_orphaned", {})
    def sync(section, keys, blank):
        cur = docs[section]
        for k in list(cur):
            if k not in keys:
                orphans[f"{section}:{k}"] = cur.pop(k)
        for k in keys:
            cur.setdefault(k, json.loads(json.dumps(blank)))
    sync("folders", list(inv["folders"]), "")
    sync("files", [f for f, m in inv["files"].items() if m["source"]], {"summary": "", "hash": None})
    sync("functions", list(inv["symbols"]), "")
    if not orphans:
        docs.pop("_orphaned")
    save(out / "docs.json", docs)
    todo = sum(1 for v in docs["folders"].values() if not v) + \
           sum(1 for v in docs["files"].values() if not v["summary"]) + \
           sum(1 for v in docs["functions"].values() if not v)
    print(f"docs.json ready: {todo} empty summaries to fill"
          + (f"; {len(orphans)} orphaned entries moved to _orphaned (no longer in code)" if orphans else ""))


# ---------------------------------------------------------------- diff
def auto_base(root):
    for ref in ("origin/main", "origin/master", "main", "master"):
        r = subprocess.run(["git", "merge-base", "HEAD", ref], cwd=root, capture_output=True, text=True)
        if r.returncode == 0:
            mb = r.stdout.strip()
            head = sh(["git", "rev-parse", "HEAD"], root).strip()
            if mb != head:
                return mb[:10], f"merge-base with {ref}"
    return "HEAD", "HEAD (only uncommitted changes; pass --base to widen)"

def hunk_lines(root, base, rel):
    """New-side line numbers touched by the diff of one file."""
    txt = sh(["git", "diff", "--no-renames", "-U0", base, "--", rel], root, False)
    lines = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", txt, re.M):
        start, cnt = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + max(cnt, 1)))
    return lines

def old_symbols(root, base, rel):
    lang = LANGS.get(Path(rel).suffix.lower())
    if not lang:
        return {}
    r = subprocess.run(["git", "show", f"{base}:{rel}"], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    try:
        syms, _ = parse_file(r.stdout, rel, lang)
    except SyntaxError:
        return {}
    return {s["qualname"]: s for s in syms}

def cmd_diff(a):
    root, out = Path(a.root).resolve(), Path(a.out)
    if not is_git(root):
        sys.exit("not a git repository: the Changes tab needs git history")
    inv = load(out / "inventory.json")
    docs = load(out / "docs.json")
    base, why = (a.base, "given") if a.base else auto_base(root)
    print(f"base ref: {base} ({why})")
    status = {}
    for line in sh(["git", "diff", "--no-renames", "--name-status", base], root).splitlines():
        st, rel = line.split("\t", 1)
        status[rel] = {"A": "added", "D": "deleted"}.get(st[0], "modified")
    for rel in sh(["git", "ls-files", "--others", "--exclude-standard"], root).splitlines():
        status[rel] = "added"
    prev = {f["path"]: f for f in docs["changes"].get("files", [])}
    files = []
    for rel, st in sorted(status.items()):
        if rel not in inv["files"] and st != "deleted":
            continue  # ignored / output dir
        cur = {sid: s for sid, s in inv["symbols"].items() if s["file"] == rel}
        old = old_symbols(root, base, rel) if st != "added" else {}
        added, modified, removed = [], [], []
        if st == "added":
            added = list(cur)
        elif st == "deleted":
            removed = list(old)
        else:
            touched = hunk_lines(root, base, rel)
            for sid, s in cur.items():
                if s["qualname"] not in old:
                    added.append(sid)
                elif touched & set(range(s["line"], s["end"] + 1)):
                    modified.append(sid)
            # keep only the innermost symbol when a class and its method both match
            modified = [m for m in modified if not any(
                o != m and o.startswith(m + ".") for o in modified)]
            removed = [q for q in old if f"{rel}::{q}" not in cur and
                       not any(s["qualname"] == q for s in cur.values())]
        p = prev.get(rel, {})
        files.append({"path": rel, "status": st, "reason": p.get("reason", ""),
                      "added": added, "modified": modified, "removed": removed})
    commits = []
    if base != "HEAD":
        for line in sh(["git", "log", "--format=%h%x09%s", f"{base}..HEAD"], root).splitlines():
            h, subj = line.split("\t", 1)
            names = sh(["git", "show", "--no-renames", "--name-only", "--format=", h], root).split()
            commits.append({"sha": h, "subject": subj, "files": names})
    dirty = [l[3:] for l in sh(["git", "status", "--porcelain", "-uall"], root).splitlines() if len(l) > 3]
    dirty = sorted(f for f in dirty if f in inv["files"] or status.get(f) == "deleted")
    if dirty:
        commits.append({"sha": "working tree", "subject": "Uncommitted changes", "files": dirty})
    docs["changes"] = {"base_ref": base, "base_note": why, "files": files, "commits": commits}
    docs["project"]["base_ref"] = base
    save(out / "docs.json", docs)
    c = {k: sum(1 for f in files if f["status"] == k) for k in ("added", "modified", "deleted")}
    print(f"changes: {c} ; write a one-line 'reason' for each file in docs.json")


# ---------------------------------------------------------------- tests
def cmd_import_tests(a):
    out = Path(a.out)
    inv, docs = load(out / "inventory.json"), load(out / "docs.json")
    tests = {sid: s for sid, s in inv["symbols"].items() if s["is_test"] and s["kind"] == "test"}
    tree = ET.parse(a.junit)
    results, unmatched = {}, []
    rank = {"fail": 3, "error": 3, "skip": 1, "pass": 0}
    for tc in tree.iter("testcase"):
        name = re.sub(r"\[.*\]$", "", tc.get("name", ""))  # pytest parametrize ids
        cls = tc.get("classname", "") or ""
        st = "pass"
        msg = None
        for tag, s in (("failure", "fail"), ("error", "error"), ("skipped", "skip")):
            el = tc.find(tag)
            if el is not None:
                st, msg = s, (el.get("message") or (el.text or "")).strip()[:300]
        cand = [sid for sid, s in tests.items() if s["name"] == name or s["qualname"] == name]
        if len(cand) > 1:  # disambiguate by classname -> module path / class
            mod = cls.replace(".", "/")
            narrowed = [sid for sid in cand if mod and (
                Path(tests[sid]["file"]).with_suffix("").as_posix() in mod
                or mod.endswith(tests[sid]["qualname"].rsplit(".", 1)[0]))]
            cand = narrowed or cand
        if len(cand) != 1:
            unmatched.append(f"{cls}::{name}")
            continue
        sid = cand[0]
        old = results.get(sid)
        if not old or rank[st] > rank[old["status"]]:
            results[sid] = {"status": st, "time": tc.get("time"), "message": msg}
    t = docs["tests"]
    t["results"], t["report"] = results, str(a.junit)
    t["ran_at"] = dt.datetime.now().isoformat(timespec="seconds")
    if a.command:
        t["command"] = a.command
    save(out / "docs.json", docs)
    cnt = {}
    for r in results.values(): cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    n_cases = sum(1 for _ in tree.iter("testcase"))
    print(f"imported {len(results)} test results {cnt} from {n_cases} report cases"
          + (" (parametrized cases merged per test function; any failing case fails it)" if n_cases > len(results) + len(unmatched) else ""))
    if unmatched:
        print(f"{len(unmatched)} report entries matched no inventory test (shown as unmatched):")
        for u in unmatched[:20]: print("  ", u)
        t["unmatched"] = unmatched
        save(out / "docs.json", docs)


# ---------------------------------------------------------------- stale / stamp
def stale_entries(inv, docs):
    stale = []
    for rel, meta in inv["files"].items():
        d = docs["files"].get(rel)
        if meta["source"] and d and d.get("summary") and d.get("hash") and d["hash"] != meta["hash"]:
            stale.append(rel)
    return stale

def cmd_stale(a):
    out = Path(a.out)
    inv, docs = load(out / "inventory.json"), load(out / "docs.json")
    st = stale_entries(inv, docs)
    if not st:
        print("nothing stale"); return
    print("source changed since these summaries were written; re-read and update them, then `stamp`:")
    for rel in st:
        syms = [sid for sid, s in inv["symbols"].items() if s["file"] == rel]
        print(f"  {rel}  ({len(syms)} symbols)")

def cmd_stamp(a):
    out = Path(a.out)
    inv, docs = load(out / "inventory.json"), load(out / "docs.json")
    targets = a.files or [r for r, d in docs["files"].items() if d.get("summary")]
    n = 0
    for rel in targets:
        if rel in docs["files"] and rel in inv["files"] and docs["files"][rel].get("summary"):
            docs["files"][rel]["hash"] = inv["files"][rel]["hash"]; n += 1
    save(out / "docs.json", docs)
    print(f"stamped {n} files")


# ---------------------------------------------------------------- verify
CTRL = re.compile(r"^\s*(autonumber|loop|alt|else|opt|par|and|critical|break|rect|end|"
                  r"activate|deactivate|box|note|Note|title|%%|link|links|create|destroy)\b")
SEQ_DECL = re.compile(r"^\s*(participant|actor)\s+(\S+?)(?:\s+as\s+(.+))?\s*$")
SEQ_MSG = re.compile(r"^\s*([^\s\-<>]+?)\s*(-{1,2}(?:>>|>|x|\)))\s*[+-]?\s*([^\s:]+)\s*:\s*(.*)$")
FLOW_NODE = re.compile(r"(?<![\w-])([A-Za-z_][\w]*)\s*(\[\[|\[\(|\(\(|\(\[|\{\{|\[|\(|\{|>)\s*\"?(.*?)\"?\s*(\]\]|\)\]|\)\)|\]\)|\}\}|\]|\)|\})")

class Refs:
    """Every name a diagram may legitimately mention, derived from the inventory."""
    def __init__(self, inv, docs):
        self.files = set(inv["files"])
        self.basenames = {Path(f).name for f in self.files}
        self.stems = {Path(f).stem for f in inv["files"] if inv["files"][f]["source"]}
        self.folders = {f for f in inv["folders"] if f != "."} | {Path(f).name for f in inv["folders"] if f != "."}
        self.sym_names = {s["name"] for s in inv["symbols"].values()} | {s["qualname"] for s in inv["symbols"].values()}
        self.classes = {s["name"] for s in inv["symbols"].values() if s["kind"] == "class"}
        self.externals = {e["name"].lower() for e in docs.get("externals", [])}
        self.inv = inv
    def token_ok(self, t):
        t = t.strip("`'\"()[]{}.,;:")
        return bool(t) and (t in self.files or t in self.basenames or t in self.stems
                            or t in self.folders or t.rstrip("/") in self.folders or t in self.sym_names)
    def label_ok(self, label):
        lab = re.sub(r"<br\s*/?>", " ", label)
        if lab.strip().lower() in self.externals:
            return True
        return any(self.token_ok(t) for t in re.split(r"[\s,/]+(?![^\s]*\.\w)", lab)) or \
               any(self.token_ok(t) for t in re.split(r"[\s,]+", lab))
    def symbols_in(self, label):
        """Symbol names defined in the file/class/module a participant label points at."""
        lab = label.strip().strip('"')
        out = set()
        for s in self.inv["symbols"].values():
            f = s["file"]
            if lab in (f, Path(f).name, Path(f).stem) or s["qualname"].split(".")[0] == lab:
                out.add(s["name"])
        return out

def check_sequence(code, refs, where, errs):
    parts = {}
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("sequenceDiagram") or line.startswith("%%"):
            continue
        m = SEQ_DECL.match(line)
        if m:
            kind, pid, label = m.group(1), m.group(2), (m.group(3) or m.group(2)).strip().strip('"')
            label_plain = re.sub(r"<br\s*/?>.*", "", label).strip()
            if kind == "actor" or label_plain.lower() in refs.externals or pid.lower() in refs.externals:
                if label_plain.lower() not in refs.externals and pid.lower() not in refs.externals:
                    errs.append(f"{where}: actor '{label}' is not listed in docs.externals")
                parts[pid] = None
            elif not refs.label_ok(label_plain):
                errs.append(f"{where}: participant '{label}' is not a file, module or class in the inventory")
                parts[pid] = set()
            else:
                parts[pid] = refs.symbols_in(label_plain) or None
            continue
        if CTRL.match(line):
            continue
        m = SEQ_MSG.match(line)
        if not m:
            continue
        src, arrow, dst, text = m.groups()
        if src not in parts or dst not in parts:
            errs.append(f"{where}: '{src if src not in parts else dst}' used without a participant/actor declaration")
            continue
        if arrow.startswith("--"):
            continue  # dashed = return value / data, free text allowed
        if parts[dst] is None:
            continue  # message into an external (DB, user, API): free text allowed
        fn = re.match(r"\s*`?([A-Za-z_$][\w$.]*)", text)
        name = fn.group(1).split(".")[-1] if fn else None
        if not name or name not in refs.sym_names:
            errs.append(f"{where}: message '{text.strip()}' does not start with a known function name")
        elif parts[dst] and name not in parts[dst]:
            errs.append(f"{where}: '{name}' is not defined in participant '{dst}'")

def check_flow(code, refs, where, errs):
    for raw in code.splitlines():
        line = raw.split("%%")[0]
        if re.match(r"\s*(flowchart|graph|classDef|class|style|linkStyle|subgraph|end|direction)\b", line):
            m = re.match(r"\s*subgraph\s+\w+\s*\[\s*\"?(.*?)\"?\s*\]", line)
            if m and not refs.label_ok(m.group(1)):
                errs.append(f"{where}: subgraph '{m.group(1)}' is not a folder/file/module in the inventory or docs.externals")
            continue
        line = re.sub(r"\|[^|]*\|", " ", line)          # edge labels = data flows, free text
        line = re.sub(r"--\s*[^->]+?\s*-->", "-->", line)
        for m in FLOW_NODE.finditer(line):
            label = m.group(3)
            if not refs.label_ok(label):
                errs.append(f"{where}: node '{label}' is not a file/folder/module/function in the inventory "
                            f"and not in docs.externals")

def cmd_verify(a):
    out = Path(a.out)
    inv, docs = load(out / "inventory.json"), load(out / "docs.json")
    errs, warns = [], []
    syms, files = inv["symbols"], inv["files"]
    for k in docs["folders"]:
        if k not in inv["folders"]: errs.append(f"folders: '{k}' not in inventory")
    for k in docs["files"]:
        if k not in files: errs.append(f"files: '{k}' not in inventory")
    for k in docs["functions"]:
        if k not in syms: errs.append(f"functions: '{k}' not in inventory")
    if docs.get("_orphaned"):
        warns.append(f"{len(docs['_orphaned'])} orphaned entries in docs.json (code was removed); delete them")
    req_all = not a.changed_only
    changed_files = {f["path"] for f in docs["changes"].get("files", [])}
    for k in inv["folders"]:
        if not docs["folders"].get(k): errs.append(f"missing folder purpose: '{k}'")
    for k, m in files.items():
        if m["source"] and (req_all or k in changed_files) and not docs["files"].get(k, {}).get("summary"):
            errs.append(f"missing file summary: '{k}'")
    for k, s in syms.items():
        if (req_all or s["file"] in changed_files) and not docs["functions"].get(k):
            errs.append(f"missing function summary: '{k}'")
    for rel in stale_entries(inv, docs):
        errs.append(f"stale: '{rel}' changed after its summary was written (update, then `stamp`)")
    for rel, d in docs["files"].items():
        if d.get("summary") and not d.get("hash"):
            errs.append(f"unstamped: '{rel}' has a summary but no hash (run `stamp` after reviewing)")
    for f in docs["changes"].get("files", []):
        if f["status"] != "deleted" and f["path"] not in files:
            errs.append(f"changes: '{f['path']}' not in inventory")
        for sid in f.get("added", []) + f.get("modified", []):
            if sid not in syms: errs.append(f"changes: symbol '{sid}' not in inventory (re-run diff)")
        if not f.get("reason"): errs.append(f"changes: no reason for '{f['path']}'")
    t = docs["tests"]
    for g in t.get("groups", []):
        if g.get("file") not in files: errs.append(f"tests: group file '{g.get('file')}' not in inventory")
        for it in g.get("tests", []):
            if it.get("id") not in syms or not syms[it["id"]]["is_test"]:
                errs.append(f"tests: '{it.get('id')}' is not a test in the inventory")
            for c in it.get("covers", []):
                if c not in syms: errs.append(f"tests: covers '{c}' not in inventory")
            if "result" in it or "status" in it:
                errs.append(f"tests: '{it.get('id')}' has a hand-written result; results come only from import-tests")
            if not it.get("summary"): errs.append(f"tests: no summary for '{it.get('id')}'")
    if t.get("groups") and not t.get("ran_at"):
        warns.append("tests were not run/imported; they will show as 'not run'")
    if t.get("report") and not Path(t["report"]).exists():
        warns.append(f"test report {t['report']} no longer exists")
    refs = Refs(inv, docs)
    for lvl in ("level0", "level1"):
        if docs["dfd"].get(lvl):
            check_flow(docs["dfd"][lvl], refs, f"dfd.{lvl}", errs)
    for i, s in enumerate(docs.get("sequences", [])):
        if s.get("entry") and s["entry"] not in syms:
            errs.append(f"sequences[{i}]: entry '{s['entry']}' not in inventory")
        code = s.get("mermaid", "")
        (check_sequence if code.lstrip().startswith("sequenceDiagram") else check_flow)(
            code, refs, f"sequences[{i}] '{s.get('title','')}'", errs)
    if not docs["dfd"].get("level0") and not docs["dfd"].get("note"):
        warns.append("no DFD and no note explaining why")
    text = json.dumps({k: v for k, v in docs.items() if k not in ("tests",)}, ensure_ascii=False)
    if "—" in text:
        warns.append("docs.json contains em dashes; replace them")
    unclear = sum(1 for v in docs["functions"].values() if v == UNCLEAR) + \
              sum(1 for v in docs["files"].values() if v.get("summary") == UNCLEAR)
    for w in warns: print("WARN ", w)
    for e in errs[:200]: print("ERROR", e)
    if len(errs) > 200: print(f"... {len(errs) - 200} more")
    print(f"\nverify: {len(errs)} errors, {len(warns)} warnings, {unclear} entries marked '{UNCLEAR}'")
    sys.exit(1 if errs else 0)


# ---------------------------------------------------------------- build
def cmd_build(a):
    out = Path(a.out)
    inv, docs = load(out / "inventory.json"), load(out / "docs.json")
    payload = dict(docs); payload["inventory"] = inv
    data = "window.CODEDOCS = " + json.dumps(payload, ensure_ascii=False).replace("</", "<\\/") + ";\n"
    (out / "data.js").write_text(data, encoding="utf-8")
    shutil.copyfile(ASSETS / "index.html", out / "index.html")
    print(f"wrote {out/'index.html'} + data.js (open index.html directly in a browser)")
    if a.single:
        html = (ASSETS / "index.html").read_text(encoding="utf-8")
        html = html.replace('<script src="data.js"></script>', "<script>\n" + data + "</script>")
        Path(a.single).write_text(html, encoding="utf-8")
        print(f"wrote single-file copy {a.single} (for publishing/sharing)")


# ---------------------------------------------------------------- render-check
def cmd_render_check(a):
    out = Path(a.out).resolve()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed: pip install playwright && playwright install chromium")
    shots = out / "screenshots"; shots.mkdir(exist_ok=True)
    bad = 0
    with sync_playwright() as p:
        kw = {"executable_path": a.chrome} if a.chrome else {}
        b = p.chromium.launch(**kw)
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: logs.append(str(e)))
        pg.goto((out / "index.html").as_uri())
        pg.wait_for_function("window.CODEDOCS_READY === true", timeout=30000)
        for tab in pg.eval_on_selector_all("nav button[data-tab]", "els => els.map(e => e.dataset.tab)"):
            pg.click(f"nav button[data-tab='{tab}']")
            pg.wait_for_function("window.CODEDOCS_PENDING === 0", timeout=30000)
            errs = pg.eval_on_selector_all(f"#tab-{tab} .mm-error", "els => els.map(e => e.textContent)")
            n_svg = pg.eval_on_selector_all(f"#tab-{tab} .mm svg", "els => els.length")
            pg.screenshot(path=str(shots / f"{tab}.png"), full_page=True)
            print(f"tab {tab}: {n_svg} diagrams rendered, {len(errs)} errors")
            for e in errs: print("   MERMAID ERROR:", e[:300])
            bad += len(errs)
        b.close()
    logs = [l for l in logs if "Failed to load resource" not in l]  # CDN miss is fine if fallback loaded
    for l in logs: print("CONSOLE:", l[:300])
    print(f"screenshots in {shots}")
    sys.exit(1 if bad or logs else 0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--out", default=None, help="output dir (default ROOT/docs/code-docs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract"); sub.add_parser("skeleton"); sub.add_parser("stale")
    d = sub.add_parser("diff"); d.add_argument("--base")
    t = sub.add_parser("import-tests"); t.add_argument("junit"); t.add_argument("--command")
    s = sub.add_parser("stamp"); s.add_argument("files", nargs="*")
    v = sub.add_parser("verify"); v.add_argument("--changed-only", action="store_true",
                                                 help="require summaries only for changed files (huge repos)")
    b = sub.add_parser("build"); b.add_argument("--single", help="also write one self-contained html here")
    r = sub.add_parser("render-check"); r.add_argument("--chrome", help="path to a Chrome/Chromium binary")
    a = ap.parse_args()
    a.out = a.out or str(Path(a.root) / "docs" / "code-docs")
    {"extract": cmd_extract, "skeleton": cmd_skeleton, "diff": cmd_diff, "stale": cmd_stale,
     "import-tests": cmd_import_tests, "stamp": cmd_stamp, "verify": cmd_verify,
     "build": cmd_build, "render-check": cmd_render_check}[a.cmd](a)

if __name__ == "__main__":
    main()
