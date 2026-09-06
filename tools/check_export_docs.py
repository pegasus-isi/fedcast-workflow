#!/usr/bin/env python3

"""Check that every field a silo-resident job may export is documented.

Cross-silo mode's claim is about what leaves a data holder. Enforcement
lives at run time, in three layers inside `fedcast_common`:

1. `write_export()` validates a payload against its declaration and
   writes it in the same call, so no object can be validated and a
   different one written, and nothing is created when validation fails.
2. It then reads the file back and validates that, so a serializer that
   reshapes the payload cannot widen the surface.
3. `guard_export()` registers an output path up front and re-validates
   the file at interpreter exit. This is the layer that matters most:
   it inspects the artifact, so it holds no matter what wrote it — the
   guarded call, an aliased `json.dump`, a hand-rolled `write()`, or a
   guarded call that turned out to be dead code.

What this script adds is narrower, and worth stating plainly because an
earlier version of it overclaimed:

* every name in each wrapper's `EXPORT_FIELDS` appears in the body rows
  of README's "What actually leaves a silo" **table**, matched exactly.
  Full path, so a nested field is documented as `splits.train` rather
  than `train` — matching the leaf alone would let an undocumented
  `private.site` pass on the strength of a documented `site`. Table rows
  only, because the prose around the table names fields while explaining
  the rules, and a field mentioned there but dropped from the table is
  not documented in the sense that matters;
* each payload is registered with `guard_export` and written with
  `write_export`, both passing `EXPORT_FIELDS`, with the payload
  argument and the label in agreement.

It does NOT prove that no other write path exists. That is not decidable
by reading Python: a write can be aliased, computed, or delegated. The
`json.dump` detection below is a lint that catches the obvious cases, not
a proof — layer 3 is what makes the absence of undeclared fields in the
artifact an actual guarantee.

    tools/check_export_docs.py            # 0 = documented and guarded

An earlier version of this tool also tried to derive the export surface
by analysing how each payload was constructed. That cannot be completed
in a dynamic language either; successive reviews found an unresolvable
name, `dict(...)` instead of a literal, then mutation after the literal.
Declaring the surface and checking the artifact is what holds.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Wrappers that build a payload leaving a silo, and the label each passes
# to check_export. The label is only used to report which call is missing.
WRAPPERS = [
    ("bin/fl_train_client.py", ["meta"],
     "per-round metadata to the aggregator"),
    ("bin/fl_validate_client.py", ["metrics"],
     "per-client validation metrics to the server"),
    ("bin/preprocess_sequences.py", ["manifest", "error_manifest"],
     "the split manifest, and its no-usable-input stand-in"),
]

# Payloads that share an output path with another payload in the same
# wrapper, so one guard_export registration covers both.
_SHARES_PATH_WITH = {"error_manifest"}

DOC_FILE = "README.md"
SECTION_MARKER = "**What actually leaves a silo, in full.**"


def declared_fields(path):
    """The EXPORT_FIELDS string tuple of a wrapper."""
    tree = ast.parse((ROOT / path).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "EXPORT_FIELDS"
                   for t in node.targets):
            continue
        value = node.value
        if not isinstance(value, (ast.Tuple, ast.List)):
            raise SystemExit(
                f"{path}: EXPORT_FIELDS must be a literal tuple or list of "
                f"strings so it can be read without executing anything."
            )
        fields = [e.value for e in value.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if len(fields) != len(value.elts):
            raise SystemExit(
                f"{path}: EXPORT_FIELDS contains a non-string entry."
            )
        return fields, tree

    raise SystemExit(
        f"{path}: no EXPORT_FIELDS declaration. Every wrapper that writes a "
        f"payload off a silo must declare the fields it may export."
    )


def guarded_payloads(tree):
    """Payload names correctly written through write_export().

    A call counts only when its payload argument is the name the label
    claims and the declaration passed is EXPORT_FIELDS — so validating one
    object while writing another does not satisfy the gate.
    """
    guarded = set()
    mismatched = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else \
            (fn.id if isinstance(fn, ast.Name) else None)
        if name != "write_export":
            continue

        # write_export(path, payload, declared, label, ...)
        if len(node.args) < 4:
            mismatched.append(
                f"write_export() call with {len(node.args)} positional "
                f"args; expected (path, payload, declared, label)")
            continue
        payload, declared, label = node.args[1], node.args[2], node.args[3]

        payload_name = payload.id if isinstance(payload, ast.Name) else None
        declared_name = declared.id if isinstance(declared, ast.Name) else None
        label_value = label.value if isinstance(label, ast.Constant) else None

        if declared_name != "EXPORT_FIELDS":
            mismatched.append(
                f"write_export(..., {ast.unparse(declared)}, ...) does not "
                f"pass EXPORT_FIELDS")
            continue
        if payload_name is None or label_value is None:
            mismatched.append(
                f"write_export() payload/label must be a plain name and a "
                f"string literal, got "
                f"({ast.unparse(payload)}, {ast.unparse(label)})")
            continue
        if payload_name != label_value:
            mismatched.append(
                f"write_export() writes '{payload_name}' but labels it "
                f"'{label_value}' — the gate would report the wrong payload")
            continue
        guarded.add(payload_name)

    return guarded, mismatched


def registered_payloads(tree):
    """Payload labels registered with guard_export(path, fields, label)."""
    registered, mismatched = set(), []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else \
            (fn.id if isinstance(fn, ast.Name) else None)
        if name != "guard_export":
            continue
        if len(node.args) < 3:
            mismatched.append(
                f"guard_export() at line {node.lineno} takes "
                f"(path, declared, label)")
            continue
        declared, label = node.args[1], node.args[2]
        if not (isinstance(declared, ast.Name)
                and declared.id == "EXPORT_FIELDS"):
            mismatched.append(
                f"guard_export(..., {ast.unparse(declared)}, ...) at line "
                f"{node.lineno} does not pass EXPORT_FIELDS")
            continue
        if not (isinstance(label, ast.Constant)
                and isinstance(label.value, str)):
            mismatched.append(
                f"guard_export() label at line {node.lineno} must be a "
                f"string literal")
            continue
        registered.add(label.value)
    return registered, mismatched


def unguarded_write_lint(tree):
    """Obvious writes that bypass write_export.

    A lint, not a proof: an aliased or computed write can always evade
    this. The exit-time artifact guard is what actually enforces the
    export surface — see this module's docstring.
    """
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "json":
            for alias in node.names:
                if alias.name in ("dump", "dumps"):
                    aliases.add(alias.asname or alias.name)

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in ("dump", "dumps") \
                and isinstance(fn.value, ast.Name) and fn.value.id == "json":
            found.append(f"json.{fn.attr}() at line {node.lineno}")
        elif isinstance(fn, ast.Name) and fn.id in aliases:
            found.append(
                f"{fn.id}() at line {node.lineno} (json.dump imported "
                f"under another name)")
        elif isinstance(fn, ast.Attribute) and fn.attr == "write_text":
            found.append(f".write_text() at line {node.lineno}")
    return found


def export_table_rows():
    """Body rows of the export table, which is the authoritative list.

    Only that one table counts, and only its contiguous rows:

    * the prose around it names fields while explaining the rules —
      including this tool's own example — so scanning paragraphs would
      let a field dropped from the table pass on a passing mention;
    * later subsections carry unrelated tables of their own (the
      preflight's exit codes, for one), so the scan stops at the first
      heading of any level and at the end of the first table it finds,
      rather than absorbing every pipe-prefixed line in the region.
    """
    text = (ROOT / DOC_FILE).read_text()
    start = text.find(SECTION_MARKER)
    if start == -1:
        raise SystemExit(
            f"{DOC_FILE}: could not find the export section "
            f"({SECTION_MARKER!r}). Update tools/check_export_docs.py if it "
            f"was renamed."
        )

    lines = text[start:].splitlines()
    # Stop at the next heading of any level, not just "## ".
    for index, line in enumerate(lines[1:], start=1):
        if line.startswith("#"):
            lines = lines[:index]
            break

    rows, started = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            started = True
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue                      # |---|---| separator
            if cells[:1] == ["From"]:
                continue                      # header
            rows.append(stripped)
        elif started:
            break                             # end of the first table

    if not rows:
        raise SystemExit(
            f"{DOC_FILE}: found the export section but no table rows in it. "
            f"The table is the authoritative list of exported fields; "
            f"update tools/check_export_docs.py if its shape changed."
        )
    return rows


def main():
    documented = set()
    for row in export_table_rows():
        documented.update(re.findall(r"`([^`]+)`", row))
    problems = []

    for path, payloads, description in WRAPPERS:
        fields, tree = declared_fields(path)
        # Exact match only. Accepting the leaf name would let an
        # undocumented nested path ride on an unrelated documented field.
        missing_docs = [f for f in fields if f not in documented]
        guarded, mismatched = guarded_payloads(tree)
        registered, reg_problems = registered_payloads(tree)
        mismatched += reg_problems
        unenforced = [p for p in payloads if p not in guarded]
        unregistered = [p for p in payloads if p not in registered
                        and p not in _SHARES_PATH_WITH]
        raw = unguarded_write_lint(tree)

        status = "ok" if not (missing_docs or unenforced or unregistered
                              or mismatched or raw) else "DRIFT"
        print(f"{status:5s} {path} ({len(fields)} declared fields) — "
              f"{description}")
        for field in missing_docs:
            print(f"        undocumented: {field}")
            problems.append(f"{path}: {field} not in the export table")
        for payload in unenforced:
            print(f"        not guarded: {payload} is never written through "
                  f"write_export(path, {payload}, EXPORT_FIELDS, "
                  f"\"{payload}\")")
            problems.append(f"{path}: {payload} unguarded")
        for problem in mismatched:
            print(f"        {problem}")
            problems.append(f"{path}: {problem}")
        for payload in unregistered:
            print(f"        not registered: guard_export(path, "
                  f"EXPORT_FIELDS, \"{payload}\") is never called, so the "
                  f"exit-time artifact check would not cover this output")
            problems.append(f"{path}: {payload} unregistered")
        for problem in raw:
            print(f"        lint: {problem} — payloads leaving a silo "
                  f"should go through write_export()")
            problems.append(f"{path}: {problem}")

    if problems:
        print(f"\n{len(problems)} problem(s). Every declared field must be "
              f"named in {DOC_FILE}'s \"What actually leaves a silo\" "
              f"table, registered with guard_export() and written "
              f"through write_export(), both with a matching label.")
        return 1

    print("\nevery declared export field is documented; every payload is "
          "registered for the exit-time artifact check and written through "
          "the guarded validate-and-write call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
