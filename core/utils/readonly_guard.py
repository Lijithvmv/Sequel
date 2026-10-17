"""Deterministic read-only guard for generated SQL.

The SQL validator is an LLM judgement, so it can't be the thing that stops a generated ``DROP`` or ``UPDATE``. This
check runs in code, right before execution, and refuses anything that isn't a single read-only statement:

* exactly one statement (a trailing ``;`` is allowed, a second statement is not);
* it starts with ``SELECT``, ``WITH``, ``SHOW``, ``DESCRIBE``/``DESC`` or ``EXPLAIN``;
* no write, DDL, privilege, session or file keyword appears anywhere outside string literals and quoted identifiers
  (this also catches data-modifying CTEs such as ``WITH x AS (DELETE ...)`` and ``SELECT ... INTO OUTFILE``);
* no MySQL executable comment (``/*! ... */``), because MySQL runs its contents.

It is deliberately strict: a refused read-only query is an inconvenience, a passed write is not. Defence in depth still
matters: connect with a database user that only has ``SELECT`` rights (see the README).
"""

import re

ALLOWED_FIRST = {"select", "with", "show", "describe", "desc", "explain"}

FORBIDDEN = {
    "insert", "update", "delete", "merge", "replace", "upsert",
    "drop", "alter", "create", "truncate", "rename", "comment",
    "grant", "revoke", "set", "reset", "lock", "unlock",
    "call", "exec", "execute", "do", "handler", "prepare", "deallocate",
    "load", "outfile", "dumpfile", "load_file", "copy", "vacuum", "analyze",
    "attach", "detach", "pragma", "kill", "shutdown", "install", "uninstall",
    "begin", "commit", "rollback", "savepoint", "start", "release",
}


class UnsafeQueryError(ValueError):
    """Raised when a query is not a single read-only statement."""


def _strip_literals_and_comments(sql: str) -> str:
    """Remove comments, string literals and quoted identifiers, so keywords inside them don't count."""
    out = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        two = sql[i:i + 2]
        if two == "--" or c == "#":
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
        elif two == "/*":
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
        elif c in ("'", '"', "`"):
            j = i + 1
            while j < n:
                if sql[j] == "\\" and c != "`":
                    j += 2
                    continue
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:  # doubled quote escape
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            out.append(" x ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def check_read_only(sql: str) -> str:
    """Return the query unchanged if it is a single read-only statement; raise UnsafeQueryError otherwise."""
    if not sql or not sql.strip():
        raise UnsafeQueryError("empty query")
    if "/*!" in sql:
        raise UnsafeQueryError("MySQL executable comments (/*! ... */) are not allowed")
    body = _strip_literals_and_comments(sql).strip()
    body = body[:-1].rstrip() if body.endswith(";") else body
    if ";" in body:
        raise UnsafeQueryError("only one statement is allowed")
    words = re.findall(r"[a-z_][a-z0-9_]*", body.lower())
    if not words or words[0] not in ALLOWED_FIRST:
        raise UnsafeQueryError(f"only read-only statements are allowed (got {words[0] if words else 'nothing'!r})")
    bad = sorted(set(words) & FORBIDDEN)
    if bad:
        raise UnsafeQueryError(f"write or session keyword not allowed in a read-only query: {', '.join(bad)}")
    return sql
