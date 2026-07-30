"""Anti-rot guard for ``MAINTAINING.md``, the contract→consumer map (TD-003).

Asserts the map's **mechanical** claims only — that its pointers are real. Two
directions, deliberately asymmetric:

* pins→rows is REQUIRED — a new ``test_interface_stability.py`` cannot land
  without a row recording who it is pinned for;
* rows→pins is OPTIONAL — a row may legitimately have no pin (the DDL row and
  the submodule-path row do not), and recording an unpinned coupling is most of
  the map's value.

What this file deliberately does NOT assert: that the consumer lists are true.
That is cross-repo state no test in this repository can observe. Symbol-level
correctness is ``tests/mcp/test_interface_stability.py``'s job. See
``MAINTAINING.md`` § Known limitations.

Paths are the unit of assertion because they are unambiguous to regex and
immune to markdown reflow — ``ruff==0.16.0`` formats ``.md`` files, so anything
depending on table **column positions** would be fragile here. Line orientation
is safe, though: a markdown table row cannot span lines by grammar, so matching
on lines that start with ``|`` survives any reflow that still renders a table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MAP_PATH: Final[Path] = REPO_ROOT / "MAINTAINING.md"

# A backticked span whose whole content is a repo-relative path under ``src/``
# or ``tests/``, with an optional ``:<line>`` suffix. MAINTAINING.md carries
# LOCAL paths only — consumer identities and their file paths live in the
# private brief tracker, because this repo is public and they are not.
_LOCAL_PATH_IN_BACKTICKS: Final[re.Pattern[str]] = re.compile(
    r"`((?:src|tests)/[A-Za-z0-9_./-]+?)(?::\d+)?`"
)

_PREFIX_HINT: Final[str] = (
    "MAINTAINING.md cites paths in THIS repo only — if you were adding a "
    "consumer-repo path, do not: this repository is public and its consumers "
    "are not. See MAINTAINING.md's authoring rule."
)


def _map_text() -> str:
    assert MAP_PATH.is_file(), f"MAINTAINING.md is missing from the repo root ({MAP_PATH})"
    return MAP_PATH.read_text(encoding="utf-8")


def test_every_stability_pin_is_named_by_the_map() -> None:
    """A new public-surface stability pin cannot land without a MAINTAINING.md row (TD-003)."""
    text = _map_text()
    pins = sorted(REPO_ROOT.joinpath("tests").rglob("test_interface_stability.py"))

    # Guard against a vacuous pass: if the pins were renamed away wholesale, the
    # loop below would assert nothing at all and the check would silently rot.
    assert pins, (
        "no tests/**/test_interface_stability.py found — either the pin was renamed "
        "(update this glob AND MAINTAINING.md) or the public-surface guard was deleted"
    )

    # Match ONLY inside Runtime Contracts table rows, not the whole file. A bare
    # `rel not in text` is satisfied by any passing mention — including the one in
    # the "Row altitude" prose — which made the MCP row deletable with a green
    # suite. Scoping to table rows is reflow-safe: a markdown row cannot span
    # lines, so this survives anything `ruff format` does to the file.
    rows_text = "\n".join(ln for ln in text.splitlines() if ln.lstrip().startswith("|"))
    assert rows_text, (
        "MAINTAINING.md contains no markdown table rows — the Runtime Contracts "
        "table is missing or stopped being a table, and this check has gone vacuous"
    )

    unmapped = [
        rel
        for rel in (pin.relative_to(REPO_ROOT).as_posix() for pin in pins)
        if rel not in rows_text
    ]
    assert not unmapped, (
        f"stability pin(s) not named by MAINTAINING.md: {unmapped}. Every pinned surface "
        "needs a Runtime Contracts row recording WHICH consumers it is pinned for — "
        "a pin whose beneficiaries are unrecorded cannot be swept when it changes."
    )


def test_every_repo_path_named_by_the_map_exists() -> None:
    """Every local path MAINTAINING.md cites still resolves in this checkout (TD-003)."""
    text = _map_text()
    cited = sorted({match.group(1) for match in _LOCAL_PATH_IN_BACKTICKS.finditer(text)})

    assert cited, (
        "MAINTAINING.md cites no local src/ or tests/ paths — the map is either empty "
        "or its paths stopped being backticked, and this check has gone vacuous"
    )

    missing = [rel for rel in cited if not REPO_ROOT.joinpath(rel).exists()]
    assert not missing, f"MAINTAINING.md cites path(s) that do not exist: {missing}. {_PREFIX_HINT}"
