"""Anti-rot guard for the declared Python support range (TS-001).

Three places declare which Pythons this package supports, and nothing but habit
kept them in agreement:

* ``pyproject.toml`` ``requires-python`` — what pip will INSTALL on;
* ``pyproject.toml`` ``classifiers`` — what PyPI ADVERTISES;
* ``.github/workflows/ci.yml`` ``matrix.python-version`` — what is actually RUN.

The defect TS-001 recorded is the gap between the first and the third: an open
``>=3.11`` bound admits every future Python, so each one missing from the matrix
is a support claim no gate checks. A green CI is not evidence about a version CI
does not run — the same shape as TD-004 (unpinned tooling) and TD-008 (actions
that execute zero times on a PR).

What this file does NOT assert: that the code works on those versions. That is
the matrix's job, and it can only be done by running it. This file asserts the
three lists agree, so adding a version to one without the others fails here
rather than shipping a claim nothing backs.

Parsing is deliberately narrow. ``tomllib`` is stdlib from 3.11 (the floor), so
pyproject is read properly rather than by regex. The workflow is matched with a
single anchored pattern instead of adding a YAML parser to the dev extra for one
line — the pins there are exact (TD-004) and a new dependency needs a better
reason than convenience.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW: Final[Path] = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# `python-version: ["3.11", "3.12", ...]` — the matrix list in ci.yml. Anchored
# on the `[` so it cannot match release.yml's scalar `python-version: "3.12"`
# shape if this pattern is ever pointed at another workflow.
_MATRIX: Final[re.Pattern[str]] = re.compile(r"python-version:\s*\[([^\]]+)\]")
_VERSION: Final[re.Pattern[str]] = re.compile(r"(\d+\.\d+)")


def _matrix_versions() -> list[str]:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    match = _MATRIX.search(text)
    assert match is not None, (
        f"no `python-version: [...]` matrix found in {CI_WORKFLOW.name} — either the "
        "matrix was removed (which would mean CI tests exactly one Python) or it "
        "stopped being an inline list, and this guard has gone vacuous"
    )
    return _VERSION.findall(match.group(1))


def _classifier_versions() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    prefix = "Programming Language :: Python :: "
    return [
        c.removeprefix(prefix)
        for c in data["project"]["classifiers"]
        # The bare `:: 3` classifier carries no minor and is not a per-version
        # claim, so it is not part of the comparison.
        if c.startswith(prefix) and "." in c.removeprefix(prefix)
    ]


def _as_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_ci_matrix_and_classifiers_declare_the_same_versions() -> None:
    """A Python version cannot be advertised on PyPI without CI running it (TS-001)."""
    matrix = _matrix_versions()
    classifiers = _classifier_versions()

    # Both must be non-empty, or the set comparison below passes trivially.
    assert matrix, "CI matrix parsed to zero versions"
    assert classifiers, "no per-version `Programming Language :: Python :: X.Y` classifiers"

    assert set(matrix) == set(classifiers), (
        f"CI matrix {sorted(matrix)} and classifiers {sorted(classifiers)} disagree. "
        "Advertising a version CI does not run is a claim nothing backs; running one "
        "that is not advertised hides work already being done. Update both together."
    )


def test_requires_python_floor_matches_the_lowest_tested_version() -> None:
    """The install floor is a version CI actually runs (TS-001)."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requires = data["project"]["requires-python"]

    floor_match = re.search(r">=\s*(\d+\.\d+)", requires)
    assert floor_match is not None, (
        f"requires-python is {requires!r}, which has no `>=X.Y` floor this guard can "
        "read. If the bound style changed deliberately, update this test with it."
    )
    floor = floor_match.group(1)
    lowest_tested = min(_matrix_versions(), key=_as_tuple)

    assert floor == lowest_tested, (
        f"requires-python floor is {floor} but the lowest version CI runs is "
        f"{lowest_tested}. pip would install on {floor} with nothing having tested it."
    )


def test_requires_python_upper_bound_is_covered_by_the_matrix() -> None:
    """An open upper bound obliges the matrix to track each new release (TS-001).

    This does not force a ceiling — an open bound is a deliberate choice for a
    library, since a ceiling ships into every downstream resolver and pip cannot
    relax it (the same reasoning that keeps the consumer-facing specifiers
    unbounded in the ``dev`` extra's TD-004 note). It pins the *consequence*: with
    no ceiling, the matrix is the only thing bounding the claim, so this asserts
    the matrix is what the declaration points at rather than lagging it silently.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requires = data["project"]["requires-python"]
    ceiling = re.search(r"<\s*=?\s*(\d+\.\d+)", requires)

    if ceiling is None:
        # Open bound: every version at or above the floor is admitted, so the
        # matrix's own top entry is the only recorded statement of what has been
        # verified. Nothing to compare it against — the guard is the sibling test
        # above plus the ci.yml comment telling a reader to extend both.
        assert _matrix_versions(), "open requires-python bound with an empty CI matrix"
        return

    highest_tested = max(_matrix_versions(), key=_as_tuple)
    assert _as_tuple(highest_tested) >= _as_tuple(ceiling.group(1)), (
        f"requires-python admits up to {ceiling.group(1)} but CI stops at "
        f"{highest_tested} — the top of the declared range is untested."
    )
