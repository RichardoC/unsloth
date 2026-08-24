# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No `client.messages.create(...)` may pass a sampling parameter as a keyword.

The anthropic Python SDK removed the typed sampling parameters -- temperature, top_p,
top_k -- when current Claude models stopped accepting them. Passing one is not a server
rejection you would see in a response; it is a TypeError raised before the request is
built:

    TypeError: Messages.create() got an unexpected keyword argument 'temperature'

Our smoke tests point the SDK at Unsloth's own Anthropic-compatible endpoint, which does
still honour those fields on the wire, so they belong in `extra_body` -- beside `seed`,
which has always had to travel that way for the same reason.

This exists because fixing the three call sites embedded in the workflow YAML left a
fourth in a shared script under .github/scripts/, and CI found it one push later. A grep
scoped to the files you happen to be looking at is not a repo-wide fix; this test is.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# `temperature = 0.0,` as an argument line -- not `"temperature": 0.0` inside a dict,
# which is how it legitimately appears in extra_body and in the OpenAI-shaped payloads
# these same files build.
_SAMPLING_KWARG = re.compile(r"^\s*(temperature|top_p|top_k)\s*=", re.M)

# Where a call's arguments end. Nesting inside the argument list is fine to skip past:
# no sampling kwarg can hide beyond the closing paren of the call.
_CALL = "messages.create("

SEARCH_GLOBS = (
    ".github/workflows/*.yml",
    ".github/scripts/**/*.py",
    "studio/backend/tests/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
)


def _call_sites() -> list[tuple[pathlib.Path, int, str]]:
    """(path, line number, argument text) for every messages.create( in the tree."""
    sites = []
    for glob in SEARCH_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob)):
            if not path.is_file() or path.name == pathlib.Path(__file__).name:
                continue
            text = path.read_text(encoding = "utf-8", errors = "replace")
            start = 0
            while (found := text.find(_CALL, start)) != -1:
                start = found + len(_CALL)
                depth, end = 1, start
                while end < len(text) and depth:
                    depth += {"(": 1, ")": -1}.get(text[end], 0)
                    end += 1
                sites.append((path, text.count("\n", 0, found) + 1, text[start:end]))
    return sites


def test_the_search_actually_finds_the_known_call_sites():
    """A test that silently matched nothing would pass forever."""
    sites = _call_sites()
    assert len(sites) >= 4, f"expected the known messages.create call sites, found {sites}"


def test_no_sampling_parameter_is_passed_as_a_keyword():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line} passes {match.group(1)} as a keyword"
        for path, line, args in _call_sites()
        if (match := _SAMPLING_KWARG.search(args))
    ]
    assert not offenders, (
        "the anthropic SDK has no typed sampling parameters; these raise TypeError "
        "before any request is made:\n  " + "\n  ".join(offenders)
        + "\nMove them into extra_body, where `seed` already travels."
    )
