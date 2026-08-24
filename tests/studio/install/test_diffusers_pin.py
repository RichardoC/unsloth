# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned Diffusers revision has to survive a fresh install.sh, not just an update.

MiniMax-H3 needs a Diffusers revision newer than any published release, and Studio
refuses to load it otherwise. The pin originally lived in
studio/backend/requirements/base.txt, which did not reach fresh install.sh installs at
the time. base.txt now reaches those installs as an independent shared phase, but it
still runs too early to hold this pin safely.

These tests pin the shape that fixes it: exactly one file names diffusers, and the step
that installs it sits outside every skip.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REQ_ROOT = REPO_ROOT / "studio" / "backend" / "requirements"
PIN_FILE = REQ_ROOT / "diffusers-pin.txt"

# The shape install_python_stack._filter_requirements writes: a dot, the source stem,
# "-filtered-", then tempfile's random suffix. NamedTemporaryFile's suffixes are
# [A-Za-z0-9_]{8}, so this cannot swallow a checked-in file that merely starts with a dot.
_GENERATED_FILTER = re.compile(r"\.[\w.-]+-filtered-\w{8}\.txt")
STACK = REPO_ROOT / "studio" / "install_python_stack.py"
INSTALL_SH = REPO_ROOT / "install.sh"


def _requirements(path: pathlib.Path) -> list[str]:
    """Requirement lines only: comments and flag lines dropped."""
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and not text.startswith("-"):
            out.append(text)
    return out


def test_the_pin_file_exists_and_names_an_exact_revision():
    assert PIN_FILE.is_file(), f"{PIN_FILE} is missing"
    lines = _requirements(PIN_FILE)
    urls = [line for line in lines if "://" in line]
    assert len(urls) == 1, f"expected exactly one pinned URL, got {urls}"
    # A branch or tag would move under us; only a 40-char commit sha is reproducible.
    assert re.search(
        r"/archive/[0-9a-f]{40}\.zip", urls[0]
    ), f"the diffusers pin must name a full commit sha, not a moving ref: {urls[0]}"
    assert 'python_version >= "3.10"' in urls[0], (
        "diffusers dropped Python 3.9 in 0.38, so the archive needs a >= 3.10 marker or "
        "the resolver has no candidate at all on a 3.9 host"
    )


def test_only_the_pin_file_names_diffusers():
    """One source of truth. A second entry anywhere is how a release creeps back in:
    whichever step runs last wins, and the step order is not obvious from any one file."""
    offenders = {}
    for path in sorted(REQ_ROOT.rglob("*.txt")):
        if path == PIN_FILE:
            continue
        # install_python_stack._filter_requirements writes `.{stem}-filtered-XXXX.txt`
        # BESIDE the source on purpose, so relative -r/-c includes still resolve, and it
        # does not delete it. So a copy of the pin file can be sitting here while this
        # runs -- transiently under pytest-xdist, where another worker is exercising that
        # function, and durably on any machine that has run a real install. It is a
        # generated temp, not a second source of the pin.
        # Matched by that exact shape rather than by "starts with a dot": a checked-in
        # hidden file such as .constraints.txt is a real requirements file and a real
        # place the pin could be overridden from, so it stays in the scan.
        if _GENERATED_FILTER.fullmatch(path.name):
            continue
        # locks/ holds machine-generated resolutions, not authored requirements. A
        # with-deps closure names diffusers because something depends on it, and no
        # edit to a lock can change that -- gen_python_locks.sh and
        # gen_macos_bundle_lock.sh regenerate them from the authored files, so the
        # pin file is still the only place a human chooses a diffusers source. What
        # actually has to hold is that the pin lands last, which is what the next
        # test asserts rather than assumes.
        if path.parent.name == "locks":
            continue
        named = [line for line in _requirements(path) if line.lower().startswith("diffusers")]
        if named:
            offenders[str(path.relative_to(REPO_ROOT))] = named
    assert not offenders, (
        f"diffusers is requirement-listed outside diffusers-pin.txt: {offenders}. "
        f"Move it into the pin file so the dedicated late step remains authoritative."
    )


def test_the_bundle_installs_the_pin_after_every_lock():
    """The macOS payload installs the locks and then the pin, and only that order makes
    the exemption above safe: darwin-arm64-bundle.lock.txt resolves diffusers from PyPI,
    so a pin step that ran first would be overwritten and the bundle would ship a
    diffusers nobody chose."""
    script = REPO_ROOT / "scripts" / "build_macos_runtime.sh"
    if not script.is_file():  # pragma: no cover - the payload builder is macOS-only
        pytest.skip("scripts/build_macos_runtime.sh is not present")
    body = script.read_text(encoding = "utf-8")

    lock_loops = [i for i, line in enumerate(body.splitlines()) if 'for lock in "${LOCK_STEPS[@]}"' in line]
    assert lock_loops, "build_macos_runtime.sh no longer installs from LOCK_STEPS"

    pin_lines = [
        i for i, line in enumerate(body.splitlines())
        if "diffusers-pin.txt" in line and not line.lstrip().startswith("#")
    ]
    assert pin_lines, "build_macos_runtime.sh no longer installs the diffusers pin"

    assert max(pin_lines) > max(lock_loops), (
        "the diffusers pin is applied before the last lock install in "
        "build_macos_runtime.sh, so the lock's PyPI diffusers would overwrite the "
        "pinned archive and the bundle would ship an unpinned diffusers"
    )


def test_the_pin_step_is_not_gated_by_skip_base_or_no_torch():
    """The pin must sit at function top level so it reaches every install path."""
    tree = ast.parse(STACK.read_text(encoding = "utf-8"))

    def _installs_pin(node: ast.AST) -> bool:
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "id", None) != "pip_install":
                continue
            for kw in call.keywords:
                if kw.arg == "req" and "diffusers-pin.txt" in ast.dump(kw.value):
                    return True
        return False

    found = False
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for stmt in func.body:  # top level of the function only, no if/else nesting
            if _installs_pin(stmt):
                found = True
    assert found, (
        "no unconditional pip_install of diffusers-pin.txt found at the top level of any "
        "function in install_python_stack.py. Nested under an `if`, the pin can miss an "
        "install path."
    )


def test_the_pin_step_runs_after_every_other_requirements_install():
    """Ordering matters: a later `uv pip install -r ...` can re-resolve diffusers back to a
    release. Keeping the pin last means nothing is left that could walk it forward."""
    source = STACK.read_text(encoding = "utf-8")
    # The install SITE, not the first mention of the name. install_python_stack.py also
    # names diffusers-pin.txt in UNLOCKED_REQUIREMENTS (recording why it has no
    # hash-verified lock), which sits above pip_install and so above every step -- an
    # `index("diffusers-pin.txt")` would anchor there and read every step as later.
    pin_at = source.index('REQ_ROOT / "diffusers-pin.txt"')
    later = [
        name
        for name in (
            "extras.txt",
            "extras-no-deps.txt",
            "studio.txt",
            "base.txt",
            "no-torch-runtime.txt",
            "data-designer-deps.txt",
            "data-designer.txt",
        )
        if source.rfind(name) > pin_at
    ]
    assert not later, f"these requirements files are installed after the diffusers pin: {later}"


def test_install_sh_still_delegates_the_core_package_skip():
    """The handoff flag skips core packages while allowing other base entries through."""
    assert 'SKIP_STUDIO_BASE="$_SKIP_BASE"' in INSTALL_SH.read_text(encoding = "utf-8")
    assert "_SKIP_BASE=1" in INSTALL_SH.read_text(encoding = "utf-8")
