# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

""""Am I the runtime inside Unsloth.app?" -- the one predicate, twice.

The macOS app runs the CLI and the backend from a signed, read-only payload at
``Unsloth.app/Contents/Resources/runtime/``, and tells the interpreter about it
through ``UNSLOTH_BUNDLED_SITE_PACKAGES`` (``studio/src-tauri/src/
bundled_runtime.rs``; ``-I`` implies ``-E``, so ``PYTHONPATH`` is not available to
carry it). Two decisions hang off the answer:

  * the CLI serves the backend in-process instead of hunting for a second
    interpreter that a self-contained install does not have;
  * the llama.cpp / whisper.cpp updaters refuse to install over the bundled copies,
    which are code-signed and, in ``/Applications``, root-owned.

Both are load-bearing in the wrong direction if the answer can be forced from the
environment: a variable left in a shell profile would send an ordinary
``curl | sh`` install down the bundled path, and -- worse -- would let anything
that can set an environment variable choose which directory an "immutable
runtime" is. So the answer requires three facts, and the caller's environment
supplies only one of them:

  1. the variable names a directory that exists;
  2. its parent holds ``BUNDLE_MANIFEST.json``, the receipt
     ``scripts/build_macos_runtime.sh`` writes and ``BundledRuntime::health``
     refuses to launch without;
  3. ``sys.prefix`` resolves inside that same parent -- the running interpreter is
     the payload's own, which no variable can arrange.

The predicate lives in two places because the backend is self-contained and runs
from environments with no ``unsloth_cli`` on ``sys.path`` (the same reason
``utils/host_policy.py`` mirrors ``unsloth_cli/_tool_policy.py``). Mirrors drift,
so this file drives BOTH copies through the same scenarios and requires the same
answer from each -- functionally, not by comparing source text.

Loaded by file path so neither package has to be importable: the CLI copy is
stdlib-only but ``unsloth_cli/__init__.py`` is not.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_MODULE_PATH = REPO_ROOT / "unsloth_cli" / "_bundled_runtime.py"
BACKEND_MODULE_PATH = REPO_ROOT / "studio" / "backend" / "utils" / "bundled_runtime.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def implementations() -> dict:
    return {
        "cli": _load(CLI_MODULE_PATH, "_probe_cli_bundled_runtime"),
        "backend": _load(BACKEND_MODULE_PATH, "_probe_backend_bundled_runtime"),
    }


def _make_runtime(root: Path, *, manifest: bool = True) -> Path:
    """The payload layout the Rust side resolves through."""
    (root / "site-packages").mkdir(parents = True, exist_ok = True)
    (root / "python" / "bin").mkdir(parents = True, exist_ok = True)
    if manifest:
        (root / "BUNDLE_MANIFEST.json").write_text(
            json.dumps({"components": {}}), encoding = "utf-8"
        )
    return root


def _roots(implementations, monkeypatch, *, site_packages, prefix) -> dict:
    """Both answers, under one environment."""
    if site_packages is None:
        monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(site_packages))
    monkeypatch.setattr(sys, "prefix", str(prefix))
    return {name: mod.bundled_runtime_root() for name, mod in implementations.items()}


def _agree(answers: dict):
    assert answers["cli"] == answers["backend"], answers
    return answers["cli"]


# ── the three facts, together and apart ──────────────────────────────────────


def test_a_whole_payload_run_by_its_own_interpreter_is_a_bundle(
    implementations, monkeypatch, tmp_path
):
    root = _make_runtime(tmp_path / "runtime")
    answers = _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        prefix = root / "python",
    )
    assert _agree(answers) == root.resolve()


def test_the_variable_on_its_own_is_not_enough(implementations, monkeypatch, tmp_path):
    """The spoofing case, and the reason this is not a bare env-var read: an export
    left in a shell profile, or inherited from a process that once launched the
    app, must not persuade a normal install that it is running from a bundle."""
    root = _make_runtime(tmp_path / "runtime")
    answers = _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        # An ordinary managed install's interpreter, nowhere near the payload.
        prefix = tmp_path / "home" / ".unsloth" / "studio" / "unsloth_studio",
    )
    assert _agree(answers) is None


def test_the_manifest_is_required(implementations, monkeypatch, tmp_path):
    root = _make_runtime(tmp_path / "runtime", manifest = False)
    answers = _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        prefix = root / "python",
    )
    assert _agree(answers) is None


def test_a_directory_that_does_not_exist_is_not_a_bundle(
    implementations, monkeypatch, tmp_path
):
    root = tmp_path / "runtime"
    answers = _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        prefix = root / "python",
    )
    assert _agree(answers) is None


def test_an_interpreter_beside_an_unmarked_tree_is_not_a_bundle(
    implementations, monkeypatch, tmp_path
):
    """Fact 3 without fact 2: a venv whose parent happens to hold a site-packages
    directory. Only the conjunction says "bundle"."""
    (tmp_path / "site-packages").mkdir()
    (tmp_path / "venv" / "bin").mkdir(parents = True)
    answers = _roots(
        implementations,
        monkeypatch,
        site_packages = tmp_path / "site-packages",
        prefix = tmp_path / "venv",
    )
    assert _agree(answers) is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_is_not_a_bundle(implementations, monkeypatch, tmp_path, value):
    root = _make_runtime(tmp_path / "runtime")
    if value is None:
        monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", value)
    monkeypatch.setattr(sys, "prefix", str(root / "python"))
    assert _agree({name: m.bundled_runtime_root() for name, m in implementations.items()}) is None


def test_a_descendant_interpreter_still_counts(implementations, monkeypatch, tmp_path):
    """Children of the bundled launch are ordinary interpreters given
    ``PYTHONHOME=runtime/python``, so their prefix is inside the payload too --
    deeper than the root, which containment has to allow."""
    root = _make_runtime(tmp_path / "runtime")
    answers = _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        prefix = root / "python" / "lib" / "somewhere",
    )
    assert _agree(answers) == root.resolve()


# ── what the callers actually ask ────────────────────────────────────────────


def test_containment_is_answered_for_both_copies(implementations, monkeypatch, tmp_path):
    """The question every writer has to ask before it writes into the .app."""
    root = _make_runtime(tmp_path / "runtime")
    _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        prefix = root / "python",
    )
    for name, module in implementations.items():
        assert module.path_is_inside_bundled_runtime(root / "llama.cpp") is True, name
        assert module.path_is_inside_bundled_runtime(root) is True, name
        assert module.path_is_inside_bundled_runtime(tmp_path / "elsewhere") is False, name
        assert module.path_is_inside_bundled_runtime(None) is False, name


def test_nothing_is_inside_a_bundle_that_is_not_there(
    implementations, monkeypatch, tmp_path
):
    """With no bundle the answer is False for every path, so the update guards are
    inert on every non-bundled install rather than merely unlikely to fire."""
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    for name, module in implementations.items():
        for candidate in (tmp_path, Path("/"), Path.home() / ".unsloth" / "llama.cpp"):
            assert module.path_is_inside_bundled_runtime(candidate) is False, (name, candidate)


def test_the_oxc_modules_are_reported_only_when_the_payload_has_them(
    implementations, monkeypatch, tmp_path
):
    root = _make_runtime(tmp_path / "runtime")
    _roots(
        implementations,
        monkeypatch,
        site_packages = root / "site-packages",
        prefix = root / "python",
    )
    for name, module in implementations.items():
        assert module.bundled_oxc_node_modules() is None, name
    (root / "oxc-node-modules").mkdir()
    for name, module in implementations.items():
        assert (
            module.bundled_oxc_node_modules() == root.resolve() / "oxc-node-modules"
        ), name


def test_both_copies_name_the_variable_the_rust_side_sets(implementations):
    """One typo here and a self-contained .dmg installs nothing and then refuses to
    start, so the name is asserted rather than assumed."""
    for name, module in implementations.items():
        assert module.SITE_PACKAGES_ENV == "UNSLOTH_BUNDLED_SITE_PACKAGES", name
        assert module.BUNDLE_MANIFEST_NAME == "BUNDLE_MANIFEST.json", name
    rust = (REPO_ROOT / "studio" / "src-tauri" / "src" / "bundled_runtime.rs").read_text(
        encoding = "utf-8"
    )
    assert 'SITE_PACKAGES_ENV: &str = "UNSLOTH_BUNDLED_SITE_PACKAGES"' in rust
    assert 'MANIFEST_NAME: &str = "BUNDLE_MANIFEST.json"' in rust
