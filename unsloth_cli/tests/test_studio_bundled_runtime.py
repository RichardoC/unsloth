# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CLI seam for the runtime baked into Unsloth.app.

`Unsloth.app/Contents/Resources/runtime/` carries a complete CPython and a
complete `site-packages`, and the desktop app launches the backend as that
interpreter. `unsloth studio` used to decide it was installed by asking whether
`sys.prefix` sits under `<STUDIO_HOME>/unsloth_studio`; from the bundled
interpreter that is false, so it went looking for a second interpreter that a
self-contained install never has and exited 1 with "Unsloth Studio not set up.
Run install.sh first." -- after preflight had already reported Ready.

Two claims are pinned here, and the second matters as much as the first:

  * from a bundled runtime the CLI serves in-process, exactly as it does from the
    managed venv;
  * with no bundle -- `curl | sh` installs, pip installs, dev checkouts, CI --
    every answer is the one it was before, including for a stray
    UNSLOTH_BUNDLED_SITE_PACKAGES left in somebody's shell.

Modeled on test_studio_secure_flag.py.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _bundled():
    from unsloth_cli import _bundled_runtime as _mod
    return _mod


# ── the payload shape studio/src-tauri/src/bundled_runtime.rs builds ─────────


def _make_runtime(root: Path, *, manifest: bool = True) -> Path:
    """A minimal but layout-faithful `Resources/runtime` tree."""
    (root / "site-packages").mkdir(parents = True, exist_ok = True)
    (root / "python" / "bin").mkdir(parents = True, exist_ok = True)
    if manifest:
        (root / "BUNDLE_MANIFEST.json").write_text(
            json.dumps({"components": {}}), encoding = "utf-8"
        )
    return root


def _pose_as_bundled(monkeypatch, root: Path) -> None:
    """Everything the real launch provides: the variable, and an interpreter whose
    prefix is the payload's own `runtime/python`."""
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(root / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(root / "python"))


# ── detection ────────────────────────────────────────────────────────────────


def test_detects_the_bundled_runtime_when_all_three_facts_hold(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    assert _bundled().bundled_runtime_root() == root.resolve()
    assert _bundled().bundled_site_packages() == root.resolve() / "site-packages"


def test_no_bundle_when_the_variable_is_unset_or_blank(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    assert _bundled().bundled_runtime_root() is None
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", "   ")
    assert _bundled().bundled_runtime_root() is None


def test_a_stray_variable_alone_does_not_make_an_install_bundled(tmp_path, monkeypatch):
    """The whole point of corroborating: an export left in a shell profile, or
    inherited from a script that once ran the app, must not talk an ordinary
    install into serving from a bundle it does not have."""
    root = _make_runtime(tmp_path / "runtime")
    # Variable set, payload real -- but this interpreter is an ordinary venv.
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(root / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "home" / ".unsloth" / "studio" / "unsloth_studio"))
    assert _bundled().bundled_runtime_root() is None


def test_no_bundle_when_the_named_directory_does_not_exist(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(root / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(root / "python"))
    assert _bundled().bundled_runtime_root() is None


def test_no_bundle_without_the_payload_manifest(tmp_path, monkeypatch):
    """BUNDLE_MANIFEST.json is the build's receipt, and the Rust health check
    refuses to launch a payload missing it -- so requiring it here cannot reject
    anything the app would have started, and it is one more fact a stray variable
    cannot supply."""
    root = _make_runtime(tmp_path / "runtime", manifest = False)
    _pose_as_bundled(monkeypatch, root)
    assert _bundled().bundled_runtime_root() is None


def test_a_directory_that_merely_contains_the_interpreter_is_not_a_bundle(
    tmp_path, monkeypatch
):
    # sys.prefix inside, manifest absent: the pair is what decides, not either half.
    prefix = tmp_path / "venv"
    (tmp_path / "site-packages").mkdir(parents = True)
    (prefix / "bin").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(tmp_path / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(prefix))
    assert _bundled().bundled_runtime_root() is None


def test_path_containment_answers_for_the_writers(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    mod = _bundled()
    assert mod.path_is_inside_bundled_runtime(root / "llama.cpp") is True
    assert mod.path_is_inside_bundled_runtime(root) is True
    assert mod.path_is_inside_bundled_runtime(tmp_path / "elsewhere") is False
    assert mod.path_is_inside_bundled_runtime(None) is False


def test_oxc_modules_are_reported_only_when_present(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    assert _bundled().bundled_oxc_node_modules() is None
    (root / "oxc-node-modules").mkdir()
    assert _bundled().bundled_oxc_node_modules() == root.resolve() / "oxc-node-modules"


# ── the gate the launch paths read ───────────────────────────────────────────


def test_serves_in_process_from_a_bundled_runtime(tmp_path, monkeypatch):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path / "data")
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    assert studio_mod._serves_backend_in_process() is True
    # Same claim as the in-venv case, so the pre-exposure gate may skip the strip.
    assert (
        studio_mod._child_self_suppresses(in_studio_venv = True, child_run_py = None) is True
    )


def test_serves_in_process_in_the_managed_venv_without_any_bundle(tmp_path, monkeypatch):
    studio_mod = _studio()
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path / "data")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "data" / "unsloth_studio"))
    assert studio_mod._serves_backend_in_process() is True


def test_does_not_serve_in_process_from_an_unrelated_interpreter(tmp_path, monkeypatch):
    studio_mod = _studio()
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path / "data")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "usr"))
    assert studio_mod._serves_backend_in_process() is False


# ── functional: `unsloth studio --api-only` end to end ───────────────────────


class _FakeRunModule(types.SimpleNamespace):
    pass


def _install_common_launch_stubs(monkeypatch, tmp_path):
    """Everything between the gate and run_server that would touch the machine."""
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path / "data")
    monkeypatch.setattr(sys, "platform", "linux")

    @contextlib.contextmanager
    def _no_guard(*, inherited = False):
        yield False

    monkeypatch.setattr(studio_mod, "_studio_runtime_launch_guard", _no_guard)

    @contextlib.contextmanager
    def _no_deps(_label):
        yield

    monkeypatch.setattr(studio_mod._studio_deps, "studio_backend_imports", _no_deps)

    calls: list[dict] = []
    shutdown = threading.Event()
    shutdown.set()  # the serve loop returns immediately

    fake = _FakeRunModule(
        run_server = lambda **kwargs: calls.append(kwargs),
        _shutdown_event = shutdown,
        _server = None,
        _graceful_shutdown = lambda _server: None,
        _wait_for_server_shutdown = lambda: None,
    )
    monkeypatch.setattr(studio_mod, "_load_run_module", lambda: fake)
    return calls


def _invoke_studio_default(args):
    app = typer.Typer()
    app.command()(_studio().studio_default)
    return CliRunner().invoke(app, args, catch_exceptions = True)


def test_studio_from_a_bundled_runtime_serves_instead_of_exiting(tmp_path, monkeypatch):
    """The blocker itself: this used to print "Unsloth Studio not set up" and exit 1
    because ~/.unsloth/studio/unsloth_studio/bin/python does not exist on a machine
    whose whole installation was copying the .app across."""
    calls = _install_common_launch_stubs(monkeypatch, tmp_path)
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)

    result = _invoke_studio_default(["--api-only", "--port", "7777"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert "not set up" not in (result.output or "")
    assert len(calls) == 1, calls
    assert calls[0]["port"] == 7777
    assert calls[0]["api_only"] is True


def test_studio_without_a_bundle_still_exits_one_when_the_venv_is_missing(
    tmp_path, monkeypatch
):
    """The load-bearing half: no bundle, no managed venv, same message and same
    exit code as before this seam existed."""
    calls = _install_common_launch_stubs(monkeypatch, tmp_path)
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "usr"))

    result = _invoke_studio_default(["--api-only", "--port", "7777"])

    assert result.exit_code == 1, (result.exit_code, result.output)
    assert "Unsloth Studio not set up. Run install.sh first." in (result.output or "")
    assert calls == []


def test_a_stray_variable_does_not_turn_the_missing_venv_into_a_launch(
    tmp_path, monkeypatch
):
    """A shell that exports UNSLOTH_BUNDLED_SITE_PACKAGES for unrelated reasons must
    get today's behaviour, not an in-process serve with no backend behind it."""
    calls = _install_common_launch_stubs(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(tmp_path / "wherever"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "usr"))

    result = _invoke_studio_default(["--api-only", "--port", "7777"])

    assert result.exit_code == 1, (result.exit_code, result.output)
    assert "Unsloth Studio not set up. Run install.sh first." in (result.output or "")
    assert calls == []


def test_run_subcommand_takes_the_same_seam(tmp_path, monkeypatch):
    """`unsloth studio run` has the identical gate around its own re-exec, and the
    bundle has no `unsloth` console script to re-exec through at all."""
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path / "data")
    root = _make_runtime(tmp_path / "runtime")

    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    monkeypatch.setattr(sys, "prefix", str(root / "python"))
    assert studio_mod._serves_backend_in_process() is False

    _pose_as_bundled(monkeypatch, root)
    assert studio_mod._serves_backend_in_process() is True


# ── the guard that stops applying, made explicit rather than removed ─────────


def test_the_hsa_override_clear_still_runs_from_a_bundled_launch(tmp_path, monkeypatch):
    """#7331's spoof clear is a chokepoint in front of every launch path, and the
    bundled path is one of them. It finds nothing in a macOS payload (no ROCm
    stack, and the distributions are not in a venv layout), so it must be a no-op
    rather than a crash -- and it must still be called."""
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path / "data")
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    assert studio_mod._clear_hsa_override_before_launch(silent = True) is None
    # Untouched: nothing contradicted it, so nothing is corrected.
    import os
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


@pytest.mark.parametrize("subcommand", ["studio_default", "run"])
def test_neither_launch_path_reads_sys_prefix_directly_any_more(subcommand):
    """Both gates go through the one predicate, so a future change cannot fix one
    and leave the other exiting 1 from the bundle."""
    import inspect

    source = inspect.getsource(getattr(_studio(), subcommand))
    assert "_serves_backend_in_process()" in source
    assert "sys.prefix.startswith" not in source
