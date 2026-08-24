# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nothing may write into the runtime that ships inside Unsloth.app.

`Unsloth.app/Contents/Resources/runtime/` carries its own llama.cpp and
whisper.cpp, and the app points UNSLOTH_LLAMA_CPP_PATH / UNSLOTH_WHISPER_CPP_PATH
at them. Every existing check therefore calls them "managed" -- truthfully: the
active binary really is the one Unsloth put there. What none of them could see is
that the tree is code-signed and, in /Applications, root-owned. An update
installing over it either fails outright or succeeds and invalidates the
signature, after which Gatekeeper refuses to launch the app the user just
"updated".

Pinned here:

  * detection: all three facts, and a stray UNSLOTH_BUNDLED_SITE_PACKAGES on its
    own is not one of them;
  * the status side offers nothing for a bundled root -- an update the user cannot
    apply must not be advertised, which is the bug the release pin already had;
  * the apply side refuses one anyway, because a direct POST never reads the
    status that would have withheld the button;
  * and none of it fires for an ordinary managed install, which must keep offering
    and applying updates exactly as before.

Hermetic: no network, no installer, no real install (mirrors
test_llama_cpp_update.py / test_combined_update.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.bundled_runtime as bundled  # noqa: E402
import utils.llama_cpp_freshness as freshness  # noqa: E402
import utils.llama_cpp_update as upd  # noqa: E402
import utils.whisper_cpp_freshness as wfresh  # noqa: E402
import utils.whisper_cpp_update as wupd  # noqa: E402
from utils.prebuilt import update_flow as flow  # noqa: E402

MARKER = "UNSLOTH_PREBUILT_INFO.json"
WHISPER_MARKER = "UNSLOTH_WHISPER_PREBUILT_INFO.json"


# ── the payload shape studio/src-tauri/src/bundled_runtime.rs builds ─────────


def _make_runtime(root: Path, *, manifest: bool = True) -> Path:
    (root / "site-packages").mkdir(parents = True, exist_ok = True)
    (root / "python" / "bin").mkdir(parents = True, exist_ok = True)
    if manifest:
        (root / "BUNDLE_MANIFEST.json").write_text(
            json.dumps({"components": {}}), encoding = "utf-8"
        )
    return root


def _pose_as_bundled(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(root / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(root / "python"))


def _write_llama_install(dir_: Path, tag: str) -> str:
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("stub")
    (dir_ / MARKER).write_text(
        json.dumps(
            {
                "tag": tag,
                "release_tag": tag,
                "published_repo": "unslothai/llama.cpp",
                "installed_at_utc": "2020-01-01T00:00:00Z",
            }
        )
    )
    return str(binary)


def _write_whisper_install(dir_: Path, tag: str) -> str:
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "whisper-server"
    binary.write_text("stub")
    (dir_ / WHISPER_MARKER).write_text(
        json.dumps(
            {
                "release_tag": tag,
                "upstream_tag": tag.split("-")[0],
                "published_repo": "unslothai/whisper.cpp",
                "backend": "cpu",
                "installed_at_utc": "2020-01-01T00:00:00Z",
            }
        )
    )
    return str(binary)


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch, tmp_path):
    freshness.reset_caches()
    wfresh.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    wupd._resolve_memo.clear()
    monkeypatch.setattr(freshness, "_cache_dir", lambda: tmp_path / ".llama_cache")
    monkeypatch.setattr(wfresh, "_cache_dir", lambda: tmp_path / ".whisper_cache")
    # Track-latest policy, as in the sibling suites; the release pin has its own.
    monkeypatch.setenv("UNSLOTH_PREBUILT_ALLOW_LATEST", "1")
    for var in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        "WHISPER_SERVER_PATH",
        "UNSLOTH_WHISPER_CPP_PATH",
        "UNSLOTH_BUNDLED_SITE_PACKAGES",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    monkeypatch.setattr(wfresh, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    # An installer script always "exists" so a refusal is never mistaken for one.
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(
        wupd, "_installer_script", lambda: tmp_path / "install_whisper_prebuilt.py"
    )
    yield
    freshness.reset_caches()
    wfresh.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    wupd._resolve_memo.clear()


# ── detection ────────────────────────────────────────────────────────────────


def test_detects_the_bundled_runtime_when_all_three_facts_hold(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    assert bundled.bundled_runtime_root() == root.resolve()


def test_a_stray_variable_alone_is_not_a_bundled_runtime(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime")
    # Set, and pointing at a real payload -- but this interpreter is not its own.
    monkeypatch.setenv("UNSLOTH_BUNDLED_SITE_PACKAGES", str(root / "site-packages"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))
    assert bundled.bundled_runtime_root() is None
    assert bundled.path_is_inside_bundled_runtime(root / "llama.cpp") is False


def test_no_bundle_without_the_payload_manifest(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime", manifest = False)
    _pose_as_bundled(monkeypatch, root)
    assert bundled.bundled_runtime_root() is None


def test_no_bundle_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    assert bundled.bundled_runtime_root() is None
    assert flow.immutable_runtime_root(tmp_path / "anything") is False


# ── llama.cpp ────────────────────────────────────────────────────────────────


def _bundled_llama(tmp_path, monkeypatch, *, installed = "b9493"):
    """A bundled runtime whose llama.cpp is the active install, exactly as the app
    arranges it: a normal marker, and UNSLOTH_LLAMA_CPP_PATH pointing at it."""
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    binary = _write_llama_install(root / "llama.cpp", installed)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root / "llama.cpp"))
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    # A newer release exists, so only the guard can be what withholds the update.
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    return root, binary


def _managed_llama(tmp_path, monkeypatch, *, installed = "b9493"):
    """The ordinary case: ~/.unsloth/llama.cpp, no bundle anywhere."""
    install = tmp_path / "home" / "llama.cpp"
    binary = _write_llama_install(install, installed)
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    return install, binary


def test_llama_status_offers_no_update_for_a_bundled_root(tmp_path, monkeypatch):
    _bundled_llama(tmp_path, monkeypatch)
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is False
    assert st["supported"] is False
    assert st["immutable_runtime"] is True
    # The installed build still shows, so the About panel is not left blank.
    assert st["installed_tag"] == "b9493"


def test_llama_status_still_offers_an_update_for_a_managed_root(tmp_path, monkeypatch):
    _managed_llama(tmp_path, monkeypatch)
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is True
    assert st["supported"] is True
    assert st["latest_tag"] == "b9518"
    assert "immutable_runtime" not in st


def test_llama_apply_is_refused_for_a_bundled_root(tmp_path, monkeypatch):
    _bundled_llama(tmp_path, monkeypatch)
    plan = upd._plan_llama_phase()
    assert plan.get("spec") is None
    assert plan["skip_reason"] == "immutable_runtime"
    message = plan["refusal"]["message"]
    assert "ships inside the Unsloth app" in message
    assert "Updating the app" in message
    # And through the public entry point, which is what a direct POST reaches.
    result = upd.start_update()
    assert result["started"] is False
    assert result["reason"] == "immutable_runtime"


def test_llama_apply_is_planned_for_a_managed_root(tmp_path, monkeypatch):
    install, _ = _managed_llama(tmp_path, monkeypatch)
    plan = upd._plan_llama_phase()
    assert plan.get("refusal") is None, plan
    assert plan["spec"]["install_dir"] == install


def test_llama_backend_switch_is_unsupported_for_a_bundled_root(tmp_path, monkeypatch):
    _bundled_llama(tmp_path, monkeypatch)
    status = upd.get_backend_status()
    assert status["supported"] is False
    assert status["reason"] == "immutable_runtime"
    assert status["options"] == []
    # A direct switch POST is refused too, not merely unadvertised.
    result = upd.start_backend_switch("cpu")
    assert result["started"] is False
    assert result["reason"] == "immutable_runtime"


def test_llama_backend_switch_stays_supported_for_a_managed_root(tmp_path, monkeypatch):
    _managed_llama(tmp_path, monkeypatch)
    monkeypatch.setattr(
        upd,
        "_resolve_backends_for_host",
        lambda install_dir, *, force_refresh = False, published_repo = None: {
            "backends": [{"backend": "cpu", "available": True, "resolved_backend": "cpu"}]
        },
    )
    monkeypatch.setattr(upd, "latest_release_assets", lambda repo, force_refresh = False: {})
    status = upd.get_backend_status()
    assert status["supported"] is True
    assert status["reason"] is None


# ── whisper.cpp ──────────────────────────────────────────────────────────────


def _bundled_whisper(tmp_path, monkeypatch, *, installed = "v1.8.0"):
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    binary = _write_whisper_install(root / "whisper.cpp", installed)
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", str(root / "whisper.cpp"))
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        wfresh, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "v1.9.0"
    )
    return root, binary


def _managed_whisper(tmp_path, monkeypatch, *, installed = "v1.8.0"):
    install = tmp_path / "home" / "whisper.cpp"
    binary = _write_whisper_install(install, installed)
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        wfresh, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "v1.9.0"
    )
    return install, binary


def test_whisper_status_offers_no_update_for_a_bundled_root(tmp_path, monkeypatch):
    _bundled_whisper(tmp_path, monkeypatch)
    st = wupd.get_update_status(force_refresh = True)
    assert st["update_available"] is False
    assert st["supported"] is False
    assert st["immutable_runtime"] is True
    assert st["installed_tag"] == "v1.8.0"


def test_whisper_status_still_offers_an_update_for_a_managed_root(tmp_path, monkeypatch):
    _managed_whisper(tmp_path, monkeypatch)
    st = wupd.get_update_status(force_refresh = True)
    assert st["update_available"] is True
    assert st["supported"] is True
    assert "immutable_runtime" not in st


def test_whisper_chain_skips_a_bundled_root(tmp_path, monkeypatch):
    """Skipped rather than refused: whisper is the piggyback phase and must never
    be the reason a llama update cannot run."""
    _bundled_whisper(tmp_path, monkeypatch)
    plan = wupd.chained_phase_plan(force_refresh = True)
    assert plan["update_available"] is False
    assert plan["skip_reason"] == "immutable_runtime"
    assert plan["phase"] is None


def test_whisper_chain_still_plans_a_phase_for_a_managed_root(tmp_path, monkeypatch):
    install, _ = _managed_whisper(tmp_path, monkeypatch)
    plan = wupd.chained_phase_plan(force_refresh = True)
    assert plan["update_available"] is True
    assert plan["phase"]["install_dir"] == install


def test_whisper_repair_pairing_refuses_a_bundled_root(tmp_path, monkeypatch):
    _bundled_whisper(tmp_path, monkeypatch)
    plan = wupd.repair_pairing_plan()
    assert plan["update_available"] is False
    assert plan["skip_reason"] == "immutable_runtime"


def test_the_combined_item_offers_nothing_when_both_are_bundled(tmp_path, monkeypatch):
    """The single update item folds whisper in, so a guard on only one half would
    still light the banner up."""
    root = _make_runtime(tmp_path / "runtime")
    _pose_as_bundled(monkeypatch, root)
    llama_binary = _write_llama_install(root / "llama.cpp", "b9493")
    whisper_binary = _write_whisper_install(root / "whisper.cpp", "v1.8.0")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root / "llama.cpp"))
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", str(root / "whisper.cpp"))
    monkeypatch.setattr(upd, "_find_binary", lambda: llama_binary)
    monkeypatch.setattr(wupd, "_find_binary", lambda: whisper_binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    monkeypatch.setattr(
        wfresh, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "v1.9.0"
    )

    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is False
    assert st["llama_update_available"] is False
    assert st["update_component"] is None
    assert (st["whisper"] or {}).get("skip_reason") == "immutable_runtime"


# ── the other place a refresh was advertised ─────────────────────────────────


def test_the_startup_stale_warning_names_a_remedy_that_exists(tmp_path, monkeypatch):
    """main.py's boot probe prints the freshness one-liner, and its remedy was
    "Run `unsloth studio update`" -- which for a bundled llama.cpp is an
    instruction with nothing to act on. The observation stays; the remedy moves."""
    info = {"installed_tag": "b9190", "latest_tag": "b9300", "age_days": 5}
    plain = freshness.format_stale_warning(info)
    assert "unsloth studio update" in plain
    bundled_message = freshness.format_stale_warning(info, bundled = True)
    assert "unsloth studio update" not in bundled_message
    assert "update the app" in bundled_message
    # Both still say what was observed, or the warning is worse than useless.
    for message in (plain, bundled_message):
        assert "b9190" in message and "b9300" in message and "5 days" in message


# ── the OXC validator's node_modules ─────────────────────────────────────────


def _run_oxc_with_captured_env(monkeypatch, tmp_path):
    """Run one _run_oxc_batch with the Node subprocess stubbed, and return the env
    it would have handed the child."""
    import core.data_recipe.local_callable_validators as val

    captured: dict = {}

    class _Proc:
        returncode = 0
        stdout = "[]"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = dict(kwargs.get("env") or {})
        return _Proc()

    monkeypatch.setattr(val.subprocess, "run", _fake_run)
    monkeypatch.setattr(val, "resolve_node_executable", lambda: "/usr/bin/node")
    monkeypatch.setattr(val, "oxc_validator_tmp_root", lambda: tmp_path / "oxc-tmp")
    val._run_oxc_batch(
        node_lang = "ts", validation_mode = "syntax", code_shape = "module", code_values = [""]
    )
    return captured


def test_node_path_is_dropped_when_there_is_no_bundle(tmp_path, monkeypatch):
    """The deliberate pop stays: NODE_PATH decides which code the validator loads,
    so an inherited value must never survive into the child."""
    monkeypatch.delenv("UNSLOTH_BUNDLED_SITE_PACKAGES", raising = False)
    monkeypatch.setenv("NODE_PATH", "/tmp/attacker/modules")
    captured = _run_oxc_with_captured_env(monkeypatch, tmp_path)
    assert "NODE_PATH" not in captured["env"]
    assert "UNSLOTH_OXC_NODE_MODULES" not in captured["env"]


def test_node_path_points_at_the_bundled_modules_when_there_is_a_bundle(
    tmp_path, monkeypatch
):
    """Replaced, not kept: the inherited value is still discarded -- it is
    overwritten with the one directory inside the signed bundle."""
    root = _make_runtime(tmp_path / "runtime")
    (root / "oxc-node-modules").mkdir()
    _pose_as_bundled(monkeypatch, root)
    monkeypatch.setenv("NODE_PATH", "/tmp/attacker/modules")
    captured = _run_oxc_with_captured_env(monkeypatch, tmp_path)
    expected = str((root / "oxc-node-modules").resolve())
    assert captured["env"]["NODE_PATH"] == expected
    # NODE_PATH is not consulted by ESM's node_modules walk, so the runner also
    # needs the directory itself to resolve `oxc-parser` through createRequire.
    assert captured["env"]["UNSLOTH_OXC_NODE_MODULES"] == expected


def test_a_bundle_without_the_oxc_modules_falls_back_to_the_pop(tmp_path, monkeypatch):
    root = _make_runtime(tmp_path / "runtime")  # no oxc-node-modules/
    _pose_as_bundled(monkeypatch, root)
    monkeypatch.setenv("NODE_PATH", "/tmp/attacker/modules")
    captured = _run_oxc_with_captured_env(monkeypatch, tmp_path)
    assert "NODE_PATH" not in captured["env"]
    assert "UNSLOTH_OXC_NODE_MODULES" not in captured["env"]


def test_the_runner_reads_the_handover_variable(tmp_path):
    """The Python half is useless without the JS half: validate.mjs has to take the
    directory from the variable, because a static `import "oxc-parser"` cannot be
    reached by NODE_PATH at all."""
    runner = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "data_recipe"
        / "oxc-validator"
        / "validate.mjs"
    )
    source = runner.read_text(encoding = "utf-8")
    assert "UNSLOTH_OXC_NODE_MODULES" in source
    assert "createRequire" in source
