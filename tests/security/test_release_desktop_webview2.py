# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The WebView2 bootstrapper embedded in the Windows installer has to be pinned.

`bundle > windows > webviewInstallMode: embedBootstrapper` makes tauri-bundler
put Microsoft's Evergreen WebView2 bootstrapper inside the NSIS installer -- the
same file the updater ships as its payload. tauri-bundler fetched it from a
rotating go.microsoft.com/fwlink redirect and verified nothing, so a
Microsoft-supplied binary reached users' machines inside a signed installer on
TLS alone, and two builds of the same commit did not contain the same bytes.
Every other file the Windows bundler downloads (NSIS 3.11, nsis_tauri_utils.dll)
is hash-checked; this one was the outlier.

The fix mirrors the AppImage runtime pin: fetch from an immutable URL, verify a
pinned SHA-256 before the file can be used, fail closed. Because no
`webviewInstallMode` variant accepts a local file, the bundler is handed the
verified copy through its own cache -- tauri-bundler 2.8.1 skips the download when
`<tools dir>/MicrosoftEdgeWebview2Setup.exe` already exists -- and a post-build
step reads the path back out of the rendered NSIS script to prove which bytes the
installer actually embedded.

Note what is pinned: the ~1.8MB bootstrapper (Microsoft Edge Update Setup
1.3.257.13), not the WebView2 runtime. It still fetches the current Evergreen
runtime at install time, so users keep receiving WebView2 security updates.
`fixedRuntime` would pin the runtime itself and end that; these tests exist partly
so that swap cannot happen quietly.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
TAURI_CONF = REPO_ROOT / "studio" / "src-tauri" / "tauri.conf.json"
WINDOWS_CONF = REPO_ROOT / "studio" / "src-tauri" / "tauri.windows.conf.json"
NSIS_TEMPLATE = REPO_ROOT / "studio" / "src-tauri" / "windows" / "installer.nsi"

# The bootstrapper this release is pinned to. Recorded from the artifact itself:
#
#   curl -sSIL 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' | grep -i '^location'
#   curl -fsSL '<resolved url>' | sha256sum
#
# Bump this together with both copies in release-desktop.yml and the copy inside
# its "Verify desktop updater and Linux package config" guard.
EXPECTED_DIGEST = "be695eb3732a94e181f008ab5cf6ee650f8644676e87f9e02b6ab0d02f2ea08e"

PIN_STEP = "Pin Windows WebView2 bootstrapper"
VERIFY_STEP = "Verify the embedded WebView2 bootstrapper"
GUARD_STEP = "Verify desktop updater and Linux package config"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason = "node is not installed")


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps():
    return _workflow()["jobs"]["build"]["steps"]


def _step(name: str):
    return next(step for step in _steps() if step.get("name") == name)


def _step_index(name: str) -> int:
    names = [step.get("name") for step in _steps()]
    assert name in names, f"build job has no step named {name!r}; steps are {names}"
    return names.index(name)


def test_the_bootstrapper_is_pinned_by_immutable_url_and_digest():
    step = _step(PIN_STEP)
    url = step["env"]["WEBVIEW2_BOOTSTRAPPER_URL"]

    # The fwlink the bundler used resolves to a per-version path under this host.
    # Pinning the redirect instead would pin nothing: Microsoft repoints it.
    assert re.fullmatch(
        r"https://msedge\.sf\.dl\.delivery\.mp\.microsoft\.com/filestreamingservice/files/"
        r"[0-9a-f-]{36}/MicrosoftEdgeWebview2Setup\.exe",
        url,
    ), url
    assert "go.microsoft.com" not in url
    assert step["env"]["WEBVIEW2_BOOTSTRAPPER_SHA256"] == EXPECTED_DIGEST
    assert len(EXPECTED_DIGEST) == 64


def test_both_windows_steps_agree_on_the_digest_and_name_their_shell():
    for name in (PIN_STEP, VERIFY_STEP):
        step = _step(name)
        assert step["if"] == "matrix.platform == 'windows-latest'", name
        # -MaximumRetryCount is PowerShell 6+ only, and a release download must
        # not inherit its interpreter from a runner default.
        assert step["shell"] == "pwsh", name
        assert step["env"]["WEBVIEW2_BOOTSTRAPPER_SHA256"] == EXPECTED_DIGEST, name


def test_a_digest_mismatch_stops_the_release_before_the_bundler_sees_the_file():
    run = _step(PIN_STEP)["run"]

    # Staged, verified, and only then moved to the path the bundler treats as
    # "already cached". A partial or substituted download must never sit there.
    assert run.index("Get-FileHash") < run.index("Move-Item")
    assert "if ($actual -ne $expected)" in run
    assert "Remove-Item -Force $dest" in run
    assert "::error::WebView2 bootstrapper digest mismatch" in run
    assert run.count("exit 1") >= 2
    # An unset LOCALAPPDATA would silently seed the wrong directory.
    assert "if (-not $env:LOCALAPPDATA)" in run
    # A warm runner image can already hold an unverified copy at the seeded path,
    # and that copy would win the bundler's existence check.
    assert "Move-Item -Force" in run


def test_the_pin_is_checked_against_what_the_installer_embedded():
    run = _step(VERIFY_STEP)["run"]

    # Checking the rendered NSIS script rather than the seeded file is the point:
    # it is the only evidence of which bytes makensis compiled in, and it survives
    # a tauri-bundler change that moves or ignores the cache.
    assert "installer.nsi" in run
    assert "INSTALLWEBVIEW2MODE" in run
    assert "embedBootstrapper" in run
    assert "WEBVIEW2BOOTSTRAPPERPATH" in run
    assert "Get-FileHash" in run
    assert "$env:WEBVIEW2_BOOTSTRAPPER_PATH" in run
    assert run.count("exit 1") >= 6
    code = [line for line in run.splitlines() if not line.strip().startswith("#")]
    assert not any("-Recurse" in line for line in code), (
        "the target directory is far too large to walk"
    )


def test_the_pin_runs_before_the_windows_build_and_the_proof_runs_after():
    pin = _step_index(PIN_STEP)
    build = _step_index("Build Windows app")
    verify = _step_index(VERIFY_STEP)
    # Before the 20-minute Defender scan, so a miss fails fast.
    scan = _step_index("Scan Windows bundles with Defender")

    assert pin < build < verify < scan


def test_the_pin_step_exports_the_seeded_path_for_the_proof_step():
    run = _step(PIN_STEP)["run"]
    assert "WEBVIEW2_BOOTSTRAPPER_PATH=$seeded" in run
    assert "$env:GITHUB_ENV" in run
    # %LOCALAPPDATA%\tauri is dirs::cache_dir()/tauri, which is where
    # tauri-bundler looks when bundle > useLocalToolsDir is false.
    assert 'Join-Path $env:LOCALAPPDATA "tauri"' in run
    assert '"MicrosoftEdgeWebview2Setup.exe"' in run


def test_the_bundler_tools_directory_is_not_moved_out_from_under_the_pin():
    config = json.loads(TAURI_CONF.read_text(encoding = "utf-8"))
    overlay = json.loads(WINDOWS_CONF.read_text(encoding = "utf-8"))

    # useLocalToolsDir: true would move the cache to target/.tauri, so the pin
    # step would seed a directory the bundler never reads and it would download
    # its own copy again.
    for source in (config, overlay):
        assert source.get("bundle", {}).get("useLocalToolsDir") in (None, False)


def test_the_config_keeps_the_install_mode_the_pin_was_built_for():
    config = json.loads(TAURI_CONF.read_text(encoding = "utf-8"))
    mode = config["bundle"]["windows"]["webviewInstallMode"]

    # downloadBootstrapper moves the unverified fetch onto the user's machine at
    # install time, where nothing here can pin it. fixedRuntime pins the WebView2
    # runtime version itself, which stops users receiving Evergreen security
    # updates -- a product decision, not a cleanup, so it must not arrive quietly.
    assert mode["type"] == "embedBootstrapper"
    assert mode["silent"] is True
    assert "webviewFixedRuntimePath" not in json.dumps(config)


def test_the_windows_config_overlay_cannot_change_the_install_mode():
    # tauri.windows.conf.json is merged over tauri.conf.json on the Windows leg,
    # so an override here would defeat the check above without touching it.
    assert "webviewInstallMode" not in WINDOWS_CONF.read_text(encoding = "utf-8")


def test_the_nsis_template_still_embeds_the_path_the_proof_step_reads():
    # The proof step reads WEBVIEW2BOOTSTRAPPERPATH out of the rendered script and
    # hashes it. That is only evidence if the template embeds that exact file.
    template = NSIS_TEMPLATE.read_text(encoding = "utf-8")
    assert '!define WEBVIEW2BOOTSTRAPPERPATH "{{webview2_bootstrapper_path}}"' in template
    assert '!define INSTALLWEBVIEW2MODE "{{install_webview2_mode}}"' in template
    assert (
        'File "/oname=$TEMP\\MicrosoftEdgeWebview2Setup.exe" "${WEBVIEW2BOOTSTRAPPERPATH}"'
        in template
    )


def test_the_digest_reaches_the_build_inputs_record():
    # "Record build inputs" scrapes ^[A-Z_]*(_URL|_SHA256): "..." out of this
    # workflow into build-inputs.json, so the pin is published with the release
    # rather than living only in a log that expires. Both keys have to keep that
    # shape for the scrape to see them.
    pinned = dict(
        re.findall(
            r'^\s*([A-Z][A-Z0-9_]*(?:_URL|_SHA256)):\s*"([^"]+)"',
            WORKFLOW.read_text(encoding = "utf-8"),
            re.M,
        )
    )
    assert pinned["WEBVIEW2_BOOTSTRAPPER_SHA256"] == EXPECTED_DIGEST
    assert pinned["WEBVIEW2_BOOTSTRAPPER_URL"].endswith("/MicrosoftEdgeWebview2Setup.exe")


def _run_guard(tmp_path: Path, mutate = None):
    """Run the in-workflow guard against a copy of the three files it reads."""
    workflow_text = WORKFLOW.read_text(encoding = "utf-8")
    config_text = TAURI_CONF.read_text(encoding = "utf-8")
    overlay_text = WINDOWS_CONF.read_text(encoding = "utf-8")
    if mutate is not None:
        workflow_text, config_text, overlay_text = mutate(
            workflow_text, config_text, overlay_text
        )

    workflow_path = tmp_path / ".github" / "workflows" / "release-desktop.yml"
    workflow_path.parent.mkdir(parents = True, exist_ok = True)
    workflow_path.write_text(workflow_text, encoding = "utf-8")
    conf_dir = tmp_path / "studio" / "src-tauri"
    conf_dir.mkdir(parents = True, exist_ok = True)
    (conf_dir / "tauri.conf.json").write_text(config_text, encoding = "utf-8")
    (conf_dir / "tauri.windows.conf.json").write_text(overlay_text, encoding = "utf-8")

    return subprocess.run(
        ["bash", "-c", _step(GUARD_STEP)["run"]],
        cwd = tmp_path,
        env = {
            "PATH": Path(shutil.which("node")).parent.as_posix() + ":/usr/bin:/bin",
            "DESKTOP_RELEASE_NOTES": _workflow()["env"]["DESKTOP_RELEASE_NOTES"],
        },
        text = True,
        capture_output = True,
        check = False,
    )


@needs_node
def test_the_guard_passes_on_the_committed_files(tmp_path):
    result = _run_guard(tmp_path)
    assert result.returncode == 0, result.stderr


@needs_node
def test_the_guard_rejects_a_downgraded_install_mode(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, config, overlay: (
            workflow,
            config.replace('"embedBootstrapper"', '"downloadBootstrapper"'),
            overlay,
        ),
    )
    assert result.returncode != 0
    assert "webviewInstallMode must stay" in result.stderr


@needs_node
def test_the_guard_rejects_a_repointed_digest(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, config, overlay: (
            workflow.replace(
                f'WEBVIEW2_BOOTSTRAPPER_SHA256: "{EXPECTED_DIGEST}"',
                'WEBVIEW2_BOOTSTRAPPER_SHA256: "' + "0" * 64 + '"',
                1,
            ),
            config,
            overlay,
        ),
    )
    assert result.returncode != 0
    assert "must pin the WebView2 bootstrapper SHA-256 digest" in result.stderr


@needs_node
def test_the_guard_rejects_the_rotating_fwlink_url(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, config, overlay: (
            re.sub(
                r'WEBVIEW2_BOOTSTRAPPER_URL: "[^"]+"',
                'WEBVIEW2_BOOTSTRAPPER_URL: '
                '"https://go.microsoft.com/fwlink/p/?LinkId=2124703"',
                workflow,
            ),
            config,
            overlay,
        ),
    )
    assert result.returncode != 0
    assert "immutable" in result.stderr


@needs_node
def test_the_guard_rejects_seeding_the_cache_before_verifying(tmp_path):
    def mutate(workflow, config, overlay):
        lines = workflow.split("\n")
        start = lines.index(f"      - name: {PIN_STEP}")
        move = next(
            index
            for index in range(start, len(lines))
            if "Move-Item -Force -Path $dest" in lines[index]
        )
        hashed = next(
            index for index in range(start, len(lines)) if "Get-FileHash" in lines[index]
        )
        lines[hashed], lines[move] = lines[move], lines[hashed]
        return "\n".join(lines), config, overlay

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert "before moving it" in result.stderr


@needs_node
def test_the_guard_rejects_dropping_the_post_build_proof(tmp_path):
    # The step definition, not the copies of its name inside the guard's own JS
    # (which appears earlier in the file).
    definition = f"\n      - name: {VERIFY_STEP}\n"

    def mutate(workflow, config, overlay):
        assert workflow.count(definition) == 1
        return workflow.replace(definition, "\n      - name: Something else\n"), config, overlay

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert VERIFY_STEP in result.stderr


@needs_node
def test_the_guard_rejects_an_overlay_that_overrides_the_install_mode(tmp_path):
    def mutate(workflow, config, overlay):
        parsed = json.loads(overlay)
        parsed["bundle"]["windows"]["webviewInstallMode"] = {"type": "skip"}
        return workflow, config, json.dumps(parsed)

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert "must not override" in result.stderr
