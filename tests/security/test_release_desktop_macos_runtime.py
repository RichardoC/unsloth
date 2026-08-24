# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The signed macOS release now ships a whole runtime inside the app bundle.

Unsloth.app/Contents/Resources/runtime holds CPython, ~262 Python
distributions, llama.cpp, whisper.cpp, Node and the OXC node_modules: roughly
700 Mach-O files that tauri's bundler seals as data and never signs.
Notarization rejects an unsigned Mach-O, so the macOS leg had to be turned
inside out:

    assemble payload -> tauri build (unsigned) -> inject -> sign every Mach-O
    -> sign the app -> rebuild the updater tarball and the .dmg -> notarize

Every step of that is ordering, and ordering fails quietly. Signing before
injecting produces a .dmg whose signature is invalid; rebuilding the image
before signing the app produces one that Gatekeeper refuses; publishing
tauri's updater tarball pushes shipped installs an unsigned app macOS will not
launch. None of it shows up in CI -- it shows up in somebody's Downloads
folder. So the order is asserted here, the in-workflow guard that enforces it
at release time is executed here, and the two steps that do the real work are
run against a synthetic app bundle with stubbed Apple tooling, so that "it
signs every Mach-O" is a test rather than a claim.

What these tests cannot prove is anything that needs a Mac: that codesign
accepts these flags, that Apple's notary service accepts the result, or that
the payload's binaries run under the hardened runtime. Those wait for a real
dispatch.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
ENTITLEMENTS = REPO_ROOT / "studio" / "src-tauri" / "Entitlements.plist"
TAURI_CONF = REPO_ROOT / "studio" / "src-tauri" / "tauri.conf.json"
WINDOWS_CONF = REPO_ROOT / "studio" / "src-tauri" / "tauri.windows.conf.json"

GUARD_STEP = "Verify desktop updater and Linux package config"
ASSEMBLE_STEP = "Assemble the bundled runtime payload"
LOCK_STEP = "Check the macOS bundle lock is not stale"
BUILD_STEP = "Build macOS app"
LOCATE_STEP = "Locate the built macOS bundle"
INJECT_STEP = "Inject the bundled runtime into the macOS app"
SIGN_STEP = "Sign the bundled runtime and the macOS app"
VERIFY_STEP = "Verify every Mach-O in the macOS bundle is signed"
UPDATER_STEP = "Rebuild the macOS updater artifact from the signed app"
DMG_STEP = "Rebuild the macOS disk image around the signed app"
NOTARIZE_STEP = "Notarize final macOS disk image"

# The order the whole design rests on.
MACOS_CHAIN = (
    LOCK_STEP,
    ASSEMBLE_STEP,
    BUILD_STEP,
    LOCATE_STEP,
    INJECT_STEP,
    SIGN_STEP,
    VERIFY_STEP,
    UPDATER_STEP,
    DMG_STEP,
    NOTARIZE_STEP,
)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason = "node is not installed")
needs_file = pytest.mark.skipif(
    shutil.which("file") is None, reason = "file(1) is the verification step's oracle"
)


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps(job = "build"):
    return _workflow()["jobs"][job]["steps"]


def _step(name: str, job = "build"):
    steps = _steps(job)
    names = [step.get("name") for step in steps]
    assert name in names, f"the {job} job has no step named {name!r}; steps are {names}"
    return steps[names.index(name)]


def _index(name: str, job = "build") -> int:
    return [step.get("name") for step in _steps(job)].index(name)


# ──────────────────────────────────────────────── the order, and who is gated

def test_the_payload_goes_in_before_anything_is_signed():
    """scripts/inject_macos_runtime.sh refuses an app that is already signed.

    That refusal is the whole reason this chain exists in this order, and the
    chain is what these indices pin: injection after the build, signing after
    the injection, verification after the signing, both repackaging steps after
    that, and notarization last.
    """
    for earlier, later in zip(MACOS_CHAIN, MACOS_CHAIN[1:]):
        assert _index(earlier) < _index(later), f"{earlier} must run before {later}"


def test_the_bundle_lock_freshness_check_gates_the_expensive_work():
    step = _step(LOCK_STEP)
    assert "scripts/gen_macos_bundle_lock.sh --check" in step["run"]
    # Cheap, and before both the ~2 GiB payload download and the Rust compile.
    assert _index(LOCK_STEP) < _index(ASSEMBLE_STEP) < _index(BUILD_STEP)
    # And after the credential check, which is cheaper still.
    assert _index("Check Apple notarization credentials") < _index(LOCK_STEP)


def test_every_new_macos_step_only_runs_on_the_macos_leg():
    for name in MACOS_CHAIN:
        if name == BUILD_STEP or name == NOTARIZE_STEP:
            continue  # pre-existing steps, asserted by their own tests
        assert _step(name).get("if") == "matrix.platform == 'macos-latest'", name
    for name in (LOCK_STEP, ASSEMBLE_STEP, "Install the pinned uv"):
        assert _step(name).get("if") == "matrix.platform == 'macos-latest'", name


def test_tauri_is_handed_no_apple_credentials():
    """If tauri signs the app, the injection that follows invalidates it.

    tauri-bundler signs, notarizes and staples the .app and then builds the .dmg
    and the updater tarball from it. All of that has to happen after the payload
    is in, so the action gets the updater key and nothing else.
    """
    env = _step(BUILD_STEP).get("env", {})
    for secret in ("APPLE_SIGNING_IDENTITY", "APPLE_ID", "APPLE_PASSWORD", "APPLE_TEAM_ID"):
        assert secret not in env, f"{secret} would make tauri sign before the injection"
    # The updater key stays: the tarball's path has to exist for the rebuild
    # step to overwrite, and its password must be empty rather than unset or the
    # CLI prompts and the job hangs.
    assert env["TAURI_SIGNING_PRIVATE_KEY"] == "${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}"
    assert env["TAURI_SIGNING_PRIVATE_KEY_PASSWORD"] == ""


def test_the_injection_leaves_tauris_disk_image_alone():
    run = _step(INJECT_STEP)["run"]
    assert "scripts/inject_macos_runtime.sh" in run
    assert "--app" in run
    live = [line for line in run.splitlines() if not line.strip().startswith("#")]
    assert not any("--dmg" in line for line in live), (
        "injecting into tauri's image would leave that image holding an unsigned app; "
        "the image is rebuilt around the signed app instead"
    )


def test_nothing_after_the_injection_reads_the_staging_payload():
    """The staging copy is deleted after the injection, and that has to stay safe.

    This leg is disk bound: it holds the app, a writable image grown to fit it, a
    compressed image, an updater tarball and a full extraction of that tarball.
    Freeing the ~2.5 GiB staging copy is what makes that fit, and it is only safe
    because everything downstream reads the app bundle instead -- including the
    provenance record, deliberately, so that what is recorded is what shipped.
    """
    inject = _step(INJECT_STEP)["run"]
    assert 'rm -rf "$RUNNER_TEMP/runtime"' in inject
    later = _steps()[_index(INJECT_STEP) + 1:]
    for step in later:
        body = (step.get("run") or "") + yaml.safe_dump(step.get("env", {}))
        assert "RUNNER_TEMP/runtime\"" not in body, step.get("name")
        assert "RUNNER_TEMP/runtime/" not in body, step.get("name")
        assert "RUNNER_TEMP}}/runtime" not in body, step.get("name")


def test_an_incomplete_payload_never_reaches_a_signature():
    """--skip-diffusers-pin and --with-sd-cpp both record themselves in the manifest.

    A payload assembled with a component missing is fine for a dev build and is
    not something to sign, notarize and publish.
    """
    run = _step(ASSEMBLE_STEP)["run"]
    assert "scripts/build_macos_runtime.sh --out" in run
    assert 'manifest["incomplete"]' in run
    assert "--skip-diffusers-pin" not in run.split("<<'PY'")[0], (
        "the release must assemble the complete payload"
    )


# ────────────────── the payload's backend is the tag's, and the stamp agrees
#
# The macOS payload builds `unsloth` from the checkout rather than installing the
# published wheel, which moved the backend from something a user's machine
# resolved to something the release decided. `pypi_version` is what the app is
# told its backend is (UNSLOTH_DESKTOP_BACKEND_VERSION, compiled in), and it only
# ever stamped that env var -- it never influenced the payload. So the two could
# disagree, in a signed and notarized app, with no venv to upgrade out of it.
#
# These run the real step against a stubbed builder and a real git repo, so the
# commit comparison is exercised rather than asserted about.

def _payload_entry(commit: str, **overrides) -> dict:
    entry = {
        "name": "unsloth",
        "version": "2026.8.18",
        "provenance": "local-checkout",
        "index_verified": False,
        "wheel": "unsloth-2026.8.18-py3-none-any.whl",
        "wheel_sha256": "a" * 64,
        "wheel_tag": "py3-none-any",
        "repo_commit": commit,
        "worktree_dirty": False,
        "replaced_index_version": "2026.8.18",
    }
    entry.update(overrides)
    return entry


def _payload_manifest(entry: dict | None, *, wheel_commit: str, incomplete = ()) -> dict:
    return {
        "incomplete": list(incomplete),
        "python": {"version": "3.13.14"},
        "distribution_count": 262,
        "total_bytes": 2_600_000_000,
        "components": {"unsloth_local_wheel": {"repo_commit": wheel_commit}},
        "local_provenance": {
            "index_verified": False,
            "distributions": ([entry] if entry else []),
        },
    }


def _run_assemble_checks(
    tmp_path: Path,
    *,
    manifest_for = None,
    typed = "2026.8.18",
    resolved = "2026.8.4",
):
    """Run the assemble step with a stubbed builder and a one-commit git repo.

    The stub stands in for scripts/build_macos_runtime.sh, whose real output is a
    2.6 GiB download; what is under test is what the step does with the manifest
    that lands beside it. `git rev-parse HEAD` is the real command against a real
    repo, so "the payload was built from the commit being released" is compared
    the same way the release compares it.
    """
    work = tmp_path / "work"
    (work / "scripts").mkdir(parents = True)
    git_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    (work / "seed").write_text("x", encoding = "utf-8")
    for argv in (["init", "-q"], ["add", "seed"], ["commit", "-qm", "seed"]):
        subprocess.run(["git", *argv], cwd = work, env = git_env, check = True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd = work, env = git_env,
        text = True, capture_output = True, check = True,
    ).stdout.strip()

    manifest_for = manifest_for or (
        lambda sha: _payload_manifest(_payload_entry(sha), wheel_commit = sha)
    )
    runner_temp = tmp_path / "runner-temp"
    (runner_temp / "runtime").mkdir(parents = True)
    (runner_temp / "runtime" / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest_for(commit)), encoding = "utf-8"
    )
    (work / "scripts" / "build_macos_runtime.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding = "utf-8"
    )

    result = subprocess.run(
        ["bash", "-c", _step(ASSEMBLE_STEP)["run"]],
        cwd = work,
        env = {
            **git_env,
            "RUNNER_TEMP": str(runner_temp),
            "UV": "/nonexistent/uv",
            "INPUT_PYPI_VERSION": typed,
            "RESOLVED_PYPI_VERSION": resolved,
        },
        text = True,
        capture_output = True,
        check = False,
    )
    return result, commit


def test_the_tag_commit_comes_from_the_checkout_and_not_the_dispatch_ref():
    """GITHUB_SHA is the dispatch ref's commit; the build checks out the tag.

    Comparing against GITHUB_SHA would pass while the payload was built from
    whatever main happened to be, which is the exact thing being ruled out.
    """
    run = _step(ASSEMBLE_STEP)["run"]
    assert 'TAG_COMMIT="$(git rev-parse HEAD)"' in run
    # Code only: the step names GITHUB_SHA in a comment saying why it is wrong here.
    code = [
        line for line in run.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if "GITHUB_SHA" in line], code


def test_a_payload_built_from_the_released_tag_passes(tmp_path):
    result, commit = _run_assemble_checks(tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"unsloth 2026.8.18 built from {commit[:12]}" in result.stdout


def test_a_payload_built_from_another_commit_stops_the_release(tmp_path):
    result, _ = _run_assemble_checks(
        tmp_path,
        manifest_for = lambda sha: _payload_manifest(
            _payload_entry("b" * 40), wheel_commit = "b" * 40
        ),
    )
    assert result.returncode != 0
    assert "not the tag being released" in result.stderr


def test_a_dirty_worktree_stops_the_release(tmp_path):
    """The recorded commit stops describing what shipped, so the release is
    unreproducible by construction -- fine for a dev build, never for a signed one."""
    result, _ = _run_assemble_checks(
        tmp_path,
        manifest_for = lambda sha: _payload_manifest(
            _payload_entry(sha, worktree_dirty = True), wheel_commit = sha
        ),
    )
    assert result.returncode != 0
    assert "dirty worktree" in result.stderr


def test_a_stamp_the_bundle_does_not_contain_stops_the_release(tmp_path):
    """The gap this closes: pypi_version stamped the binary and never decided the
    payload, so a mismatched dispatch shipped an app that reports a backend
    version it does not hold -- and cannot install, because it has no venv."""
    result, _ = _run_assemble_checks(tmp_path, typed = "2026.9.0")
    assert result.returncode != 0
    assert "2026.9.0" in result.stderr and "2026.8.18" in result.stderr
    assert "does not contain" in result.stderr


def test_a_published_wheel_in_the_payload_stops_the_release(tmp_path):
    result, _ = _run_assemble_checks(
        tmp_path,
        manifest_for = lambda sha: _payload_manifest(None, wheel_commit = sha),
    )
    assert result.returncode != 0
    assert "records no locally built unsloth" in result.stderr


def test_a_payload_resolved_from_an_index_stops_the_release(tmp_path):
    result, _ = _run_assemble_checks(
        tmp_path,
        manifest_for = lambda sha: _payload_manifest(
            _payload_entry(sha, provenance = "index", index_verified = True),
            wheel_commit = sha,
        ),
    )
    assert result.returncode != 0
    assert "not built from the checkout" in result.stderr


def test_the_two_manifest_records_of_the_local_wheel_must_agree(tmp_path):
    result, _ = _run_assemble_checks(
        tmp_path,
        manifest_for = lambda sha: _payload_manifest(
            _payload_entry(sha), wheel_commit = "c" * 40
        ),
    )
    assert result.returncode != 0
    assert "disagree" in result.stderr


def test_a_blank_pypi_version_treats_the_minimum_as_a_floor(tmp_path):
    """Blank resolves to MIN_DESKTOP_BACKEND_VERSION, which is a floor rather than
    a claim about what ships, so a newer payload passes -- and says so."""
    result, _ = _run_assemble_checks(tmp_path, typed = "", resolved = "2026.8.4")
    assert result.returncode == 0, result.stderr
    assert "pypi_version was blank" in result.stdout


def test_a_blank_pypi_version_still_rejects_a_backend_below_the_floor(tmp_path):
    """A payload under the compiled-in minimum is a backend the app itself rejects."""
    result, _ = _run_assemble_checks(tmp_path, typed = "", resolved = "2026.9.1")
    assert result.returncode != 0
    assert "below the desktop minimum" in result.stderr


def test_the_floor_comparison_fails_closed_on_a_version_it_cannot_parse(tmp_path):
    result, _ = _run_assemble_checks(tmp_path, typed = "", resolved = "not-a-version")
    assert result.returncode != 0
    assert "cannot compare" in result.stderr


def test_the_notarization_step_keeps_its_contract():
    """It is the release's only notarization, and it must still read tauri's path.

    The rebuild steps overwrite the .dmg in place at the path tauri reported, so
    this step and "Stage release assets" need no knowledge of any of it.
    """
    step = _step(NOTARIZE_STEP)
    assert step["env"]["ARTIFACT_PATHS"] == "${{ steps.build_macos.outputs.artifactPaths }}"
    assert isinstance(step["timeout-minutes"], int)
    assert _index(NOTARIZE_STEP) < _index("Stage release assets")
    # One notarization, not two: nothing else may submit to the notary service.
    submitting = [
        step.get("name") for step in _steps()
        if "notarytool submit" in (step.get("run") or "")
    ]
    assert submitting == [NOTARIZE_STEP], submitting


def test_the_macos_leg_gets_its_own_timeout():
    """A shared cap would have to be the macOS one, and that is now three hours."""
    build = _workflow()["jobs"]["build"]
    assert build["timeout-minutes"] == "${{ matrix.timeout_minutes }}"
    legs = {entry["artifact"]: entry["timeout_minutes"] for entry in build["strategy"]["matrix"]["include"]}
    assert legs["macos-aarch64"] > legs["linux-x64"]
    assert legs["macos-aarch64"] > legs["windows-x64"]
    # desktop-dev-build.yml budgets 90 minutes for the payload and the build with
    # no signing and no notarization at all, so the release leg needs more.
    dev = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "desktop-dev-build.yml").read_text(encoding = "utf-8")
    )
    assert legs["macos-aarch64"] > dev["jobs"]["dmg"]["timeout-minutes"]

    # And the publisher has to outlast the leg it waits for, or it kills itself
    # while a healthy build is still running.
    publish = _workflow()["jobs"]["publish-release"]
    assert publish["timeout-minutes"] > legs["macos-aarch64"]
    wait = _step("Wait for the build matrix", "publish-release")["run"]
    deadline = int(
        re.search(r"^\s*DEADLINE=\$\(\( \$\(date \+%s\) \+ (\d+) \* 60 \)\)", wait, re.M).group(1)
    )
    assert legs["macos-aarch64"] < deadline < publish["timeout-minutes"]


# ──────────────────────────────────────────────────────────── the entitlements

def test_the_app_is_not_signed_with_library_validation_relief():
    """The one entitlement the bundled runtime made removable.

    It was there for "Python/venv libraries not signed by us", from when the
    stack was installed at first run into ~/.unsloth. That reasoning never
    applied to this executable: entitlements govern the process they are signed
    into, and the Python stack is a child process with its own signature. The
    app itself is a Rust/Tauri binary that loads Apple frameworks and its own
    statically linked code, dlopen()s nothing, has no sidecar and no
    Contents/Frameworks -- so the relief only ever widened what could be loaded
    into the highest-value process in the bundle.
    """
    entitlements = plistlib.loads(ENTITLEMENTS.read_bytes())
    assert "com.apple.security.cs.disable-library-validation" not in entitlements
    # The rest is unchanged. allow-unsigned-executable-memory stays: WebKit's
    # JavaScript engine needs it, as it does in every WKWebView host.
    assert entitlements == {
        "com.apple.security.cs.allow-unsigned-executable-memory": True,
        "com.apple.security.network.client": True,
        "com.apple.security.device.audio-input": True,
    }


def test_the_reason_the_key_is_gone_is_written_down_where_it_was():
    text = ENTITLEMENTS.read_text(encoding = "utf-8")
    assert "disable-library-validation is deliberately NOT here" in text
    assert "release-desktop.yml" in text, (
        "the plist has to say where the payload's own entitlements live, or the next "
        "reader concludes the payload is signed with this one"
    )


def test_the_payload_gets_the_relief_the_app_gave_up():
    """The bundled interpreter needs it, and for a reason the app never had.

    Inside a notarized bundle the hardened runtime is mandatory, and under it a
    process may only load libraries signed by the same team or by Apple.
    Everything shipped in the payload passes that; what a user's own work
    produces does not -- torch.compile writes a .so and dlopen()s it, a
    `pip install` of anything with a native extension lands an unsigned .so, and
    node loads .node addons the same way.
    """
    run = _step(SIGN_STEP)["run"]
    assert "<key>com.apple.security.cs.disable-library-validation</key>" in run
    assert "<key>com.apple.security.cs.allow-unsigned-executable-memory</key>" in run
    # Not the app's plist: the two sets are different on purpose.
    assert "runtime-entitlements.plist" in run
    assert "--entitlements studio/src-tauri/Entitlements.plist" in run


def test_the_payload_entitlements_are_not_quietly_widened():
    """A second look at the same heredoc, as a parsed plist rather than as text.

    Written out and parsed, so a key added to it has to be added to this list
    too. allow-dyld-environment-variables is the one that would be tempting and
    is deliberately absent: the backend's DYLD_LIBRARY_PATH fallback for
    llama-server is inert under the hardened runtime, and the payload's
    llama-server resolves through @loader_path anyway.
    """
    run = _step(SIGN_STEP)["run"]
    body = run.split("<<'PLIST'\n", 1)[1].split("\nPLIST\n", 1)[0]
    plist = plistlib.loads(body.encode())
    assert plist == {
        "com.apple.security.cs.disable-library-validation": True,
        "com.apple.security.cs.allow-unsigned-executable-memory": True,
    }


# ──────────────────────────────────────────── a synthetic bundle and Apple stubs

MACHO_LE64 = b"\xcf\xfa\xed\xfe"
MH_OBJECT, MH_EXECUTE, MH_DYLIB, MH_BUNDLE = 0x1, 0x2, 0x6, 0x8


def _macho(filetype: int) -> bytes:
    """A minimal little-endian arm64 Mach-O header, enough for file(1) too."""
    return (
        MACHO_LE64
        + struct.pack("<I", 0x0100000C)   # cputype arm64
        + struct.pack("<I", 0)            # cpusubtype
        + struct.pack("<I", filetype)
        + struct.pack("<IIII", 2, 0, 0x00200085, 0)
        + b"\0" * 96
    )


def _fat(filetype: int) -> bytes:
    inner = _macho(filetype)
    header = b"\xca\xfe\xba\xbe" + struct.pack(">I", 1) + struct.pack(
        ">IIIII", 0x0100000C, 0, 4096, len(inner), 12
    )
    return header.ljust(4096, b"\0") + inner


def _java_class() -> bytes:
    """Shares CA FE BA BE with a universal binary, and must not be signed."""
    return b"\xca\xfe\xba\xbe" + struct.pack(">HH", 0, 52) + b"\0" * 200


def _write(path: Path, blob: bytes, mode = 0o644):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(blob)
    path.chmod(mode)


def _build_app(root: Path, *, site_packages = 560, python = 34, llama = 22) -> Path:
    """A fake Unsloth.app with a payload shaped like the real one."""
    app = root / "Unsloth.app"
    contents = app / "Contents"
    _write(contents / "MacOS" / "Unsloth", _macho(MH_EXECUTE), 0o755)
    (contents / "Info.plist").write_bytes(
        plistlib.dumps({"CFBundleExecutable": "Unsloth", "CFBundleIdentifier": "ai.unsloth.app"})
    )
    runtime = contents / "Resources" / "runtime"
    (runtime / "BUNDLE_MANIFEST.json").parent.mkdir(parents = True, exist_ok = True)
    (runtime / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "distribution_count": 262,
                "incomplete": [],
                "python": {"version": "3.13.14"},
                "locks": {"darwin-arm64-bundle.lock.txt": {"sha256": "0" * 64}},
            }
        ),
        encoding = "utf-8",
    )

    for index in range(site_packages):
        _write(runtime / "site-packages" / f"pkg{index}" / "_ext.so", _macho(MH_BUNDLE))
    # Not every file in site-packages is a Mach-O, and the scan has to say so.
    _write(runtime / "site-packages" / "pkg0" / "__init__.py", b"x = 1\n")
    _write(runtime / "site-packages" / "pkg1" / "notes.txt", b"hello\n")
    _write(runtime / "site-packages" / "pkg2" / "Widget.class", _java_class())
    _write(runtime / "site-packages" / "torch" / "lib" / "libtorch.dylib", _macho(MH_DYLIB))
    _write(runtime / "site-packages" / "torch" / "bin" / "torch_shm_manager",
           _macho(MH_EXECUTE), 0o755)
    # A universal2 wheel ships one of these.
    _write(runtime / "site-packages" / "pkg3" / "_fat.so", _fat(MH_BUNDLE))

    for index in range(python):
        _write(runtime / "python" / "lib" / "python3.13" / "lib-dynload" / f"_m{index}.so",
               _macho(MH_BUNDLE))
    _write(runtime / "python" / "lib" / "libpython3.13.dylib", _macho(MH_DYLIB))
    _write(runtime / "python" / "bin" / "python3.13", _macho(MH_EXECUTE), 0o755)
    # The path the app actually invokes is a symlink; the signature belongs on
    # its target, and a symlink must never be signed twice.
    (runtime / "python" / "bin" / "python3").symlink_to("python3.13")
    # An object file: Mach-O, unsignable, and not required to be signed.
    _write(runtime / "python" / "lib" / "python3.13" / "config-3.13-darwin" / "python.o",
           _macho(MH_OBJECT))
    _write(runtime / "python" / "lib" / "python3.13" / "config-3.13-darwin" / "libpython3.13.a",
           b"!<arch>\n" + b"\0" * 64)
    # A console script: text, with a shebang, and not code.
    _write(runtime / "site-packages" / "bin" / "unsloth", b"#!/bin/sh\nexec python3 -m x\n", 0o755)

    for index in range(llama):
        _write(runtime / "llama.cpp" / "build" / "bin" / f"libggml{index}.dylib", _macho(MH_DYLIB))
    _write(runtime / "llama.cpp" / "build" / "bin" / "llama-server", _macho(MH_EXECUTE), 0o755)
    (runtime / "llama.cpp" / "build" / "bin" / "libggml.0.dylib").symlink_to("libggml0.dylib")
    _write(runtime / "whisper.cpp" / "build" / "bin" / "whisper-server", _macho(MH_EXECUTE), 0o755)
    _write(runtime / "node" / "bin" / "node", _macho(MH_EXECUTE), 0o755)
    _write(runtime / "oxc-node-modules" / "oxc-parser" / "oxc.darwin-arm64.node", _macho(MH_DYLIB))
    return app


CODESIGN_STUB = r"""#!/bin/sh
printf '%s\n' "$*" >> "$CODESIGN_LOG"
for last; do :; done
case "$1" in
  --display)
    case "$*" in
      *--entitlements*)
        case "$last" in
          */runtime/*) cat "$STUB_RUNTIME_ENTITLEMENTS" ;;
          *) cat "$STUB_APP_ENTITLEMENTS" ;;
        esac
        exit 0 ;;
    esac
    if [ -n "${STUB_UNSIGNED:-}" ]; then
      case "$last" in
        *"$STUB_UNSIGNED"*) echo "code object is not signed at all" >&2; exit 1 ;;
      esac
    fi
    {
      echo "Executable=$last"
      echo "CodeDirectory v=20500 size=100 flags=${STUB_FLAGS:-0x10000(runtime)}"
      echo "Signature=${STUB_SIGNATURE:-3ff9c0d0}"
      echo "TeamIdentifier=${STUB_TEAM:-ABCDE12345}"
    } >&2
    exit 0 ;;
  --verify)
    exit "${STUB_VERIFY_STATUS:-0}"
    ;;
esac
if [ -n "${STUB_SIGN_FAIL:-}" ]; then
  case "$*" in
    *"$STUB_SIGN_FAIL"*)
      count=0
      [ -f "$STUB_SIGN_FAIL_COUNT" ] && count="$(cat "$STUB_SIGN_FAIL_COUNT")"
      count=$((count + 1))
      printf '%s\n' "$count" > "$STUB_SIGN_FAIL_COUNT"
      if [ "$count" -le "${STUB_SIGN_FAIL_TIMES:-1}" ]; then
        echo "the timestamp service is not available" >&2
        exit 1
      fi
      ;;
  esac
fi
exit 0
"""

PLUTIL_STUB = r"""#!/usr/bin/env python3
import plistlib
import sys

args = sys.argv[1:]
if args[0] == "-lint":
    for path in args[1:]:
        with open(path, "rb") as handle:
            plistlib.load(handle)
        print(f"{path}: OK")
elif args[0] == "-p":
    with open(args[1], "rb") as handle:
        print(plistlib.load(handle))
elif args[0] == "-extract":
    key = args[1]
    path = args[-1]
    with open(path, "rb") as handle:
        print(plistlib.load(handle)[key])
else:
    sys.exit(f"plutil stub does not implement {args}")
"""

SECURITY_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$SECURITY_LOG"
exit 0
"""

SLEEP_STUB = """#!/bin/sh
printf 'sleep %s\\n' "$*" >> "$CODESIGN_LOG"
exit 0
"""


def _stub_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir(exist_ok = True)
    for name, body in (
        ("codesign", CODESIGN_STUB),
        ("plutil", PLUTIL_STUB),
        ("security", SECURITY_STUB),
        ("sleep", SLEEP_STUB),
    ):
        path = bin_dir / name
        path.write_text(body, encoding = "utf-8")
        path.chmod(0o755)
    return bin_dir


def _sandbox(tmp_path: Path, **app_kwargs):
    """A working tree, a synthetic app, stubbed Apple tools and a clean env."""
    work = tmp_path / "work"
    (work / "studio" / "src-tauri").mkdir(parents = True, exist_ok = True)
    shutil.copy2(ENTITLEMENTS, work / "studio" / "src-tauri" / "Entitlements.plist")
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(exist_ok = True)
    app = _build_app(tmp_path / "bundle", **app_kwargs)
    bin_dir = _stub_bin(tmp_path)

    env = {
        "PATH": f"{bin_dir}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
        "GITHUB_ENV": str(tmp_path / "github-env"),
        "CODESIGN_LOG": str(tmp_path / "codesign.log"),
        "SECURITY_LOG": str(tmp_path / "security.log"),
        "STUB_SIGN_FAIL_COUNT": str(tmp_path / "sign-fail-count"),
        "APP": str(app),
        "APPLE_SIGNING_IDENTITY": "Developer ID Application: Unsloth AI Inc. (ABCDE12345)",
        "KEYCHAIN_PASSWORD": "keychain-secret",
    }
    # The floors are declared in the step's own `env:` block, so the test signs
    # against the real numbers rather than a copy of them.
    env.update({
        name: value
        for name, value in _step(SIGN_STEP).get("env", {}).items()
        if isinstance(value, str) and "${{" not in value
    })
    for path in ("github-output", "github-env", "codesign.log", "security.log"):
        (tmp_path / path).write_text("", encoding = "utf-8")
    return work, app, runner_temp, env


def _run(step_name: str, work: Path, env: dict, *, extra_env: dict | None = None):
    merged = dict(env)
    merged.update(extra_env or {})
    return subprocess.run(
        ["bash", "-c", _step(step_name)["run"]],
        cwd = work,
        env = merged,
        text = True,
        capture_output = True,
        check = False,
    )


def _codesign_calls(env: dict) -> list[str]:
    return [
        line for line in Path(env["CODESIGN_LOG"]).read_text(encoding = "utf-8").splitlines()
        if line
    ]


def _signing_calls(env: dict) -> list[str]:
    return [
        line for line in _codesign_calls(env)
        if not line.startswith("--display") and not line.startswith("--verify")
        and not line.startswith("sleep ")
    ]


# ───────────────────────────────────────────── which files the chain rewrites

def _run_locate(tmp_path: Path, artifact_paths, *, app = "Unsloth.app"):
    """The step resolves the three files tauri reported, from tauri's own output."""
    work = tmp_path / "work"
    bundle = work / "studio/src-tauri/target/aarch64-apple-darwin/release/bundle/macos"
    if app:
        (bundle / app / "Contents").mkdir(parents = True)
        (bundle / app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({}))
    else:
        bundle.mkdir(parents = True)
    output = tmp_path / "github-output"
    output.write_text("", encoding = "utf-8")
    result = subprocess.run(
        ["bash", "-c", _step(LOCATE_STEP)["run"]],
        cwd = work,
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "GITHUB_OUTPUT": str(output),
            "ARTIFACT_PATHS": json.dumps(artifact_paths),
        },
        text = True,
        capture_output = True,
        check = False,
    )
    values = dict(
        line.split("=", 1) for line in output.read_text(encoding = "utf-8").splitlines() if "=" in line
    )
    return result, values


ARTIFACTS = [
    "/b/dmg/Unsloth_0.1.52_aarch64.dmg",
    "/b/macos/Unsloth.app.tar.gz",
    "/b/macos/Unsloth.app.tar.gz.sig",
    "/b/macos/Unsloth.app",
]


def test_the_rebuild_targets_come_from_tauris_own_output(tmp_path):
    """Rewriting a sibling copy would publish tauri's stale files instead.

    "Stage release assets" and the notarization step both read
    steps.build_macos.outputs.artifactPaths, so the rebuild steps have to resolve
    the same list rather than guessing filenames.
    """
    result, values = _run_locate(tmp_path, ARTIFACTS)
    assert result.returncode == 0, result.stderr
    assert values["dmg_path"] == "/b/dmg/Unsloth_0.1.52_aarch64.dmg"
    assert values["updater_path"] == "/b/macos/Unsloth.app.tar.gz"
    assert values["updater_sig_path"] == "/b/macos/Unsloth.app.tar.gz.sig"
    assert values["app_path"].endswith("bundle/macos/Unsloth.app")
    # The signature also ends in .sig and the tarball's name is a prefix of it,
    # so the two must not be confused for each other.
    assert values["updater_path"] != values["updater_sig_path"]


@pytest.mark.parametrize(
    "artifact_paths",
    [
        ARTIFACTS + ["/b/dmg/second.dmg"],
        [path for path in ARTIFACTS if not path.endswith(".dmg")],
        [path for path in ARTIFACTS if not path.endswith(".app.tar.gz")],
        [path for path in ARTIFACTS if not path.endswith(".sig")],
        "not a list",
    ],
)
def test_an_ambiguous_artifact_list_fails_closed(tmp_path, artifact_paths):
    result, values = _run_locate(tmp_path, artifact_paths)
    assert result.returncode != 0
    assert "dmg_path" not in values


def test_a_missing_app_bundle_fails_closed(tmp_path):
    result, _ = _run_locate(tmp_path, ARTIFACTS, app = "")
    assert result.returncode != 0
    assert "no .app was produced" in result.stderr


# ────────────────────────────────────────────────── the signing step, executed

@pytest.fixture(scope = "module")
def signed(tmp_path_factory):
    """Run the real signing step once against a synthetic bundle and keep the log."""
    tmp_path = tmp_path_factory.mktemp("signed")
    work, app, runner_temp, env = _sandbox(tmp_path)
    result = _run(SIGN_STEP, work, env)
    return result, app, runner_temp, env


def test_the_signing_step_succeeds_on_a_plausible_payload(signed):
    result, _, _, _ = signed
    assert result.returncode == 0, result.stderr


def test_mach_o_files_are_found_by_their_header_and_not_by_extension(signed):
    """.so, .dylib, .node and bare executables all appear in this payload.

    A suffix list would have to be right about all of them forever, so the step
    reads the first four bytes of every regular file and then the Mach-O
    `filetype` field. These assertions are what that buys: an extensionless
    binary is signed, a .class file that shares CA FE BA BE with a universal
    binary is not, a symlink is never signed (its target is), and a universal
    binary is recognised through its first slice.
    """
    _, app, _, env = signed
    signed_paths = {line.split(" ")[-1] for line in _signing_calls(env)}
    runtime = app / "Contents" / "Resources" / "runtime"

    for relative in (
        "node/bin/node",                                  # no extension
        "llama.cpp/build/bin/llama-server",               # no extension
        "python/bin/python3.13",                          # no extension
        "site-packages/torch/bin/torch_shm_manager",      # no extension
        "site-packages/torch/lib/libtorch.dylib",         # .dylib
        "site-packages/pkg0/_ext.so",                     # .so
        "site-packages/pkg3/_fat.so",                     # universal
        "oxc-node-modules/oxc-parser/oxc.darwin-arm64.node",  # .node
    ):
        assert str(runtime / relative) in signed_paths, relative

    for relative in (
        "site-packages/pkg2/Widget.class",                # Java, not Mach-O
        "site-packages/pkg0/__init__.py",
        "site-packages/pkg1/notes.txt",
        "site-packages/bin/unsloth",                      # shell script
        "python/lib/python3.13/config-3.13-darwin/libpython3.13.a",
        "python/lib/python3.13/config-3.13-darwin/python.o",   # unsignable
        "python/bin/python3",                             # symlink to python3.13
        "llama.cpp/build/bin/libggml.0.dylib",            # symlink
    ):
        assert str(runtime / relative) not in signed_paths, relative


def test_the_app_bundle_is_signed_last(signed):
    """Signing a container seals what is inside it, so the payload comes first."""
    _, app, _, env = signed
    calls = _signing_calls(env)
    assert calls[-1].endswith(str(app)), calls[-1]
    assert sum(1 for call in calls if call.endswith(str(app))) == 1
    # And the app's own executable is not signed separately: signing the bundle
    # covers it, with the bundle's entitlements.
    assert not any(call.endswith("Contents/MacOS/Unsloth") for call in calls)


def test_each_signing_pass_goes_deepest_first(signed):
    """Insurance, not the guarantee: the guarantee is that no nested bundle exists.

    Libraries and executables are signed in two passes because they get different
    entitlements, so depth is monotonic within a pass rather than across both.
    """
    _, _, _, env = signed
    libraries, executables = [], []
    for call in _signing_calls(env):
        if "/Contents/Resources/runtime/" not in call:
            continue
        (executables if "--entitlements" in call else libraries).append(call.split(" ")[-1])
    for group in (libraries, executables):
        depths = [path.count(os.sep) for path in group]
        assert depths == sorted(depths, reverse = True), group[:5]
    # Libraries first, then executables, then the bundle: all payload code is
    # signed before the container that seals it.
    assert libraries and executables


def test_every_signature_is_hardened_and_timestamped(signed):
    _, _, _, env = signed
    calls = _signing_calls(env)
    assert len(calls) > 600
    for call in calls:
        assert "--options runtime" in call, call
        assert "--timestamp" in call, call
        assert "--force" in call, call
    # --deep is not a substitute: it walks nested bundles, not files stored as
    # resources, so it would report success over the whole unsigned payload.
    assert not any("--deep" in call for call in calls)


def test_only_executables_carry_the_runtime_entitlements(signed):
    """A library's entitlements are ignored; the process that loads it carries them."""
    _, app, runner_temp, env = signed
    entitled = {
        call.split(" ")[-1] for call in _signing_calls(env)
        if "--entitlements" in call and "runtime-entitlements.plist" in call
    }
    runtime = app / "Contents" / "Resources" / "runtime"
    assert str(runtime / "node" / "bin" / "node") in entitled
    assert str(runtime / "python" / "bin" / "python3.13") in entitled
    assert str(runtime / "llama.cpp" / "build" / "bin" / "llama-server") in entitled
    assert str(runtime / "site-packages" / "pkg0" / "_ext.so") not in entitled
    assert str(runtime / "site-packages" / "torch" / "lib" / "libtorch.dylib") not in entitled

    # And the plist those signatures point at really is the two-key one.
    written = plistlib.loads((runner_temp / "runtime-entitlements.plist").read_bytes())
    assert written == {
        "com.apple.security.cs.disable-library-validation": True,
        "com.apple.security.cs.allow-unsigned-executable-memory": True,
    }


def test_the_app_keeps_its_own_entitlements(signed):
    """codesign --force replaces a signature wholesale, entitlements included."""
    _, app, _, env = signed
    app_call = next(call for call in _signing_calls(env) if call.endswith(str(app)))
    assert "--entitlements studio/src-tauri/Entitlements.plist" in app_call


def test_the_count_is_reported_for_the_provenance_record(signed):
    result, _, _, env = signed
    output = Path(env["GITHUB_OUTPUT"]).read_text(encoding = "utf-8")
    count = int(next(line for line in output.splitlines() if line.startswith("signed_macho_count="))
                .split("=")[1])
    assert count == len(_signing_calls(env)) - 1  # every payload file, not the app
    assert f"SIGNED_MACHO_COUNT={count}" in Path(env["GITHUB_ENV"]).read_text(encoding = "utf-8")
    assert f"signed {count} Mach-O files" in result.stdout


def test_the_keychain_is_unlocked_where_it_is_used(signed):
    """The certificate is imported over an hour before the first signature now.

    `security set-keychain-settings -t 3600` relocks an idle keychain, and
    codesign against a locked one fails with "User interaction is not allowed".
    """
    _, _, _, env = signed
    calls = Path(env["SECURITY_LOG"]).read_text(encoding = "utf-8")
    assert "unlock-keychain" in calls
    # Both steps that sign have to do it, not just the first.
    for name in (SIGN_STEP, DMG_STEP):
        assert "security unlock-keychain" in _step(name)["run"], name
        assert _step(name)["env"]["KEYCHAIN_PASSWORD"] == "${{ secrets.KEYCHAIN_PASSWORD }}"


def test_the_identity_is_never_traced_into_the_log():
    for name in (SIGN_STEP, DMG_STEP, UPDATER_STEP):
        live = [
            line for line in _step(name)["run"].splitlines()
            if not line.strip().startswith("#")
        ]
        assert not any(re.match(r"\s*set\s+-[a-z]*x", line) for line in live), name


# ─────────────────────────────────────── the signing step's fail-closed paths

def test_an_empty_payload_cannot_reach_notarization(tmp_path):
    """The failure this guard exists for: nothing to sign, so nothing to reject."""
    work, app, _, env = _sandbox(tmp_path, site_packages = 0, python = 0, llama = 0)
    shutil.rmtree(app / "Contents" / "Resources" / "runtime" / "site-packages")
    shutil.rmtree(app / "Contents" / "Resources" / "runtime" / "python")
    shutil.rmtree(app / "Contents" / "Resources" / "runtime" / "llama.cpp")
    result = _run(SIGN_STEP, work, env)
    assert result.returncode != 0
    assert "implausibly thin" in result.stderr
    assert _signing_calls(env) == []


def test_a_payload_that_lost_one_component_cannot_hide_in_a_healthy_total(tmp_path):
    """Which is why the floors are per component and not just a total.

    site-packages alone clears the total floor, so a payload with no llama.cpp in
    it would otherwise sign 600 files, notarize cleanly, and ship an app whose
    inference backend is missing.
    """
    work, app, _, env = _sandbox(tmp_path)
    shutil.rmtree(app / "Contents" / "Resources" / "runtime" / "llama.cpp")
    result = _run(SIGN_STEP, work, env)
    assert result.returncode != 0
    assert "llama.cpp: 0 Mach-O files" in result.stderr
    assert _signing_calls(env) == []


def test_a_missing_payload_fails_before_anything_is_signed(tmp_path):
    work, app, _, env = _sandbox(tmp_path, site_packages = 1, python = 1, llama = 1)
    shutil.rmtree(app / "Contents" / "Resources" / "runtime")
    result = _run(SIGN_STEP, work, env)
    assert result.returncode != 0
    assert "the injection step did not run" in result.stderr


def test_a_nested_code_bundle_is_refused_rather_than_signed_as_a_file(tmp_path):
    """A .framework has to be signed as a bundle, and would need its own order."""
    work, app, _, env = _sandbox(tmp_path, site_packages = 4, python = 2, llama = 2)
    framework = app / "Contents" / "Resources" / "runtime" / "site-packages" / "Q.framework"
    _write(framework / "Versions" / "A" / "Q", _macho(MH_DYLIB))
    result = _run(SIGN_STEP, work, env)
    assert result.returncode != 0
    assert "nested code bundles" in result.stderr
    assert _signing_calls(env) == []


def test_a_mach_o_outside_the_payload_fails_closed(tmp_path):
    """Signing the bundle covers Contents/MacOS. It does not cover a sidecar."""
    work, app, _, env = _sandbox(tmp_path, site_packages = 4, python = 2, llama = 2)
    _write(app / "Contents" / "Frameworks" / "libextra.dylib", _macho(MH_DYLIB))
    result = _run(SIGN_STEP, work, env)
    assert result.returncode != 0
    assert "outside both the payload and Contents/MacOS" in result.stderr


def test_an_unrecognised_mach_o_filetype_stops_the_release(tmp_path):
    """Deciding what a new filetype is beats leaving it unsigned by default."""
    work, app, _, env = _sandbox(tmp_path, site_packages = 4, python = 2, llama = 2)
    _write(app / "Contents" / "Resources" / "runtime" / "site-packages" / "weird.kext",
           _macho(0xB))  # MH_KEXT_BUNDLE
    result = _run(SIGN_STEP, work, env)
    assert result.returncode != 0
    assert "unrecognised Mach-O filetypes" in result.stderr


def test_a_throttled_timestamp_is_retried(tmp_path):
    """--timestamp is a round trip to Apple's timestamp service, 700 times over."""
    work, app, _, env = _sandbox(tmp_path)
    target = str(app / "Contents" / "Resources" / "runtime" / "node" / "bin" / "node")
    result = _run(
        SIGN_STEP, work, env,
        extra_env = {"STUB_SIGN_FAIL": target, "STUB_SIGN_FAIL_TIMES": "2"},
    )
    assert result.returncode == 0, result.stderr
    attempts = [call for call in _signing_calls(env) if call.endswith(target)]
    assert len(attempts) == 3
    assert "sleep 10" in _codesign_calls(env)
    assert "sleep 20" in _codesign_calls(env)


def test_a_persistently_failing_signature_stops_the_release(tmp_path):
    work, app, _, env = _sandbox(tmp_path)
    target = str(app / "Contents" / "Resources" / "runtime" / "node" / "bin" / "node")
    result = _run(
        SIGN_STEP, work, env,
        extra_env = {"STUB_SIGN_FAIL": target, "STUB_SIGN_FAIL_TIMES": "99"},
    )
    assert result.returncode != 0
    assert "codesign failed three times" in result.stderr
    # And the app was never signed over an incompletely signed payload.
    assert not any(call.endswith(str(app)) for call in _signing_calls(env))


def test_a_missing_identity_stops_before_the_keychain_is_touched(tmp_path):
    work, _, _, env = _sandbox(tmp_path, site_packages = 2, python = 1, llama = 1)
    result = _run(SIGN_STEP, work, env, extra_env = {"APPLE_SIGNING_IDENTITY": ""})
    assert result.returncode != 0
    assert "refusing to publish an unsigned bundle" in result.stderr
    assert Path(env["SECURITY_LOG"]).read_text(encoding = "utf-8") == ""


# ────────────────────────────────────────────── the verification step, executed

@needs_file
def test_the_verification_step_accepts_a_fully_signed_bundle(tmp_path):
    work, app, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    result = _run(VERIFY_STEP, work, env, extra_env = _verify_env(tmp_path, runner_temp))
    assert result.returncode == 0, result.stderr + result.stdout
    assert "file(1) and the signer agree" in result.stdout
    assert "entitlements: app without library-validation relief" in result.stdout


def _verify_env(tmp_path: Path, runner_temp: Path) -> dict:
    """What the verify step reads that the sign step does not provide."""
    app_plist = tmp_path / "stub-app-entitlements.plist"
    app_plist.write_bytes(ENTITLEMENTS.read_bytes())
    return {
        "SIGNED": str(
            len({
                entry
                for name in ("macho-executables", "macho-libraries")
                for entry in (runner_temp / name).read_bytes().decode().split("\0")
                if entry
            })
        ),
        "STUB_APP_ENTITLEMENTS": str(app_plist),
        "STUB_RUNTIME_ENTITLEMENTS": str(runner_temp / "runtime-entitlements.plist"),
    }


@needs_file
def test_a_mach_o_the_signer_missed_fails_the_verification(tmp_path):
    """file(1) is the independent oracle: it is what notarization would agree with."""
    work, app, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    # Appears after the signing pass, exactly like a Mach-O the header parser
    # failed to classify would.
    _write(app / "Contents" / "Resources" / "runtime" / "node" / "lib" / "sneaky.dylib",
           _macho(MH_DYLIB))
    result = _run(VERIFY_STEP, work, env, extra_env = _verify_env(tmp_path, runner_temp))
    assert result.returncode != 0
    assert "only file(1) saw" in result.stderr
    assert "sneaky.dylib" in result.stderr


@needs_file
def test_a_signature_without_hardened_runtime_fails_the_verification(tmp_path):
    """Notarization requires the runtime flag, and library validation depends on it."""
    work, _, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    extra = _verify_env(tmp_path, runner_temp) | {"STUB_FLAGS": "0x0(none)"}
    result = _run(VERIFY_STEP, work, env, extra_env = extra)
    assert result.returncode != 0
    assert "signed without hardened runtime" in result.stderr


@needs_file
def test_an_ad_hoc_or_teamless_signature_fails_the_verification(tmp_path):
    work, _, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    base = _verify_env(tmp_path, runner_temp)
    adhoc = _run(VERIFY_STEP, work, env, extra_env = base | {"STUB_SIGNATURE": "adhoc"})
    assert adhoc.returncode != 0
    assert "ad hoc signature" in adhoc.stderr
    teamless = _run(VERIFY_STEP, work, env, extra_env = base | {"STUB_TEAM": "not set"})
    assert teamless.returncode != 0
    assert "not signed with the release identity" in teamless.stderr


@needs_file
def test_an_unsigned_payload_file_fails_the_verification(tmp_path):
    work, _, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    extra = _verify_env(tmp_path, runner_temp) | {"STUB_UNSIGNED": "libtorch.dylib"}
    result = _run(VERIFY_STEP, work, env, extra_env = extra)
    assert result.returncode != 0
    assert "not signed: " in result.stderr


@needs_file
def test_the_verification_rejects_an_app_that_lost_its_entitlements(tmp_path):
    """The realistic mistake: --force without --entitlements strips them silently."""
    work, _, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    stripped = tmp_path / "stripped.plist"
    stripped.write_bytes(plistlib.dumps({}))
    extra = _verify_env(tmp_path, runner_temp) | {"STUB_APP_ENTITLEMENTS": str(stripped)}
    result = _run(VERIFY_STEP, work, env, extra_env = extra)
    assert result.returncode != 0
    assert "the app lost its entitlements" in result.stderr


@needs_file
def test_the_verification_rejects_reintroducing_relief_on_the_app(tmp_path):
    work, _, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    widened = tmp_path / "widened.plist"
    widened.write_bytes(
        plistlib.dumps({
            "com.apple.security.network.client": True,
            "com.apple.security.cs.disable-library-validation": True,
        })
    )
    extra = _verify_env(tmp_path, runner_temp) | {"STUB_APP_ENTITLEMENTS": str(widened)}
    result = _run(VERIFY_STEP, work, env, extra_env = extra)
    assert result.returncode != 0
    assert "it does not need it" in result.stderr


@needs_file
def test_the_verification_rejects_an_interpreter_without_the_relief(tmp_path):
    work, _, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    narrowed = tmp_path / "narrowed.plist"
    narrowed.write_bytes(plistlib.dumps({}))
    extra = _verify_env(tmp_path, runner_temp) | {"STUB_RUNTIME_ENTITLEMENTS": str(narrowed)}
    result = _run(VERIFY_STEP, work, env, extra_env = extra)
    assert result.returncode != 0
    assert "will not load" in result.stderr


@needs_file
def test_the_verification_runs_the_deep_strict_check_on_the_app(tmp_path):
    work, app, runner_temp, env = _sandbox(tmp_path)
    assert _run(SIGN_STEP, work, env).returncode == 0
    assert _run(VERIFY_STEP, work, env, extra_env = _verify_env(tmp_path, runner_temp)).returncode == 0
    assert f"--verify --deep --strict --verbose=2 {app}" in _codesign_calls(env)


# ─────────────────────────────────────────── the updater artifact, rebuilt

NPX_STUB = r"""#!/bin/sh
printf '%s\n' "$*" >> "$NPX_LOG"
for last; do :; done
if [ -n "${NPX_WRITE_SIG:-}" ]; then
  printf '%s' "$NPX_WRITE_SIG" > "$last.sig"
fi
exit 0
"""


def _minisign_signature() -> str:
    import base64

    body = (
        b"untrusted comment: signature from tauri secret key\n"
        b"RUR" + b"A" * 40 + b"\n"
        b"trusted comment: timestamp:1787000000\tfile:Unsloth.app.tar.gz\n"
        b"AAAA\n"
    )
    return base64.b64encode(body).decode()


def _updater_sandbox(tmp_path: Path):
    work, app, runner_temp, env = _sandbox(tmp_path, site_packages = 4, python = 2, llama = 2)
    updater = tmp_path / "bundle" / "Unsloth.app.tar.gz"
    updater.write_bytes(b"stale tarball built before the payload existed")
    sig = tmp_path / "bundle" / "Unsloth.app.tar.gz.sig"
    sig.write_text("stale-signature", encoding = "utf-8")
    npx = tmp_path / "stubbin" / "npx"
    npx.write_text(NPX_STUB, encoding = "utf-8")
    npx.chmod(0o755)
    env = dict(env)
    env.update({
        "UPDATER": str(updater),
        "UPDATER_SIG": str(sig),
        "NPX_LOG": str(tmp_path / "npx.log"),
        "NPX_WRITE_SIG": _minisign_signature(),
        "TAURI_SIGNING_PRIVATE_KEY": "dGF1cmkta2V5",
        "TAURI_SIGNING_PRIVATE_KEY_PASSWORD": "",
    })
    (tmp_path / "npx.log").write_text("", encoding = "utf-8")
    return work, app, updater, sig, env


def test_the_updater_tarball_is_rebuilt_from_the_signed_app(tmp_path):
    """Publishing tauri's copy would push shipped installs an unsigned app.

    tauri builds the updater tarball from the .app as it stood before the
    injection: no runtime inside it and no Apple signature on it. macOS refuses
    to launch that, so the update would brick the install it replaced.
    """
    work, app, updater, sig, env = _updater_sandbox(tmp_path)
    result = _run(UPDATER_STEP, work, env)
    assert result.returncode == 0, result.stderr

    # A real tarball, holding the app the injection and signing produced.
    listing = subprocess.run(
        ["tar", "-tzf", str(updater)], capture_output = True, text = True, check = True
    ).stdout.splitlines()
    assert "Unsloth.app/" in listing or "Unsloth.app" in listing[0]
    assert any(entry.endswith("Contents/Resources/runtime/python/bin/python3") for entry in listing)
    assert any(entry.endswith("Contents/MacOS/Unsloth") for entry in listing)

    # The signature is regenerated over the new bytes, and it is the shape
    # publish-release will demand of it.
    assert sig.read_text(encoding = "utf-8").strip() == _minisign_signature()
    assert "signer sign" in Path(env["NPX_LOG"]).read_text(encoding = "utf-8")
    assert "updater signature regenerated" in result.stdout


def test_the_rebuilt_tarball_is_checked_by_unpacking_it(tmp_path):
    """The round trip is where a signature dies, so the check is on the artifact."""
    work, _, _, _, env = _updater_sandbox(tmp_path)
    assert _run(UPDATER_STEP, work, env).returncode == 0
    verified = [call for call in _codesign_calls(env) if call.startswith("--verify")]
    assert verified, "the extracted app was never verified"
    assert all("--deep --strict" in call for call in verified)
    assert any("updater-check" in call for call in verified)


def test_symlinks_and_apple_double_files_are_kept_out_of_the_tarball(tmp_path):
    """Two ways the round trip breaks the seal, both avoided rather than hoped for.

    Dereferenced symlinks and `._` AppleDouble members both leave the extracted
    bundle different from the one that was signed.
    """
    run = _step(UPDATER_STEP)["run"]
    assert "COPYFILE_DISABLE=1" in run
    work, app, updater, _, env = _updater_sandbox(tmp_path)
    assert _run(UPDATER_STEP, work, env).returncode == 0
    entries = subprocess.run(
        ["tar", "-tvzf", str(updater)], capture_output = True, text = True, check = True
    ).stdout
    assert not any(line.split("/")[-1].startswith("._") for line in entries.splitlines())
    assert "python3 -> python3.13" in entries or "python3 link to python3.13" in entries


def test_a_missing_updater_signature_stops_the_release(tmp_path):
    work, _, _, sig, env = _updater_sandbox(tmp_path)
    result = _run(UPDATER_STEP, work, env, extra_env = {"NPX_WRITE_SIG": ""})
    assert result.returncode != 0
    assert not sig.exists()


def test_a_malformed_updater_signature_stops_the_release(tmp_path):
    work, _, _, _, env = _updater_sandbox(tmp_path)
    result = _run(UPDATER_STEP, work, env, extra_env = {"NPX_WRITE_SIG": "not base64 at all!!"})
    assert result.returncode != 0
    assert "not base64" in result.stderr


def test_the_updater_signature_shape_matches_what_publish_release_demands():
    """Two steps, one contract: the check here has to be the check there."""
    rebuild = _step(UPDATER_STEP)["run"]
    publish = _step("Generate versioned updater metadata", "publish-release")["run"]
    for marker in ("untrusted comment:", "trusted comment:", "b64decode"):
        assert marker in rebuild, marker
        assert marker in publish, marker


# ─────────────────────────────────────────────── the disk image, rebuilt

# A disk image is stood in for by a file plus a sibling `<image>.contents`
# directory holding what mounting it would show. That is enough to exercise
# everything this step decides: that the image is converted rather than
# recreated, that the app inside is replaced by name, and that what ends up in
# the final image is the signed bundle.
HDIUTIL_STUB = r"""#!/bin/sh
printf '%s\n' "$*" >> "$HDIUTIL_LOG"
case "$1" in
  convert)
    src="$2"
    out=""
    next=0
    for arg in "$@"; do
      if [ "$next" = "1" ]; then out="$arg"; next=0; fi
      if [ "$arg" = "-o" ]; then next=1; fi
    done
    cp "$src" "$out"
    rm -rf "$out.contents"
    if [ -d "$src.contents" ]; then cp -a "$src.contents" "$out.contents"; fi
    ;;
  resize)
    ;;
  attach)
    image="$2"
    mount=""
    next=0
    for arg in "$@"; do
      if [ "$next" = "1" ]; then mount="$arg"; next=0; fi
      if [ "$arg" = "-mountpoint" ]; then next=1; fi
    done
    mkdir -p "$mount"
    if [ -d "$image.contents" ]; then cp -a "$image.contents/." "$mount/"; fi
    printf '%s\n' "$image" > "$HDIUTIL_IMAGE"
    ;;
  detach)
    mount="$2"
    if [ -f "$HDIUTIL_IMAGE" ]; then
      image="$(cat "$HDIUTIL_IMAGE")"
      rm -rf "$image.contents"
      mkdir -p "$image.contents"
      cp -a "$mount/." "$image.contents/" 2>/dev/null || true
    fi
    rm -rf "$mount"
    ;;
esac
exit 0
"""

DITTO_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$HDIUTIL_LOG"
cp -a "$1" "$2"
exit 0
"""


def _dmg_sandbox(tmp_path: Path, *, app_name = "Unsloth.app"):
    work, app, runner_temp, env = _sandbox(tmp_path, site_packages = 4, python = 2, llama = 2)
    dmg = tmp_path / "bundle" / "Unsloth_0.1.52_aarch64.dmg"
    dmg.write_bytes(b"a disk image tauri built around an unsigned app")
    # What "mounting" it will produce: the pre-injection app, by name.
    contents = Path(f"{dmg}.contents")
    (contents / app_name / "Contents").mkdir(parents = True)
    (contents / app_name / "Contents" / "Info.plist").write_bytes(plistlib.dumps({}))
    (contents / "Applications").symlink_to("/Applications")
    for name, body in (("hdiutil", HDIUTIL_STUB), ("ditto", DITTO_STUB)):
        path = tmp_path / "stubbin" / name
        path.write_text(body, encoding = "utf-8")
        path.chmod(0o755)
    env = dict(env)
    env.update({
        "DMG": str(dmg),
        "HDIUTIL_LOG": str(tmp_path / "hdiutil.log"),
        "HDIUTIL_IMAGE": str(tmp_path / "current-image"),
    })
    (tmp_path / "hdiutil.log").write_text("", encoding = "utf-8")
    return work, app, dmg, env


def test_the_disk_image_is_rebuilt_around_the_signed_app(tmp_path):
    work, app, dmg, env = _dmg_sandbox(tmp_path)
    result = _run(DMG_STEP, work, env)
    assert result.returncode == 0, result.stderr

    log = Path(env["HDIUTIL_LOG"]).read_text(encoding = "utf-8")
    # Converted and grown rather than recreated: tauri's image carries the
    # background, the window geometry, the icon positions and the /Applications
    # symlink, all encoded in a .DS_Store that hdiutil create would not reproduce.
    assert "convert" in log and "resize" in log
    assert "-format UDRW" in log
    # ULFO, matching the dev build's image.
    assert "-format ULFO" in log
    # ditto, not cp -R: symlinks, permissions and extended attributes survive.
    assert str(app) in log

    # And what landed in the image that becomes the .dmg is the signed bundle,
    # payload included, under the name the .DS_Store layout expects.
    final = Path(env["RUNNER_TEMP"]) / "dmg-rebuild" / "out.dmg.contents"
    assert (final / "Unsloth.app" / "Contents" / "Resources" / "runtime"
            / "python" / "bin" / "python3").is_symlink()
    assert (final / "Unsloth.app" / "Contents" / "MacOS" / "Unsloth").is_file()
    assert (final / "Applications").is_symlink(), "the /Applications shortcut survived"

    # The image ends up signed, with a timestamp, because notarization needs it
    # and nothing else signs it now.
    signing = [call for call in _signing_calls(env) if call.endswith(str(dmg))]
    assert len(signing) == 1, _signing_calls(env)
    assert "--timestamp" in signing[0]
    assert f"--verify --verbose=2 {dmg}" in _codesign_calls(env)


def test_a_disk_image_holding_a_differently_named_app_fails_closed(tmp_path):
    """The .DS_Store layout is keyed by name, so a rename is not a silent fixup."""
    work, _, _, env = _dmg_sandbox(tmp_path, app_name = "Unsloth Studio.app")
    result = _run(DMG_STEP, work, env)
    assert result.returncode != 0
    assert "keyed by name" in result.stderr


def test_the_rebuilt_image_is_checked_before_it_is_closed(tmp_path):
    work, _, _, env = _dmg_sandbox(tmp_path)
    assert _run(DMG_STEP, work, env).returncode == 0
    calls = _codesign_calls(env)
    mounted_verify = next(
        (index for index, call in enumerate(calls) if "--verify" in call and "dmg-rebuild" in call),
        None,
    )
    assert mounted_verify is not None, calls
    closed = next(index for index, call in enumerate(calls) if call.endswith(env["DMG"]))
    assert mounted_verify < closed, "the app inside the image is verified before the image is signed"


# ────────────────────────────────────────────────────── provenance, executed

def _build_input_record(**overrides) -> dict:
    record = {
        "artifact": "macos-aarch64",
        "runner_label": "macos-latest",
        "runner": {"os": "macOS"},
        "commit": "c" * 40,
        "source_date_epoch": "1787000000",
        "toolchain": {"rustc": "rustc 1.94.1 (abc 2026-01-01)"},
        "lockfiles": {"studio/src-tauri/Cargo.lock": "d" * 64},
        "pinned_actions": ["actions/checkout@" + "0" * 40],
        "pinned_downloads": {"APPIMAGETOOL_SHA256": "e" * 64},
    }
    record.update(overrides)
    return record


def _bundled_runtime_block() -> dict:
    return {
        "root": "Contents/Resources/runtime",
        "manifest_sha256": "f" * 64,
        "manifest": {
            "distribution_count": 262,
            "python": {"version": "3.13.14"},
            "locks": {"darwin-arm64-bundle.lock.txt": {"sha256": "a" * 64}},
        },
        "signed_macho_count": 721,
    }


def _run_assemble(tmp_path: Path, *, bundled = True):
    runner_temp = tmp_path / "runner-temp"
    (runner_temp / "build-inputs").mkdir(parents = True, exist_ok = True)
    (runner_temp / "desktop-release-assets").mkdir(parents = True, exist_ok = True)
    for artifact in ("macos-aarch64", "linux-x64", "windows-x64"):
        record = _build_input_record(artifact = artifact)
        if artifact == "macos-aarch64" and bundled:
            record["bundled_runtime"] = _bundled_runtime_block()
        (runner_temp / "build-inputs" / f"build-inputs-{artifact}.json").write_text(
            json.dumps(record), encoding = "utf-8"
        )
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "RUNNER_TEMP": str(runner_temp),
        "STUDIO_VERSION": "v0.1.52-beta",
        "APP_VERSION": "0.1.52",
        "PYPI_VERSION": "2026.8.18",
        "DESKTOP_RELEASE_TAG": "v0.1.52-beta",
        "GITHUB_REPOSITORY": "unslothai/unsloth",
        "GITHUB_RUN_ID": "1",
    }
    result = subprocess.run(
        ["bash", "-c", _step("Assemble build-inputs.json", "publish-release")["run"]],
        cwd = tmp_path,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )
    return result, runner_temp / "desktop-release-assets" / "build-inputs.json"


def test_the_release_record_carries_the_payload_manifest_and_lock_digest(tmp_path):
    """A shipped .dmg contains a whole Python stack; the record has to say which one.

    BUNDLE_MANIFEST.json is embedded whole rather than summarised, because it
    already answers every question worth asking: which distribution at which
    version, the digest of every prebuilt archive, and -- in its own `locks` map
    -- the SHA-256 of darwin-arm64-bundle.lock.txt. Nothing about the payload is
    restated in the workflow, the same way the pinned downloads are scraped out
    of it rather than copied into it.
    """
    result, out = _run_assemble(tmp_path)
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text(encoding = "utf-8"))
    bundled = record["platforms"]["macos-aarch64"]["bundled_runtime"]
    assert bundled["manifest_sha256"] == "f" * 64
    assert bundled["manifest"]["locks"]["darwin-arm64-bundle.lock.txt"]["sha256"] == "a" * 64
    assert bundled["signed_macho_count"] == 721
    assert bundled["root"] == "Contents/Resources/runtime"
    # Only macOS ships one, so only macOS records one.
    for artifact in ("linux-x64", "windows-x64"):
        assert "bundled_runtime" not in record["platforms"][artifact]


def test_a_release_that_cannot_describe_its_payload_is_not_published(tmp_path):
    result, out = _run_assemble(tmp_path, bundled = False)
    assert result.returncode != 0
    assert "no bundled runtime manifest" in result.stderr
    assert not out.exists()


def test_the_record_is_read_out_of_the_signed_app_and_not_the_staging_directory():
    """What shipped, not what was assembled. They can differ, and only one matters."""
    step = _step("Record build inputs")
    assert step["env"]["BUNDLED_RUNTIME_APP"] == "${{ steps.macos_bundle.outputs.app_path }}"
    run = step["run"]
    assert "Contents/Resources/runtime/BUNDLE_MANIFEST.json" in run
    # A macOS leg that recorded no payload at all must fail rather than publish a
    # .dmg whose contents nobody can enumerate afterwards.
    assert "the macOS leg recorded no app bundle" in run
    assert "SIGNED_MACHO_COUNT" in run


# ─────────────────────────────────────── the in-workflow guard, executed

def _run_guard(tmp_path: Path, mutate = None):
    """Execute the release-time guard against a copy of the files it reads."""
    workflow_text = WORKFLOW.read_text(encoding = "utf-8")
    entitlements_text = ENTITLEMENTS.read_text(encoding = "utf-8")
    if mutate is not None:
        workflow_text, entitlements_text = mutate(workflow_text, entitlements_text)

    workflow_path = tmp_path / ".github" / "workflows" / "release-desktop.yml"
    workflow_path.parent.mkdir(parents = True, exist_ok = True)
    workflow_path.write_text(workflow_text, encoding = "utf-8")
    conf_dir = tmp_path / "studio" / "src-tauri"
    conf_dir.mkdir(parents = True, exist_ok = True)
    for source in (TAURI_CONF, WINDOWS_CONF):
        shutil.copy2(source, conf_dir / source.name)
    (conf_dir / "Entitlements.plist").write_text(entitlements_text, encoding = "utf-8")

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


def _swap_steps(workflow: str, first: str, second: str) -> str:
    """Reorder two whole steps, so the guard sees the order it forbids."""
    lines = workflow.split("\n")

    def block(name):
        start = lines.index(f"      - name: {name}")
        end = start + 1
        while end < len(lines) and not lines[end].startswith("      - name: "):
            end += 1
        return start, end

    first_start, first_end = block(first)
    second_start, second_end = block(second)
    assert first_end <= second_start
    head = lines[:first_start]
    middle = lines[first_end:second_start]
    tail = lines[second_end:]
    return "\n".join(
        head + lines[second_start:second_end] + middle + lines[first_start:first_end] + tail
    )


@needs_node
def test_the_guard_passes_on_the_committed_files(tmp_path):
    result = _run_guard(tmp_path)
    assert result.returncode == 0, result.stderr


@needs_node
def test_the_guard_rejects_signing_before_the_payload_is_injected(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (_swap_steps(workflow, INJECT_STEP, SIGN_STEP), plist),
    )
    assert result.returncode != 0
    assert f'must run "{INJECT_STEP}" before "{SIGN_STEP}"' in result.stderr


@needs_node
def test_the_guard_rejects_rebuilding_the_image_before_the_app_is_signed(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (_swap_steps(workflow, VERIFY_STEP, DMG_STEP), plist),
    )
    assert result.returncode != 0
    assert f'must run "{VERIFY_STEP}" before' in result.stderr


@needs_node
def test_the_guard_rejects_assembling_the_payload_after_the_build(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (_swap_steps(workflow, ASSEMBLE_STEP, BUILD_STEP), plist),
    )
    assert result.returncode != 0
    assert f'must run "{ASSEMBLE_STEP}" before "{BUILD_STEP}"' in result.stderr


@needs_node
def test_the_guard_rejects_handing_tauri_a_signing_identity(tmp_path):
    def mutate(workflow, plist):
        return (
            workflow.replace(
                "          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ''\n        with:\n"
                "          projectPath: studio\n          tauriScript: npx --prefix . tauri\n"
                "          args: -v ${{ matrix.args }}",
                "          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ''\n"
                "          APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}\n"
                "        with:\n          projectPath: studio\n"
                "          tauriScript: npx --prefix . tauri\n          args: -v ${{ matrix.args }}",
                1,
            ),
            plist,
        )

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert "must not receive APPLE_SIGNING_IDENTITY" in result.stderr


@needs_node
def test_the_guard_rejects_injecting_into_tauris_disk_image(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (
            workflow.replace(
                '            --app "$APP"',
                '            --app "$APP" \\\n            --dmg "$DMG"',
                1,
            ),
            plist,
        ),
    )
    assert result.returncode != 0
    assert "must not inject into tauri's disk image" in result.stderr


@needs_node
def test_the_guard_rejects_codesign_deep_as_a_substitute(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (
            workflow.replace(
                '                if codesign --force --sign "$APPLE_SIGNING_IDENTITY" \\',
                '                if codesign --deep --force --sign "$APPLE_SIGNING_IDENTITY" \\',
                1,
            ),
            plist,
        ),
    )
    assert result.returncode != 0
    assert "codesign --deep must not sign the bundled runtime" in result.stderr


@needs_node
def test_the_guard_rejects_dropping_the_hardened_runtime(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (
            workflow.replace("--options runtime", "--options library"), plist
        ),
    )
    assert result.returncode != 0
    assert "must pass --options runtime" in result.stderr


@needs_node
def test_the_guard_rejects_dropping_the_count_floor(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (workflow.replace("MIN_SIGNED_TOTAL:", "UNUSED_TOTAL:"), plist),
    )
    assert result.returncode != 0
    assert "must declare a floor" in result.stderr


@needs_node
def test_the_guard_rejects_signing_the_app_without_its_entitlements(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (
            workflow.replace("--entitlements studio/src-tauri/Entitlements.plist", "\\"),
            plist,
        ),
    )
    assert result.returncode != 0
    assert "must be re-signed with studio/src-tauri/Entitlements.plist" in result.stderr


@needs_node
def test_the_guard_rejects_putting_library_validation_relief_back_on_the_app(tmp_path):
    def mutate(workflow, plist):
        return workflow, plist.replace(
            "<key>com.apple.security.network.client</key>",
            "<key>com.apple.security.cs.disable-library-validation</key>\n    <true/>\n"
            "    <key>com.apple.security.network.client</key>",
            1,
        )

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert "must not carry" in result.stderr


@needs_node
def test_the_guard_rejects_taking_the_relief_off_the_payload(tmp_path):
    def mutate(workflow, plist):
        return (
            workflow.replace(
                "    <key>com.apple.security.cs.disable-library-validation</key>\n"
                "              <true/>\n",
                "",
                1,
            ),
            plist,
        )

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert "must be signed with" in result.stderr


@needs_node
def test_the_guard_rejects_ungating_a_macos_step(tmp_path):
    result = _run_guard(
        tmp_path,
        lambda workflow, plist: (
            workflow.replace(
                "      - name: Sign the bundled runtime and the macOS app\n"
                "        id: sign_macos\n        if: matrix.platform == 'macos-latest'\n",
                "      - name: Sign the bundled runtime and the macOS app\n"
                "        id: sign_macos\n",
                1,
            ),
            plist,
        ),
    )
    assert result.returncode != 0
    assert "must be gated on the macOS leg" in result.stderr


@needs_node
def test_the_guard_rejects_deleting_a_step_outright(tmp_path):
    def mutate(workflow, plist):
        lines = workflow.split("\n")
        start = lines.index(f"      - name: {VERIFY_STEP}")
        end = start + 1
        while end < len(lines) and not lines[end].startswith("      - name: "):
            end += 1
        return "\n".join(lines[:start] + lines[end:]), plist

    result = _run_guard(tmp_path, mutate)
    assert result.returncode != 0
    assert f'must keep the "{VERIFY_STEP}" step' in result.stderr
