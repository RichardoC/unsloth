"""Checks that one desktop version tag can only ever serve one set of binaries."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

RELEASE_TAG = "v0.1.50-beta"
SOURCE_SHA = "1f02275b86f0e0d3a5b1c9f2a4d6e8b0c2a4e6f8"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _step_index(workflow, job, name):
    """Locate a step and report its available names on failure."""
    names = [step.get("name") for step in _steps(workflow, job)]
    assert name in names, f"{job} has no step named {name!r}; steps are {names}"
    return names.index(name)


def _step(workflow, job, name):
    return _steps(workflow, job)[_step_index(workflow, job, name)]


LOCKFILES = (
    "studio/src-tauri/Cargo.lock",
    "studio/package-lock.json",
    "studio/frontend/package-lock.json",
)


def _lockfile_digest(path: Path) -> str:
    """Mirror the workflow: hash the source, not the checkout's line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _fake_lockfile_repo(tmp_path: Path) -> None:
    """A throwaway checkout carrying the three lockfiles, committed."""
    contents = {
        "studio/src-tauri/Cargo.lock": (
            '[[package]]\nname = "unsloth-studio"\nversion = "0.0.0"\n'
            '\n[[package]]\nname = "serde"\nversion = "1.0.0"\n'
        ),
        "studio/package-lock.json": '{"name": "unsloth-studio-tauri-cli"}\n',
        "studio/frontend/package-lock.json": '{"name": "unsloth-frontend"}\n',
    }
    for relative, text in contents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text, encoding = "utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c", "user.email=release@example.invalid",
            "-c", "user.name=release",
            "commit", "-qm", "lockfiles",
        ],
    ):
        subprocess.run(command, cwd = tmp_path, check = True, capture_output = True)


def test_the_release_installs_from_lockfiles_rather_than_resolving_afresh():
    """`npm install` re-resolves and may rewrite a lockfile; `npm ci` cannot.

    studio/package.json exists only to hold the Tauri CLI pin, so installing it
    by name (`npm install --save-dev @tauri-apps/cli@2.10.1`) both ignored that
    lockfile and mutated the tree it is supposed to reproduce.
    """
    build = _workflow()["jobs"]["build"]
    installs = [
        line.strip()
        for step in build["steps"]
        for line in str(step.get("run", "")).splitlines()
        if line.strip().startswith(("npm install", "npm ci"))
    ]
    assert installs, "the build job installs no npm dependencies at all"
    for command in installs:
        assert command.startswith("npm ci"), command

    package = json.loads((REPO_ROOT / "studio" / "package.json").read_text(encoding = "utf-8"))
    lock = json.loads(
        (REPO_ROOT / "studio" / "package-lock.json").read_text(encoding = "utf-8")
    )
    # `npm ci` installs what the lockfile says, so the pin has to be in both.
    assert package["devDependencies"]["@tauri-apps/cli"] == "2.10.1"
    assert lock["packages"]["node_modules/@tauri-apps/cli"]["version"] == "2.10.1"
    verify = _step(_workflow(), "build", "Verify pinned Tauri CLI")
    assert "tauri-cli 2.10.1" in verify["run"]


def test_the_release_pins_an_exact_node_and_an_exact_rustc():
    """A major-only pin still floats; the release must name the build.

    The toolchain version is the one that has to agree with
    studio/src-tauri/rust-toolchain.toml: the action's `toolchain` input
    defaults to `stable`, so leaving it out installs the cross-compilation
    targets against stable while rust-toolchain.toml forces the build itself
    onto the pinned compiler, and the macOS leg loses its Apple std.
    """
    steps = _workflow()["jobs"]["build"]["steps"]

    node = next(
        step for step in steps if str(step.get("uses", "")).startswith("actions/setup-node@")
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(node["with"]["node-version"])), node["with"]

    rust = next(
        step for step in steps if str(step.get("uses", "")).startswith("dtolnay/rust-toolchain@")
    )
    toolchain = re.search(
        r'^\s*channel\s*=\s*"([^"]+)"',
        (REPO_ROOT / "studio" / "src-tauri" / "rust-toolchain.toml").read_text(encoding = "utf-8"),
        re.M,
    )
    assert toolchain, "rust-toolchain.toml declares no channel"
    assert re.fullmatch(r"\d+\.\d+(?:\.\d+)?", toolchain.group(1)), toolchain.group(1)
    assert str(rust["with"]["toolchain"]) == toolchain.group(1)


def test_the_build_records_the_tag_commit_time_as_source_date_epoch():
    steps = _workflow()["jobs"]["build"]["steps"]
    names = [step.get("name") or str(step.get("uses")) for step in steps]
    export = _step(_workflow(), "build", "Export SOURCE_DATE_EPOCH from the tag commit")

    # The tag commit's committer timestamp: a property of the source, not of
    # when the runner happened to start.
    assert "git log -1 --format=%ct" in export["run"]
    assert 'echo "SOURCE_DATE_EPOCH=$epoch" >> "$GITHUB_ENV"' in export["run"]

    checkout = next(
        index for index, step in enumerate(steps) if "actions/checkout" in str(step.get("uses", ""))
    )
    # After the checkout it reads from, and before anything it could influence.
    assert checkout < names.index("Export SOURCE_DATE_EPOCH from the tag commit")
    first_build = min(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("tauri-apps/tauri-action@")
    )
    assert names.index("Export SOURCE_DATE_EPOCH from the tag commit") < first_build


def test_a_lockfile_the_build_rewrote_fails_the_release():
    """Both halves of the check, and where they sit.

    The snapshot has to be taken after both npm installs (so an install that
    rewrote a lockfile is caught) and after the version patch (the one
    sanctioned Cargo.lock mutation, so it is not mistaken for drift). The
    verification has to run after every bundle is built and before anything is
    staged for release.
    """
    workflow = _workflow()
    steps = workflow["jobs"]["build"]["steps"]
    names = [step.get("name") or str(step.get("uses")) for step in steps]

    snapshot = names.index("Snapshot lockfile digests")
    verify = names.index("Verify the build rewrote no lockfile")
    assert names.index("Install pinned Tauri CLI") < snapshot
    assert names.index("Install frontend dependencies") < snapshot
    assert names.index("Patch desktop app version") < snapshot
    build_steps = [
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("tauri-apps/tauri-action@")
        or step.get("name") == "Build and sign thin Linux AppImage"
    ]
    assert snapshot < min(build_steps)
    assert max(build_steps) < verify
    assert verify < names.index("Stage release assets")

    snapshot_run = _step(workflow, "build", "Snapshot lockfile digests")["run"]
    verify_run = _step(workflow, "build", "Verify the build rewrote no lockfile")["run"]
    for run in (snapshot_run, verify_run):
        # The two npm lockfiles are never patched, so HEAD stays their baseline.
        assert (
            "git diff --exit-code -- studio/package-lock.json studio/frontend/package-lock.json"
            in run
        )
    for lockfile in LOCKFILES:
        assert lockfile in snapshot_run, lockfile
    # Cargo.lock's baseline is the snapshot the patch step's successor wrote,
    # which is also what makes the check fail closed when it is missing.
    assert "lockfile-digests.json" in snapshot_run
    assert "lockfile-digests.json" in verify_run


def test_the_lockfile_snapshot_only_tolerates_the_release_version_patch(tmp_path):
    workflow = _workflow()
    _fake_lockfile_repo(tmp_path)
    cargo_lock = tmp_path / "studio" / "src-tauri" / "Cargo.lock"

    # Unpatched: the version already matches, so there is nothing to allow.
    result, _ = _run_step(workflow, "build", "Snapshot lockfile digests", tmp_path)
    assert result.returncode == 0, result.stderr
    snapshot = json.loads((tmp_path / "lockfile-digests.json").read_text(encoding = "utf-8"))
    assert set(snapshot) == set(LOCKFILES)
    assert snapshot["studio/src-tauri/Cargo.lock"] == _lockfile_digest(cargo_lock)

    # Patched exactly as "Patch desktop app version" does it.
    cargo_lock.write_text(
        cargo_lock.read_text(encoding = "utf-8").replace(
            'name = "unsloth-studio"\nversion = "0.0.0"',
            'name = "unsloth-studio"\nversion = "0.1.50"',
        ),
        encoding = "utf-8",
    )
    result, _ = _run_step(workflow, "build", "Snapshot lockfile digests", tmp_path)
    assert result.returncode == 0, result.stderr

    # A dependency swapped in beside it is not the version patch.
    cargo_lock.write_text(
        cargo_lock.read_text(encoding = "utf-8").replace(
            'name = "serde"\nversion = "1.0.0"',
            'name = "serde"\nversion = "9.9.9"',
        ),
        encoding = "utf-8",
    )
    result, _ = _run_step(workflow, "build", "Snapshot lockfile digests", tmp_path)
    assert result.returncode == 1
    assert "more than the release version patch" in result.stderr

    # And an npm lockfile the install rewrote fails before Cargo.lock is read.
    _fake_lockfile_repo_reset = tmp_path / "studio" / "package-lock.json"
    _fake_lockfile_repo_reset.write_text('{"name": "rewritten"}\n', encoding = "utf-8")
    result, _ = _run_step(workflow, "build", "Snapshot lockfile digests", tmp_path)
    assert result.returncode == 1


def test_the_post_build_check_catches_a_rewritten_lockfile(tmp_path):
    workflow = _workflow()
    _fake_lockfile_repo(tmp_path)
    snapshot = {
        lockfile: _lockfile_digest(tmp_path / lockfile) for lockfile in LOCKFILES
    }
    digests = tmp_path / "lockfile-digests.json"
    digests.write_text(json.dumps(snapshot), encoding = "utf-8")

    result, _ = _run_step(workflow, "build", "Verify the build rewrote no lockfile", tmp_path)
    assert result.returncode == 0, result.stderr

    # Cargo.lock is measured against the post-patch snapshot, not HEAD, so a
    # build that re-resolved it is caught even though the patch step made it
    # differ from the tag.
    cargo_lock = tmp_path / "studio" / "src-tauri" / "Cargo.lock"
    cargo_lock.write_text(
        cargo_lock.read_text(encoding = "utf-8").replace('version = "1.0.0"', 'version = "9.9.9"'),
        encoding = "utf-8",
    )
    result, _ = _run_step(workflow, "build", "Verify the build rewrote no lockfile", tmp_path)
    assert result.returncode == 1
    assert "The build rewrote a lockfile" in result.stderr

    # A missing snapshot must fail closed rather than read as "nothing changed".
    digests.unlink()
    result, _ = _run_step(workflow, "build", "Verify the build rewrote no lockfile", tmp_path)
    assert result.returncode == 1
    assert "No lockfile digest snapshot" in result.stderr


def test_the_build_inputs_record_never_joins_the_release_asset_artifacts():
    """publish-release and virustotal-scan both merge `desktop-release-*`.

    A per-leg JSON landing in that namespace would be merged into the bundle
    directory and fail the exact-set contract, so it travels under its own name.
    """
    workflow = _workflow()
    upload = _step(workflow, "build", "Upload build input record")
    assert upload["with"]["name"] == "desktop-build-inputs-${{ matrix.artifact }}"
    assert not upload["with"]["name"].startswith("desktop-release-")
    assert upload["with"]["if-no-files-found"] == "error"

    download = _step(workflow, "publish-release", "Download build input records")
    assert download["with"]["pattern"] == "desktop-build-inputs-*"

    # Assembled into the asset directory before the set is validated, so the
    # exact-set check covers the record too.
    names = [step.get("name") for step in _steps(workflow, "publish-release")]
    assert names.index("Download build input records") < names.index("Assemble build-inputs.json")
    assert names.index("Assemble build-inputs.json") < names.index("Validate release asset set")

    assemble = _step(workflow, "publish-release", "Assemble build-inputs.json")["run"]
    # Every leg has to agree on what it built, or the release is a mixed set.
    assert "did not all build the same commit" in assemble
    assert "did not all build from the same lockfiles" in assemble

    # And every leg has to be represented, or the record describes the release
    # only partly. The list is hardcoded, like the wait step's, so tie it to the
    # matrix here: enabling a leg must not be discoverable an hour into a run.
    legs = {
        entry["artifact"]
        for entry in workflow["jobs"]["build"]["strategy"]["matrix"]["include"]
    }
    declared = re.search(r"EXPECTED_LEGS = \{([^}]*)\}", assemble)
    assert declared, assemble
    assert {
        value.strip().strip("'") for value in declared.group(1).split(",") if value.strip()
    } == legs
    for field in ("pypi_version", "commit", "source_date_epoch", "lockfiles"):
        assert field in assemble, field

    # Everything the release was built with, in the one place that outlives the
    # runner logs: toolchain versions, lockfile digests and the pinned tool
    # digests and action SHAs, read out of the workflow itself so a bumped pin
    # cannot be recorded as the old one.
    record = _step(workflow, "build", "Record build inputs")["run"]
    for probe in (
        "rustc -V",
        "cargo -V",
        "node -v",
        "npm -v",
        "tauri --version",
        "git rev-parse HEAD",
        "pinned_actions",
        "pinned_downloads",
        "_SHA256",
        "ImageVersion",
    ):
        assert probe in record, probe


def test_windows_release_build_restores_but_does_not_save_rust_cache():
    cache = _step(_workflow(), "build", "Rust cache")
    assert cache["with"]["workspaces"] == "studio/src-tauri -> target"
    assert cache["with"]["save-if"] == "${{ matrix.platform != 'windows-latest' }}"


def _write_fake_gh(path: Path):
    """Record gh arguments and return configured statuses."""
    path.write_text(
        """#!/bin/sh
set -eu
printf 'gh %s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "api" ]; then
  include=0
  endpoint=""
  for argument in "$@"; do
    case "$argument" in
      --include) include=1 ;;
      repos/*) endpoint="$argument" ;;
    esac
  done
  case "$endpoint" in
    */commits/*) printf '%s\n' "$SOURCE_COMMIT_SHA"; exit 0 ;;
    */releases/tags/*) status="$TARGET_HTTP_STATUS" ;;
    *) exit 0 ;;
  esac
  if [ "$include" = "1" ]; then
    printf 'HTTP/2.0 %s Test Response\n' "$status"
  fi
  if [ "$status" = "200" ]; then
    if [ "$TARGET_HAS_DESKTOP_ASSETS" = "1" ]; then
      printf '{"tag_name":"%s","draft":false,"assets":[{"name":"latest.json"}]}\n' "$DESKTOP_RELEASE_TAG"
    else
      printf '{"tag_name":"%s","draft":false,"assets":[]}\n' "$DESKTOP_RELEASE_TAG"
    fi
    exit 0
  fi
  exit 1
fi

if [ "$1" = "release" ] && [ "$2" = "download" ]; then
  if [ "$TARGET_HAS_DESKTOP_ASSETS" != "1" ]; then
    echo "release not found" >&2
    exit 1
  fi
  directory=""
  want_directory=0
  for argument in "$@"; do
    if [ "$want_directory" = "1" ]; then directory="$argument"; want_directory=0; continue; fi
    [ "$argument" = "--dir" ] && want_directory=1
  done
  [ -n "$directory" ] || directory="."
  mkdir -p "$directory"
  printf '{"version":"%s","platforms":{}}\n' "$TARGET_MANIFEST_VERSION" > "$directory/latest.json"
  exit 0
fi

exit 0
""",
        encoding = "utf-8",
    )
    path.chmod(0o755)


def _run_step(
    workflow,
    job: str,
    name: str,
    tmp_path: Path,
    *,
    target_http_status: int = 200,
    target_has_desktop_assets: bool = False,
    target_manifest_version: str = RELEASE_TAG,
    extra_env: dict[str, str] | None = None,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok = True)
    _write_fake_gh(fake_bin / "gh")
    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")

    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "DESKTOP_RELEASE_TAG": RELEASE_TAG,
            "GH_REPO": "unslothai/unsloth",
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "GH_TOKEN": "masked-token",
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ASSET_VERSION": "0_1_50_beta",
            "RUNNER_TEMP": str(tmp_path),
            "SOURCE_COMMIT_SHA": SOURCE_SHA,
            "TARGET_HAS_DESKTOP_ASSETS": "1" if target_has_desktop_assets else "0",
            "TARGET_HTTP_STATUS": str(target_http_status),
            "TARGET_MANIFEST_VERSION": target_manifest_version,
        }
    )
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", "-c", _step(workflow, job, name)["run"]],
        cwd = tmp_path,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )
    return result, log.read_text(encoding = "utf-8").splitlines()


def _stage_assets(tmp_path: Path) -> dict[str, str]:
    """Create release assets and return their digests."""
    asset_dir = tmp_path / "desktop-release-assets"
    asset_dir.mkdir(exist_ok = True)
    signature = base64.b64encode(
        b"untrusted comment: signature from tauri secret key\n"
        b"test signature bytes\n"
        b"trusted comment: timestamp:1\tfile:test\n"
        b"test global signature bytes\n"
    )
    digests = {}
    for name, payload in (
        ("Unsloth-Desktop-0_1_50_beta-MacOS.dmg", b"disk image"),
        ("Unsloth-Desktop-0_1_50_beta-Ubuntu.deb", b"package"),
        ("Unsloth-Desktop-0_1_50_beta-ARM64.app.tar.gz", b"mac updater"),
        ("Unsloth-Desktop-0_1_50_beta-ARM64.app.tar.gz.sig", signature),
        ("Unsloth-Desktop-0_1_50_beta-Linux.AppImage", b"linux updater"),
        ("Unsloth-Desktop-0_1_50_beta-Linux.AppImage.sig", signature),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe", b"installer"),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe.sig", signature),
        # Assembled from the three legs' records before the asset set is
        # validated, so it is on disk for every step that follows. Mirrors
        # "Validate release asset set" -- see the test below, which runs that
        # step against exactly this directory.
        ("build-inputs.json", b'{"schema": "unsloth-desktop-build-inputs/1"}'),
    ):
        (asset_dir / name).write_bytes(payload)
        if not name.endswith(".sig"):
            digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _run_create_release(
    workflow,
    tmp_path: Path,
    *,
    invalid_signature = False,
    **kwargs,
):
    _stage_assets(tmp_path)
    if invalid_signature:
        (
            tmp_path / "desktop-release-assets" / "Unsloth-Desktop-0_1_50_beta-Linux.AppImage.sig"
        ).write_text("Tauri signer diagnostic, not a signature\n", encoding = "utf-8")
    env = {
        "DESKTOP_RELEASE_NOTES": workflow["env"]["DESKTOP_RELEASE_NOTES"],
        "APP_VERSION": "0.1.50",
        "GITHUB_SHA": SOURCE_SHA,
        "GITHUB_REPOSITORY": "unslothai/unsloth",
        "PYPI_VERSION": "2026.8.7",
        "RELEASE_DRAFT": "true",
        "STUDIO_VERSION": "v0.1.50-beta",
    }
    env.update(kwargs.pop("extra_env", None) or {})

    # Execute the production publish sequence in one shell so the notes and
    # metadata files cross the same step boundaries as Actions.
    names = (
        "Validate versioned release state",
        "Generate versioned updater metadata",
    )
    host = "Generate versioned updater metadata"
    create_step = _step(workflow, "publish-release", host)
    create_step["run"] = "\n".join(
        _step(workflow, "publish-release", name)["run"] for name in names
    )
    return _run_step(
        workflow,
        "publish-release",
        host,
        tmp_path,
        extra_env = env,
        **kwargs,
    )


def _upload_commands(workflow):
    commands = []
    for step in _steps(workflow, "publish-release"):
        # Join backslash continuations so a flag parked on the next line counts.
        for line in step.get("run", "").replace("\\\n", " ").splitlines():
            stripped = line.strip()
            if stripped.startswith("gh release upload"):
                commands.append(stripped)
    return commands


def test_a_used_version_fails_the_guard_before_any_build_work(tmp_path):
    workflow = _workflow()
    # Fail before the build matrix and notarization.
    assert _step_index(
        workflow, "prepare-version", "Guard against republishing an existing version"
    ) < _step_index(workflow, "prepare-version", "Verify PyPI package and Unsloth stamp")
    assert workflow["jobs"]["build"]["needs"] == "prepare-version"

    for case, expected in (
        ({"target_has_desktop_assets": True}, 1),
        ({"target_http_status": 404}, 1),
        ({}, 0),
    ):
        case_dir = tmp_path / ("-".join(case) or "unused-version")
        case_dir.mkdir()
        result, _ = _run_step(
            workflow,
            "prepare-version",
            "Guard against republishing an existing version",
            case_dir,
            **case,
        )
        assert result.returncode == expected, (case, result.stderr)
        if expected:
            assert RELEASE_TAG in result.stderr


def test_a_missing_target_release_says_how_to_create_it(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_http_status = 404,
    )
    assert result.returncode == 1
    assert f"Release {RELEASE_TAG} does not exist." in result.stderr
    assert "Tag main and publish it first" in result.stderr


def test_existing_desktop_assets_name_the_cleanup_command(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_has_desktop_assets = True,
    )
    assert result.returncode == 1
    assert f"gh release delete-asset {RELEASE_TAG} latest.json --yes" in result.stderr


def test_a_failed_guard_probe_fails_closed_before_any_build_work(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_http_status = 500,
    )
    assert result.returncode == 1
    assert "Could not read release" in result.stderr


def test_publish_refuses_to_reuse_an_existing_release(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, target_has_desktop_assets = True)
    assert result.returncode == 1
    assert "Refusing to republish" in result.stderr
    assert f"gh release delete-asset {RELEASE_TAG} latest.json --yes" in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_publish_fails_closed_when_the_target_release_is_missing(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, target_http_status = 404)
    assert result.returncode == 1
    assert f"Release {RELEASE_TAG} does not exist." in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_publish_rejects_signer_diagnostics_as_updater_signatures(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path, invalid_signature = True)
    assert result.returncode == 1
    assert "Invalid base64 updater signature" in result.stderr
    assert not [line for line in commands if line.startswith("gh release create")]


def test_the_publish_sequence_never_rewrites_the_release_body(tmp_path):
    workflow = _workflow()
    result, commands = _run_create_release(workflow, tmp_path)
    assert result.returncode == 0, result.stderr

    # The release already exists, so nothing is created and no tag is reserved.
    assert not [line for line in commands if line.startswith("gh release create")]
    assert not [line for line in commands if "git/refs" in line]
    # The body is the maintainer's changelog. Assets are uploaded beside it and
    # the notes are never edited, so nothing this workflow does can clobber it.
    assert not [line for line in commands if line.startswith("gh release edit")]
    assert not (tmp_path / "desktop-release-body.md").exists()

    latest = tmp_path / "latest.json"
    assert latest.is_file()
    metadata = yaml.safe_load(latest.read_text(encoding = "utf-8"))
    for platform in metadata["platforms"].values():
        decoded = base64.b64decode(platform["signature"], validate = True)
        assert decoded.startswith(b"untrusted comment:")
        assert b"\ntrusted comment:" in decoded

    # The updater popup shows the maintainer notes, never build metadata.
    notes = (tmp_path / "desktop-release-notes.md").read_text(encoding = "utf-8")
    assert "Build provenance" not in notes
    assert "Desktop app for Unsloth." in notes


def test_versioned_uploads_never_clobber_or_mutate_the_legacy_channel():
    uploads = _upload_commands(_workflow())
    versioned = [line for line in uploads if "$DESKTOP_RELEASE_TAG" in line]
    channel = [line for line in uploads if "desktop-latest" in line]
    assert len(versioned) == 2, uploads
    assert channel == [], uploads

    # latest.json is the moving updater pointer and may already hold a carried
    # forward manifest, so only it may be replaced. Bundles stay immutable.
    for line in versioned:
        if "latest.json" not in line:
            assert "--clobber" not in line, line


def test_a_carried_forward_manifest_does_not_block_the_guard(tmp_path):
    workflow = _workflow()
    result, _ = _run_step(
        workflow,
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
        target_has_desktop_assets = True,
        target_manifest_version = "v0.1.49-beta",
    )
    assert result.returncode == 0, result.stderr


def test_a_validation_only_run_touches_nothing_public():
    steps = _workflow()["jobs"]["publish-release"]["steps"]
    names = [step.get("name") for step in steps]
    mutating = (
        # An attestation is a write too: it publishes a signed statement about
        # these files into the repository's attestation store, where `gh
        # attestation verify` finds it. A validation-only run must leave no
        # record that a release happened, so it is gated like the uploads.
        "Attest build provenance for the release assets",
        "Publish versioned release assets",
        "Publish versioned updater metadata",
        "Promote normal release to GitHub latest",
    )
    for name in mutating:
        step = steps[names.index(name)]
        assert step.get("if") == "${{ !inputs.draft }}", name

    # Promotion last, so latest only moves once the assets are actually on the
    # release and a partial upload cannot leave latest pointing at an empty one.
    for earlier in mutating[:-1]:
        assert names.index(earlier) < names.index(mutating[-1])


def test_every_published_asset_is_attested_before_it_is_uploaded():
    """The attestation is what lets a third party check where a bundle came from.

    It has to cover the validated set and nothing else, so it runs after
    "Validate release asset set" (which is an exact-set check, so an extra file
    cannot ride along) and before the upload, so no file reaches the release
    without one.
    """
    workflow = _workflow()
    names = [step.get("name") for step in _steps(workflow, "publish-release")]
    attest = names.index("Attest build provenance for the release assets")
    assert names.index("Validate release asset set") < attest
    assert attest < names.index("Publish versioned release assets")

    # Same directory the upload reads from, so the subjects are the files that
    # are published rather than a separately assembled list.
    step = _step(workflow, "publish-release", "Attest build provenance for the release assets")
    assert step["with"]["subject-path"] == "${{ runner.temp }}/desktop-release-assets/*"
    upload = _step(workflow, "publish-release", "Publish versioned release assets")
    assert '"$RUNNER_TEMP/desktop-release-assets"/*' in upload["run"]


def test_the_asset_set_validator_accepts_exactly_the_staged_set(tmp_path):
    """The exact-set check and the fixture above must not drift apart.

    "Validate release asset set" is the gate that stops an unexpected file being
    published beside the bundles, so run the real step against the staged
    directory: it passes on exactly that set, and fails on one file more or one
    file fewer.
    """
    workflow = _workflow()

    _stage_assets(tmp_path)
    result, _ = _run_step(workflow, "publish-release", "Validate release asset set", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "build-inputs.json" in result.stdout

    assets = tmp_path / "desktop-release-assets"
    (assets / "unexpected.txt").write_text("stowaway", encoding = "utf-8")
    result, _ = _run_step(workflow, "publish-release", "Validate release asset set", tmp_path)
    assert result.returncode == 1
    assert "unexpected=['unexpected.txt']" in result.stderr
    (assets / "unexpected.txt").unlink()

    (assets / "build-inputs.json").unlink()
    result, _ = _run_step(workflow, "publish-release", "Validate release asset set", tmp_path)
    assert result.returncode == 1
    assert "missing=['build-inputs.json']" in result.stderr


def test_the_guard_rejects_a_prerelease_target_before_anything_is_built():
    workflow = _workflow()
    guard = _step(workflow, "prepare-version", "Guard against republishing an existing version")
    assert "is a prerelease" in guard["run"]
    # And again in publish-release, which is the one holding write scope.
    state = _step(workflow, "publish-release", "Validate versioned release state")
    assert "is a prerelease" in state["run"]


def test_the_build_uses_the_release_tag_not_the_dispatch_ref():
    build = _workflow()["jobs"]["build"]["steps"]
    checkout = next(s for s in build if "actions/checkout" in str(s.get("uses", "")))
    assert checkout["with"]["ref"] == "${{ needs.prepare-version.outputs.desktop_release_tag }}"


def test_the_tag_is_validated_before_it_is_checked_out(tmp_path):
    # actions/checkout resolves the free-text input, so a malformed tag would fail
    # on a generic missing-ref error and none of the corrections would be printed.
    steps = _workflow()["jobs"]["prepare-version"]["steps"]
    names = [step.get("name") or str(step.get("uses")) for step in steps]
    checkout = next(
        i for i, step in enumerate(steps) if "actions/checkout" in str(step.get("uses", ""))
    )
    assert names.index("Validate release versions") < checkout, names
    # And the checkout uses the validated value, not the raw input.
    assert steps[checkout]["with"]["ref"] == "${{ steps.prepare.outputs.studio_version }}"

    for index, (bad, expected) in enumerate(
        (
            ("v.0.1.52-beta", "did you mean v0.1.52-beta?"),
            ("0.1.52-beta", "must start with v"),
            ("2026.8.3", "not a date-style backend version"),
        )
    ):
        case_dir = tmp_path / f"case-{index}"
        case_dir.mkdir()
        result, _ = _run_step(
            _workflow(),
            "prepare-version",
            "Validate release versions",
            case_dir,
            extra_env = {"INPUT_STUDIO_VERSION": bad},
        )
        assert result.returncode == 1, bad
        assert expected in result.stderr, (bad, result.stderr)


def test_the_promotion_guard_orders_numbered_prereleases_by_number():
    guard = _step(_workflow(), "publish-release", "Promote normal release to GitHub latest")["run"]
    body = guard.split('python3 - "$latest_before"', 1)[1].split("\nPY", 1)[0]
    body = "\n".join(line[10:] if line.startswith(" " * 10) else line for line in body.split("\n"))
    body = body.split("\n", 1)[1].lstrip("\n")
    namespace: dict = {}
    exec(body.split("current = json.loads", 1)[0], namespace)
    key = namespace["key"]
    # v1.2.3-beta10 is newer than v1.2.3-beta2, and a release beats its prerelease.
    assert key("v1.2.3-beta10") > key("v1.2.3-beta2")
    assert key("v1.2.3") > key("v1.2.3-beta10")
    assert key("v0.1.527-beta") > key("v0.1.526-beta")
    assert key("not-a-tag") is None


def test_the_promotion_guard_fails_closed_on_a_failed_latest_lookup():
    guard = _step(_workflow(), "publish-release", "Promote normal release to GitHub latest")["run"]
    # A 404 means no latest yet; anything else must stop before the PATCH.
    fallback = guard.split("elif grep -Fq '(HTTP 404)'", 1)[1].split("gh api --method PATCH", 1)[0]
    assert "refusing to promote" in fallback.lower()
    assert "exit 1" in fallback
    assert "2>/dev/null" not in guard.split("releases/latest", 1)[1].split("\n", 1)[0]


def _guarded_bodies(script, header):
    """Return the body of every `header` block, delimited by matching braces."""
    bodies = []
    at = script.find(header)
    while at != -1:
        start = at + len(header)
        depth = 1
        for index in range(start, len(script)):
            if script[index] == "{":
                depth += 1
            elif script[index] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append(script[start:index])
                    break
        else:
            raise AssertionError(f"unbalanced braces after {header!r}")
        at = script.find(header, start)
    assert bodies, f"{header!r} is gone"
    return bodies


def test_dead_defender_cmdlets_do_not_skip_the_bundle_scan():
    """Dead cmdlets must not read as "no scanner"; only a dead engine may.

    The escape hatch added for a one-off runner incident became the permanent
    path: the Defender WMI provider and service RPC endpoint have been down on
    every Windows runner since 2026-08-06, so `Get-MpComputerStatus` throws and
    three releases shipped unscanned. MpCmdRun.exe answers independently of the
    cmdlets, so an unavailable cmdlet surface may only cost the configuration
    checks, never the scan itself.
    """
    scan = _step(_workflow(), "build", "Scan Windows bundles with Defender")["run"]

    # The unavailable branch records the fact and keeps going.
    unavailable = scan.split("$cmdletsDown = [bool]$unavailable", 1)
    assert len(unavailable) == 2, "the cmdlet-unavailable branch no longer sets $cmdletsDown"
    before_control = unavailable[1].split("EICAR positive control", 1)[0]
    assert (
        "exit 0" not in before_control
    ), "unavailable cmdlets still short-circuit the scan before the positive control"

    # The two cmdlets fail independently, so each probe must sit under its OWN
    # guard, not merely some guard: pooling both bodies would accept
    # $pref.MAPSReporting under `if ($status)`, where a dead status cmdlet again
    # discards a readable MAPSReporting=0 and scans blind to the "!ml" cloud
    # verdicts this gate exists to catch.
    guards = {
        "$status": _guarded_bodies(scan, "if ($status) {"),
        "$pref": _guarded_bodies(scan, "if ($pref) {"),
    }
    for probe in (
        "$status.RealTimeProtectionEnabled",
        "$pref.MAPSReporting",
        "$pref.DisableBlockAtFirstSeen",
        "$pref.SubmitSamplesConsent",
        "$pref.CloudBlockLevel",
        "$pref.ExclusionPath",
    ):
        owner = probe.split(".", 1)[0]
        assert any(
            probe in body for body in guards[owner]
        ), f"{probe} left the `if ({owner})` guard that proves it was read"
        for other, bodies in guards.items():
            if other != owner and any(probe in body for body in bodies):
                raise AssertionError(
                    f"{probe} is gated on `if ({other})`, which fails independently "
                    f"of {owner}; one dead cmdlet would discard the other cmdlet's "
                    "readable result"
                )
    outside = scan
    for bodies in guards.values():
        for body in bodies:
            outside = outside.replace(body, "", 1)
    for held in ("$status.", "$pref."):
        assert held not in outside, f"a {held[:-1]} dereference sits outside its availability guard"

    config = scan.split("$fatal = @()", 1)[1].split("# Configuration is not connectivity", 1)[0]
    assert "$cmdletsDown" not in config, (
        "the configuration checks are gated on the blanket flag again; one dead "
        "cmdlet would discard the other cmdlet's readable result"
    )

    # The only remaining skip: a control that will not fire, the one signal that
    # MpCmdRun cannot scan either.
    skip = scan.split("MpCmdRun could not fire the EICAR positive control", 1)
    assert len(skip) == 2, "the missing-scanner skip no longer keys off the positive control"
    assert "exit 0" in skip[1].split("\n", 3)[1] + skip[1].split("\n", 3)[2]
    assert "not a clean verdict" in skip[0].rsplit("::warning::", 1)[1] + skip[1]

    # A detection still fails the job, cmdlets or not.
    assert "Refusing to publish a Windows bundle Defender flags" in scan
    assert "Refusing to publish bundles Defender could not scan" in scan


def test_a_sample_quarantined_mid_scan_passes_the_positive_control():
    """A sample that vanishes during the scan is a live engine, not a missing one.

    Defender remediates asynchronously and MpCmdRun opening the sample is itself
    the trigger, so the write can succeed, `Test-Path` can see the file, and
    real-time protection can quarantine it mid-scan. MpCmdRun then reports no
    threat, `$controlPassed` stays false, and with the cmdlets down the skip branch
    exits 0, publishing every bundle unscanned on a runner whose scanner just
    proved itself. Only a sample that survives means no scanner.
    """
    scan = _step(_workflow(), "build", "Scan Windows bundles with Defender")["run"]

    body = _guarded_bodies(scan, "if (Test-Path $eicarPath) {")[0]
    _, scanned, after = body.partition("-DisableRemediation")
    assert scanned, "the positive control no longer scans the sample with MpCmdRun"
    # The re-check has to land after the scan and before this step's own cleanup,
    # or it proves nothing about who removed the file.
    recheck, cleaned, _ = after.partition("Remove-Item $eicarPath")
    assert cleaned, "the positive control no longer removes the sample afterwards"
    assert "-not (Test-Path $eicarPath)" in recheck, (
        "the positive control never re-checks the sample after the scan, so a "
        "sample quarantined mid-scan reads as a missing scanner and skips the "
        "bundle scan on a runner where Defender is demonstrably live"
    )
    assert (
        "$controlPassed = $true" in recheck
    ), "the vanished sample is noticed but still does not pass the control"
    # Only a vanished sample may pass this way. -DisableRemediation stops the scan
    # from deleting the file, so with no engine it survives and the skip applies.
    assert recheck.index("-not (Test-Path $eicarPath)") < recheck.index(
        "$controlPassed = $true"
    ), "the control passes without first confirming the sample is gone"
