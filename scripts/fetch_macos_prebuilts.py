#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fetch the non-Python halves of the bundled macOS arm64 runtime.

``scripts/build_macos_runtime.sh`` owns the Python side (locks + uv). This owns
everything that arrives as a prebuilt archive: CPython itself, llama.cpp,
whisper.cpp, Node, and -- behind ``--with-sd-cpp`` -- stable-diffusion.cpp.

WHY A SEPARATE SCRIPT AND NOT THE INSTALLERS

``studio/install_llama_prebuilt.py`` and friends resolve the asset from the HOST
they are running on: ``detect_host()`` reads ``platform.system()`` /
``platform.machine()`` and probes for NVIDIA, ROCm and Intel GPUs. That is right
for an install and wrong for a cross-build, where the answer must be "macOS
arm64" regardless of the machine doing the building -- including the Linux CI
runner this is exercised on. There is no supported way to lie to ``detect_host``.

So the HOST DECISION is reimplemented here (as a constant: ``macos-arm64``), and
nothing else is. The TRUST CHAIN is imported from ``studio/prebuilt_core.py`` and
runs exactly as it does at install time:

    in-tree digest (prebuilt_release_pins.json)
      -> release checksum index, verified against it
        -> per-archive sha256, read out of that index
          -> archive on disk

``prebuilt_core.verify_pinned_checksum_index`` is the link the release host cannot
forge, and it is called with an EXPLICIT expected digest here so that the
``UNSLOTH_PREBUILT_ALLOW_LATEST`` / ``ALLOW_UNVERIFIED`` escape hatches, which turn
verification into a no-op at install time, cannot turn it into a no-op in a build
that ships signed inside an app. Node and CPython have no checksum index; their
digests are frozen in-tree (``studio/node_prebuilt_pins.json``,
``studio/macos_runtime_pins.json``) and compared directly.

Every fetch fails closed. There is no unverified path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_DIR = REPO_ROOT / "studio"

sys.path.insert(0, str(STUDIO_DIR))

import prebuilt_core  # noqa: E402  (needs the sys.path line above)

# The one host this script builds for. Not detected -- decided.
TARGET_OS = "macos"
TARGET_ARCH = "arm64"

# The artifact `kind` each component publishes for that host, as it appears in the
# release's checksum index. Matching on `kind` rather than on a filename pattern is
# deliberate: the filename carries the release tag, so a tag bump would silently
# stop matching, while `kind` is the release's own stable description of the host it
# targets.
LLAMA_ARTIFACT_KIND = "macos-arm64-app"
WHISPER_ARTIFACT_KIND = "macos-arm64-slim-bundle"

# Node's asset name is a pure function of version + host, exactly as
# install_node_prebuilt.node_asset_name builds it.
NODE_DIST_BASE = "https://nodejs.org/dist"
NODE_ASSET_TEMPLATE = "node-v{version}-darwin-arm64.tar.gz"

USER_AGENT = "unsloth-macos-runtime-build"


class FetchError(RuntimeError):
    """A fetch or verification failed. Always fatal; there is no fallback."""


# ── plumbing ──────────────────────────────────────────────────────────────────
def _log(message: str) -> None:
    print(f"[prebuilts] {message}", flush = True)


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers = {"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout = 120) as response:  # noqa: S310 (https only, below)
        if not url.startswith("https://"):
            raise FetchError(f"refusing a non-https fetch: {url}")
        return response.read()


def _download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise FetchError(f"refusing a non-https download: {url}")
    destination.parent.mkdir(parents = True, exist_ok = True)
    request = urllib.request.Request(url, headers = {"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout = 600) as response:  # noqa: S310
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length = 1024 * 1024)


def _verify(path: Path, expected_sha256: str, *, label: str) -> str:
    """sha256 ``path`` and refuse anything but ``expected_sha256``."""
    want = (expected_sha256 or "").strip().lower()
    if len(want) != 64 or any(character not in "0123456789abcdef" for character in want):
        raise FetchError(f"{label}: expected sha256 is not a 64-hex digest: {expected_sha256!r}")
    got = prebuilt_core.sha256_file(path)
    if got != want:
        raise FetchError(
            f"{label}: sha256 mismatch. expected {want}, got {got}. "
            f"The build stops rather than baking unverified bytes into a signed app."
        )
    _log(f"verified {label} sha256={got}")
    return got


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FetchError(f"missing required pins file: {path}")
    return json.loads(path.read_text(encoding = "utf-8"))


def _extract(archive: Path, destination: Path) -> Path:
    """Safely extract ``archive`` and return the directory holding its content.

    Uses prebuilt_core.extract_archive (path traversal, symlink escape and member
    type checks) and then collapses a single top-level directory, which is how
    llama.cpp, Node and CPython all ship and which the install-time flow collapses
    the same way.
    """
    destination.mkdir(parents = True, exist_ok = True)
    prebuilt_core.extract_archive(archive, destination)
    prebuilt_core.restore_tar_exec_bits(archive, destination)
    entries = [entry for entry in destination.iterdir() if entry.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
        return entries[0]
    return destination


def _copy_tree_into(source: Path, destination: Path) -> None:
    destination.mkdir(parents = True, exist_ok = True)
    for entry in sorted(source.iterdir()):
        target = destination / entry.name
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, target, symlinks = True)
        else:
            shutil.copy2(entry, target, follow_symlinks = False)


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ── the ggml family: llama.cpp and whisper.cpp ────────────────────────────────
def _pinned_release(component: str) -> dict[str, Any]:
    pins = _read_json(STUDIO_DIR / prebuilt_core.RELEASE_PINS_FILENAME)
    try:
        entry = pins["components"][component]
    except (KeyError, TypeError) as error:
        raise FetchError(
            f"{prebuilt_core.RELEASE_PINS_FILENAME} has no components.{component} entry"
        ) from error
    for field in ("repo", "release_tag", "checksum_index_asset", "checksum_index_sha256"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise FetchError(
                f"{prebuilt_core.RELEASE_PINS_FILENAME}: components.{component}.{field} "
                f"is missing or empty, so there is no trust anchor for this component"
            )
    return entry


def _verified_checksum_index(component: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The release's checksum index, digest-checked against the in-tree pin.

    Returns ``(pin_entry, index)``. The expected digest is passed explicitly so no
    environment variable can turn this into a no-op -- unlike the install-time path,
    a build has no user to fall back for.
    """
    entry = _pinned_release(component)
    repo = entry["repo"]
    tag = entry["release_tag"]
    asset = entry["checksum_index_asset"]
    url = prebuilt_core.release_asset_download_url(repo, tag, asset)
    _log(f"{component}: fetching {asset} from {repo}@{tag}")
    raw = _http_get(url)
    prebuilt_core.verify_pinned_checksum_index(
        repo,
        tag,
        raw,
        checksum_index_asset = asset,
        expected = entry["checksum_index_sha256"],
    )
    _log(f"{component}: checksum index matches the in-tree digest")
    index = json.loads(raw.decode("utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("artifacts"), dict):
        raise FetchError(f"{component}: checksum index has no artifacts map")
    return entry, index


def _artifact_by_kind(component: str, index: dict[str, Any], kind: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, meta)
        for name, meta in sorted(index["artifacts"].items())
        if isinstance(meta, dict) and meta.get("kind") == kind
    ]
    if not matches:
        available = sorted(
            {
                str(meta.get("kind"))
                for meta in index["artifacts"].values()
                if isinstance(meta, dict)
            }
        )
        raise FetchError(
            f"{component}: the pinned release publishes no artifact of kind {kind!r} "
            f"for {TARGET_OS}-{TARGET_ARCH}. Kinds present: {available}"
        )
    if len(matches) > 1:
        raise FetchError(
            f"{component}: {len(matches)} artifacts claim kind {kind!r} "
            f"({[name for name, _ in matches]}); refusing to guess"
        )
    return matches[0]


def _install_ggml_component(
    *,
    component: str,
    artifact_kind: str,
    marker_filename: str,
    server_binary: str,
    destination: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Lay out ``destination`` the way the installer lays out ~/.unsloth/<component>.

    That layout is ``<root>/build/bin/<server + co-located dylibs>`` plus a marker
    JSON at the root -- see prebuilt_core.assemble_install_tree and
    install_whisper_prebuilt.runtime_bin_dir. Reproducing it exactly is the point:
    the app must find the bundled copy through the same paths it already knows.
    """
    entry, index = _verified_checksum_index(component)
    asset_name, artifact = _artifact_by_kind(component, index, artifact_kind)
    expected = artifact.get("sha256")
    if not isinstance(expected, str):
        raise FetchError(f"{component}: {asset_name} carries no sha256 in the checksum index")

    url = prebuilt_core.release_asset_download_url(entry["repo"], entry["release_tag"], asset_name)
    archive = work_dir / asset_name
    _log(f"{component}: downloading {asset_name}")
    _download(url, archive)
    digest = _verify(archive, expected, label = f"{component}/{asset_name}")

    unpacked = _extract(archive, work_dir / f"{component}-unpacked")
    bin_dir = destination / "build" / "bin"
    if destination.exists():
        shutil.rmtree(destination)
    bin_dir.mkdir(parents = True)
    for entry_path in sorted(unpacked.iterdir()):
        if entry_path.name == marker_filename:
            # The archive's own copy of the marker is superseded by the one written
            # below, which records the asset and digest this build actually used.
            continue
        target = bin_dir / entry_path.name
        if entry_path.is_dir() and not entry_path.is_symlink():
            shutil.copytree(entry_path, target, symlinks = True)
        else:
            shutil.copy2(entry_path, target, follow_symlinks = False)

    server = bin_dir / server_binary
    if not server.is_file():
        raise FetchError(f"{component}: {asset_name} did not contain {server_binary}")
    _make_executable(server)

    marker = {
        "schema_version": 1,
        "component": index.get("component", component),
        "published_repo": entry["repo"],
        "release_tag": entry["release_tag"],
        "upstream_tag": index.get("upstream_tag"),
        "source_commit": index.get("source_commit"),
        "asset": asset_name,
        "asset_sha256": digest,
        "backend": "metal",
        "studio_protocol": index.get("studio_protocol"),
        # Deliberately no install_fingerprint. That field is what
        # prebuilt_core.existing_install_matches uses to decide an install is
        # already current, and only the installer may compute it. Omitting it means
        # a user who explicitly asks to update gets a normal install into the
        # writable root rather than a silent no-op against a read-only bundle.
        "bundled_in_app": True,
        "bundle_source": "scripts/fetch_macos_prebuilts.py",
    }
    if component == "whisper_cpp":
        marker["install_kind"] = "slim"
        marker["paired_llama_tag"] = index.get("paired_llama_tag")
    (destination / marker_filename).write_text(
        json.dumps(marker, indent = 2) + "\n", encoding = "utf-8"
    )

    return {
        "repo": entry["repo"],
        "release_tag": entry["release_tag"],
        "upstream_tag": index.get("upstream_tag"),
        "source_commit": index.get("source_commit"),
        "asset": asset_name,
        "asset_sha256": digest,
        "asset_url": url,
        "checksum_index_asset": entry["checksum_index_asset"],
        "checksum_index_sha256": entry["checksum_index_sha256"],
        "layout": "build/bin",
        "server_binary": f"build/bin/{server_binary}",
    }


# ── Node ──────────────────────────────────────────────────────────────────────
def _install_node(*, destination: Path, work_dir: Path) -> dict[str, Any]:
    pins = _read_json(STUDIO_DIR / "node_prebuilt_pins.json")
    version = pins.get("default_version")
    if not isinstance(version, str) or not version.strip():
        raise FetchError("node_prebuilt_pins.json has no default_version")
    asset = NODE_ASSET_TEMPLATE.format(version = version)
    try:
        expected = pins["versions"][version][asset]
    except (KeyError, TypeError) as error:
        raise FetchError(
            f"node_prebuilt_pins.json has no sha256 for {asset}; the pins file is the "
            f"only trust anchor for a nodejs.org download, so the build stops"
        ) from error

    url = f"{NODE_DIST_BASE}/v{version}/{asset}"
    archive = work_dir / asset
    _log(f"node: downloading {asset}")
    _download(url, archive)
    digest = _verify(archive, expected, label = f"node/{asset}")

    unpacked = _extract(archive, work_dir / "node-unpacked")
    if destination.exists():
        shutil.rmtree(destination)
    _copy_tree_into(unpacked, destination)

    node_binary = destination / "bin" / "node"
    if not node_binary.is_file():
        raise FetchError(f"node: {asset} did not contain bin/node")
    _make_executable(node_binary)
    npm_cli = destination / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    if not npm_cli.is_file():
        raise FetchError(f"node: {asset} did not contain lib/node_modules/npm")

    return {
        "version": version,
        "asset": asset,
        "asset_sha256": digest,
        "asset_url": url,
        "pins_file": "studio/node_prebuilt_pins.json",
    }


# ── CPython ───────────────────────────────────────────────────────────────────
def _install_cpython(*, destination: Path, work_dir: Path) -> dict[str, Any]:
    pins = _read_json(STUDIO_DIR / "macos_runtime_pins.json")
    try:
        entry = pins["components"]["cpython"]
    except (KeyError, TypeError) as error:
        raise FetchError("macos_runtime_pins.json has no components.cpython entry") from error
    for field in ("repo", "release", "python_version", "asset", "sha256"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise FetchError(f"macos_runtime_pins.json: components.cpython.{field} is missing")

    asset = entry["asset"]
    # The '+' in a python-build-standalone asset name is a literal in the path and
    # must be percent-encoded, or GitHub serves a 404.
    quoted = asset.replace("+", "%2B")
    url = f"https://github.com/{entry['repo']}/releases/download/{entry['release']}/{quoted}"
    archive = work_dir / asset.replace("+", "_")
    _log(f"cpython: downloading {asset}")
    _download(url, archive)
    digest = _verify(archive, entry["sha256"], label = f"cpython/{asset}")

    unpacked = _extract(archive, work_dir / "cpython-unpacked")
    if destination.exists():
        shutil.rmtree(destination)
    _copy_tree_into(unpacked, destination)

    interpreter = destination / "bin" / "python3"
    if not interpreter.exists():
        raise FetchError(f"cpython: {asset} did not contain bin/python3")
    real = destination / "bin" / f"python{entry['python_minor']}"
    if not real.is_file():
        raise FetchError(f"cpython: {asset} did not contain bin/python{entry['python_minor']}")
    _make_executable(real)

    return {
        "repo": entry["repo"],
        "release": entry["release"],
        "python_version": entry["python_version"],
        "python_minor": entry["python_minor"],
        "asset": asset,
        "asset_sha256": digest,
        "asset_url": url,
        "checksum_asset": entry.get("checksum_asset"),
        "pins_file": "studio/macos_runtime_pins.json",
    }


# ── stable-diffusion.cpp (opt in) ─────────────────────────────────────────────
def _install_sd_cpp(*, destination: Path, work_dir: Path) -> dict[str, Any]:
    """Fetch the sd-cli prebuilt for macOS arm64.

    Off by default, and behind its own flag, because sd.cpp is the one prebuilt in
    this tree with NO in-tree digest anchor: install_sd_cpp_prebuilt.py verifies
    against the ``digest`` field GitHub publishes alongside the asset, which arrives
    over the same channel as the asset. That is fine for an opportunistic install
    and is not the standard the other three components are held to here, so a build
    only ships it when asked. See the report in the PR that added this file.
    """
    sys.path.insert(0, str(STUDIO_DIR))
    import install_sd_cpp_prebuilt as sd  # noqa: PLC0415  (optional component)

    tag = sd.DEFAULT_TAG
    repo = sd.DEFAULT_REPO
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    _log(f"sd.cpp: resolving {repo}@{tag}")
    release = json.loads(_http_get(api).decode("utf-8"))
    assets = release.get("assets") or []
    names = [asset["name"] for asset in assets]
    chosen = sd.resolve_release_asset(names, system = "Darwin", machine = "arm64")
    if not chosen:
        raise FetchError(f"sd.cpp: {repo}@{tag} publishes no {TARGET_OS}-{TARGET_ARCH} asset")
    asset = next(item for item in assets if item["name"] == chosen)
    expected = (asset.get("digest") or "").split(":", 1)[-1]
    if not expected:
        raise FetchError(
            f"sd.cpp: {chosen} was published without a sha256 digest, so this download "
            f"cannot be verified"
        )
    archive = work_dir / chosen
    _download(asset["browser_download_url"], archive)
    digest = _verify(archive, expected, label = f"sd.cpp/{chosen}")

    unpacked = _extract(archive, work_dir / "sd-unpacked")
    if destination.exists():
        shutil.rmtree(destination)
    _copy_tree_into(unpacked, destination)
    binaries = sorted(destination.rglob("sd-cli"))
    if not binaries:
        raise FetchError(f"sd.cpp: {chosen} did not contain an sd-cli binary")
    _make_executable(binaries[0])

    return {
        "repo": repo,
        "release_tag": tag,
        "asset": chosen,
        "asset_sha256": digest,
        "asset_url": asset["browser_download_url"],
        "digest_source": "github release asset digest (no in-tree anchor)",
    }


# ── entry point ───────────────────────────────────────────────────────────────
COMPONENT_ORDER = ("cpython", "llama.cpp", "whisper.cpp", "node", "sd.cpp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Fetch the verified macOS arm64 prebuilts for the app bundle runtime."
    )
    parser.add_argument(
        "--runtime-dir",
        required = True,
        help = "the runtime/ directory to populate (python/, llama.cpp/, whisper.cpp/, node/)",
    )
    parser.add_argument(
        "--report",
        required = True,
        help = "write the per-component provenance JSON here, for BUNDLE_MANIFEST.json",
    )
    parser.add_argument(
        "--work-dir",
        default = "",
        help = "scratch directory for downloads (default: a temp dir removed on exit)",
    )
    parser.add_argument(
        "--only",
        action = "append",
        choices = COMPONENT_ORDER,
        default = None,
        help = "fetch only these components (repeatable); default is all but sd.cpp",
    )
    parser.add_argument(
        "--with-sd-cpp",
        action = "store_true",
        help = "also fetch stable-diffusion.cpp (see _install_sd_cpp for why this is opt-in)",
    )
    args = parser.parse_args(argv)

    runtime_dir = Path(args.runtime_dir).resolve()
    runtime_dir.mkdir(parents = True, exist_ok = True)

    wanted = list(args.only) if args.only else ["cpython", "llama.cpp", "whisper.cpp", "node"]
    if args.with_sd_cpp and "sd.cpp" not in wanted:
        wanted.append("sd.cpp")

    owned_work_dir = not args.work_dir
    work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp())
    work_dir.mkdir(parents = True, exist_ok = True)

    report: dict[str, Any] = {}
    try:
        for component in COMPONENT_ORDER:
            if component not in wanted:
                continue
            if component == "cpython":
                report["python"] = _install_cpython(
                    destination = runtime_dir / "python", work_dir = work_dir
                )
            elif component == "llama.cpp":
                report["llama_cpp"] = _install_ggml_component(
                    component = "llama_cpp",
                    artifact_kind = LLAMA_ARTIFACT_KIND,
                    marker_filename = "UNSLOTH_PREBUILT_INFO.json",
                    server_binary = "llama-server",
                    destination = runtime_dir / "llama.cpp",
                    work_dir = work_dir,
                )
            elif component == "whisper.cpp":
                report["whisper_cpp"] = _install_ggml_component(
                    component = "whisper_cpp",
                    artifact_kind = WHISPER_ARTIFACT_KIND,
                    marker_filename = "UNSLOTH_WHISPER_PREBUILT_INFO.json",
                    server_binary = "whisper-server",
                    destination = runtime_dir / "whisper.cpp",
                    work_dir = work_dir,
                )
            elif component == "node":
                report["node"] = _install_node(
                    destination = runtime_dir / "node", work_dir = work_dir
                )
            elif component == "sd.cpp":
                report["sd_cpp"] = _install_sd_cpp(
                    destination = runtime_dir / "sd.cpp", work_dir = work_dir
                )
    except (FetchError, prebuilt_core.PrebuiltFallback, OSError, ValueError) as error:
        print(f"error: {error}", file = sys.stderr)
        return 1
    finally:
        if owned_work_dir:
            shutil.rmtree(work_dir, ignore_errors = True)

    Path(args.report).write_text(json.dumps(report, indent = 2, sort_keys = True) + "\n", encoding = "utf-8")
    _log(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
