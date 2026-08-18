# SPDX-License-Identifier: AGPL-3.0-only
# Integrity tests for studio/install_sd_cpp_prebuilt.py. No network.
#
# sd-cli is downloaded, extracted and EXECUTED, and it is installed lazily at the
# first image generation rather than during bootstrap. Both facts push the same
# way: an unverified download must stop the install rather than print a warning
# nobody is watching for, and a pinned release that stopped resolving must not be
# quietly replaced by whatever "latest" is today.
#
# These live here rather than next to the other sd.cpp tests in
# studio/backend/tests/ because install_sd_cpp_prebuilt.py is deliberately
# stdlib-only (it runs before the backend package is importable), so this file
# needs none of that package's test dependencies.

import hashlib
import sys
import urllib.error
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import install_sd_cpp_prebuilt as sd  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        "UNSLOTH_SD_CPP_TAG",
        "UNSLOTH_SD_CPP_REPO",
        sd.ALLOW_LATEST_ENV,
        sd.ALLOW_UNVERIFIED_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(sd.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sd.platform, "machine", lambda: "x86_64")


# ── _verify_sha256 fails closed ──
def _asset(tmp_path: Path, data: bytes = b"sd-cli bytes") -> Path:
    path = tmp_path / "asset.zip"
    path.write_bytes(data)
    return path


def test_a_matching_digest_still_passes(tmp_path):
    path = _asset(tmp_path)
    sd._verify_sha256(path, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())


def test_a_mismatched_digest_still_raises(tmp_path):
    path = _asset(tmp_path)
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        sd._verify_sha256(path, "sha256:" + hashlib.sha256(b"something else").hexdigest())


def test_a_missing_digest_now_fails_instead_of_warning(tmp_path):
    # Previously: print a warning and extract + run the binary anyway.
    path = _asset(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        sd._verify_sha256(path, None)
    text = str(exc.value)
    assert "cannot be verified" in text
    assert sd.ALLOW_UNVERIFIED_ENV in text


def test_an_unrecognised_algorithm_now_fails_instead_of_skipping(tmp_path):
    path = _asset(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        sd._verify_sha256(path, "md5:abc")
    assert sd.ALLOW_UNVERIFIED_ENV in str(exc.value)
    with pytest.raises(RuntimeError):
        sd._verify_sha256(path, "sha256:")  # algorithm named, digest missing


@pytest.mark.parametrize("digest", (None, "md5:abc"))
def test_the_unverified_hatch_restores_the_old_behaviour(tmp_path, monkeypatch, capsys, digest):
    monkeypatch.setenv(sd.ALLOW_UNVERIFIED_ENV, "1")
    sd._verify_sha256(_asset(tmp_path), digest)
    assert sd.ALLOW_UNVERIFIED_ENV in capsys.readouterr().out


def test_the_hatch_never_excuses_an_actual_mismatch(tmp_path, monkeypatch):
    # "I accept unverified bytes" is not "I accept bytes that failed the check".
    monkeypatch.setenv(sd.ALLOW_UNVERIFIED_ENV, "1")
    path = _asset(tmp_path)
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        sd._verify_sha256(path, "sha256:" + hashlib.sha256(b"other").hexdigest())


# ── A pinned tag that stopped resolving fails closed ──
def _stub_fetch(monkeypatch, serve):
    """``serve(repo, tag) -> release dict | None``; None means that repo has no
    such release. Records every (repo, tag) asked for."""
    asked: list[tuple[str, object]] = []

    def _fake(tag = None, *, repo = None, token = None, timeout = 30.0, allow_latest = True):
        repo = repo or sd.DEFAULT_REPO
        asked.append((repo, tag))
        release = serve(repo, tag)
        if release is not None:
            return release
        if tag and not allow_latest:
            return None
        raise urllib.error.HTTPError(f"https://api/{repo}", 404, "not found", None, None)

    monkeypatch.setattr(sd, "_fetch_release", _fake)
    return asked


def _release(tag: str, asset: str) -> dict:
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": asset,
                "browser_download_url": f"https://example.invalid/{asset}",
                "digest": "sha256:" + "c" * 64,
            }
        ],
    }


_LINUX_CPU = "sd-{tag}-bin-Linux-Ubuntu-22.04-x86_64.zip"


def test_a_vanished_pinned_tag_is_not_replaced_by_latest(monkeypatch):
    """The old behaviour: 404 on the pin -> install that repo's newest release,
    which is an unreviewed build wearing the pinned install's name."""
    latest = _release("master-999-newer", _LINUX_CPU.format(tag = "master-999-newer"))
    asked = _stub_fetch(monkeypatch, lambda repo, tag: None if tag else latest)
    repo, release, chosen = sd._resolve_with_fallback("cpu", None)
    assert release is None and chosen is None
    # Nothing ever asked for an unpinned latest on either repo.
    assert [t for _r, t in asked if t is None] == []


def test_the_vanished_pin_error_names_the_pin_and_the_override(monkeypatch, tmp_path):
    _stub_fetch(monkeypatch, lambda repo, tag: None)
    with pytest.raises(RuntimeError) as exc:
        sd.install(install_dir = tmp_path / "sd")
    text = str(exc.value)
    assert "No prebuilt sd-cli" in text  # the long-standing opening still holds
    assert sd.DEFAULT_TAG in text
    assert "UNSLOTH_SD_CPP_TAG" in text
    assert sd.ALLOW_LATEST_ENV in text
    # Lazy install: this surfaces mid-generation, so it has to scope the damage.
    assert "rest of Studio is unchanged" in text


def test_allow_latest_restores_the_latest_fallback(monkeypatch):
    monkeypatch.setenv(sd.ALLOW_LATEST_ENV, "1")
    latest = _release("master-999-newer", _LINUX_CPU.format(tag = "master-999-newer"))
    asked = _stub_fetch(monkeypatch, lambda repo, tag: None if tag else latest)
    repo, release, chosen = sd._resolve_with_fallback("cpu", None)
    assert release is not None and chosen
    assert (sd.DEFAULT_REPO, None) in asked


def test_an_explicitly_overridden_tag_is_honoured(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "master-777-mine")
    mine = _release("master-777-mine", _LINUX_CPU.format(tag = "master-777-mine"))
    _stub_fetch(monkeypatch, lambda repo, tag: mine if tag == "master-777-mine" else None)
    repo, release, chosen = sd._resolve_with_fallback("cpu", None)
    assert release["tag_name"] == "master-777-mine"


def test_an_empty_tag_still_tracks_latest(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "")
    latest = _release("master-999-newer", _LINUX_CPU.format(tag = "master-999-newer"))
    asked = _stub_fetch(monkeypatch, lambda repo, tag: latest if tag is None else None)
    repo, release, chosen = sd._resolve_with_fallback("cpu", None)
    assert release is not None and chosen
    assert asked[0] == (sd.DEFAULT_REPO, None)


# ── What the fail-closed change must NOT break ──
def test_mirror_to_upstream_at_the_same_pinned_version_still_works(monkeypatch):
    """The surviving fallback is about HOST COVERAGE, not version drift: the mirror
    does not build Linux Vulkan at all, so resolution moves to upstream -- but it
    asks upstream for the release the pin was built from, so the version installed
    is still the reviewed one. Removing this would cost those hosts a native engine
    with no security benefit whatsoever."""
    upstream_tag = sd.upstream_tag_for(sd.DEFAULT_TAG)
    assert upstream_tag != sd.DEFAULT_TAG

    def _serve(repo, tag):
        if repo == sd.DEFAULT_REPO and tag == sd.DEFAULT_TAG:
            # The mirror has the pin but builds no Vulkan asset for it.
            return _release(sd.DEFAULT_TAG, _LINUX_CPU.format(tag = sd.DEFAULT_TAG))
        if repo == sd.UPSTREAM_FALLBACK_REPO and tag == upstream_tag:
            return _release(upstream_tag, f"sd-{upstream_tag}-bin-Linux-x86_64-vulkan.zip")
        return None

    asked = _stub_fetch(monkeypatch, _serve)
    repo, release, chosen = sd._resolve_with_fallback("vulkan", None)
    assert repo == sd.UPSTREAM_FALLBACK_REPO
    assert release["tag_name"] == upstream_tag
    assert chosen.endswith("-vulkan.zip")
    # Still never the mirror-only string upstream cannot have, and still no latest.
    assert (sd.UPSTREAM_FALLBACK_REPO, sd.DEFAULT_TAG) not in asked
    assert [t for _r, t in asked if t is None] == []


def test_a_pinned_release_that_serves_this_host_installs_unchanged(monkeypatch, tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("build/bin/sd-cli", b"#!/bin/sh\necho sd-cli\n")
    payload = buf.getvalue()
    asset = _LINUX_CPU.format(tag = sd.DEFAULT_TAG)
    release = {
        "tag_name": sd.DEFAULT_TAG,
        "assets": [
            {
                "name": asset,
                "browser_download_url": f"https://example.invalid/{asset}",
                "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    _stub_fetch(monkeypatch, lambda repo, tag: release if repo == sd.DEFAULT_REPO else None)
    monkeypatch.setattr(sd, "_download", lambda url, dest, **kw: dest.write_bytes(payload))
    cli = sd.install(install_dir = tmp_path / "sd")
    assert cli.name == "sd-cli" and cli.is_file()


def test_an_install_whose_asset_publishes_no_digest_is_refused(monkeypatch, tmp_path):
    asset = _LINUX_CPU.format(tag = sd.DEFAULT_TAG)
    release = {
        "tag_name": sd.DEFAULT_TAG,
        "assets": [{"name": asset, "browser_download_url": f"https://example.invalid/{asset}"}],
    }
    _stub_fetch(monkeypatch, lambda repo, tag: release if repo == sd.DEFAULT_REPO else None)
    monkeypatch.setattr(sd, "_download", lambda url, dest, **kw: dest.write_bytes(b"whatever"))
    with pytest.raises(RuntimeError, match = "cannot be verified"):
        sd.install(install_dir = tmp_path / "sd")
