# SPDX-License-Identifier: AGPL-3.0-only
# Trust-anchor tests for the checksum-index digest pinned in
# studio/prebuilt_release_pins.json. No network: every byte here is synthetic.
#
# The property under test: an archive's sha256 comes from the release's checksum
# index, and that index is only believed when its raw bytes match a sha256 that
# lives in reviewed, committed code. Verifying an archive against an index served
# by the same release over the same TLS channel proves only that the release is
# self-consistent; whoever can serve one can serve the other. The in-tree digest
# is the one link the release host cannot forge.

import hashlib
import json
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
PINS_PATH = STUDIO_DIR / "prebuilt_release_pins.json"

if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import prebuilt_core as core  # noqa: E402
import install_llama_prebuilt as llama  # noqa: E402
import install_whisper_prebuilt as whisper  # noqa: E402

PrebuiltFallback = core.PrebuiltFallback

PINNED_LLAMA = core.pinned_release("llama_cpp")
PINNED_WHISPER = core.pinned_release("whisper_cpp")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    """Neither escape hatch may be inherited from the developer's shell: both of
    them switch enforcement off, so a leaked one would make every assertion here
    pass for the wrong reason."""
    for name in (
        "UNSLOTH_LLAMA_RELEASE_TAG",
        "UNSLOTH_WHISPER_RELEASE_TAG",
        core.ALLOW_LATEST_ENV,
        core.ALLOW_UNVERIFIED_ENV,
    ):
        monkeypatch.delenv(name, raising = False)


# ── The recorded digests themselves ──
@pytest.mark.parametrize("component", ("llama_cpp", "whisper_cpp"))
def test_recorded_digest_is_wellformed_lowercase_hex(component):
    entry = json.loads(PINS_PATH.read_text(encoding = "utf-8"))["components"][component]
    digest = entry["checksum_index_sha256"]
    assert isinstance(digest, str)
    assert len(digest) == 64, f"{component}: sha256 must be 64 hex chars, got {len(digest)}"
    assert digest == digest.lower(), f"{component}: digest must be lowercase"
    assert all(c in "0123456789abcdef" for c in digest), f"{component}: not hex: {digest!r}"
    # And it must name the asset it is the digest OF, or a bump cannot be checked.
    assert entry["checksum_index_asset"].endswith(".json")


@pytest.mark.parametrize(
    ("entry", "asset"),
    (
        (PINNED_LLAMA, llama.DEFAULT_PUBLISHED_SHA256_ASSET),
        (PINNED_WHISPER, whisper.SHA256_ASSET_NAME),
    ),
)
def test_shipped_pin_is_discoverable_by_repo_and_tag(entry, asset):
    # Lookup is by (repo, release_tag) so any caller holding a resolved release can
    # ask "is this the release the tree pinned?" without knowing the component name.
    found = core.pinned_checksum_index_digest(
        entry["repo"], entry["release_tag"], checksum_index_asset = asset
    )
    assert found == entry["checksum_index_sha256"]
    assert entry["checksum_index_asset"] == asset


# ── Scoping: enforcement applies to the pinned release and nothing else ──
def test_no_pin_covers_another_tag_in_the_same_repo():
    assert (
        core.pinned_checksum_index_digest(PINNED_WHISPER["repo"], "v0.0.0-not-the-pin") is None
    )


def test_no_pin_covers_another_publisher():
    assert core.pinned_checksum_index_digest("someone/else", PINNED_WHISPER["release_tag"]) is None


def test_allow_latest_turns_the_digest_pin_off(monkeypatch):
    # ALLOW_LATEST means "I accept a different version", and today's latest may well
    # BE the pinned tag; enforcement must not survive the opt-out on that accident.
    monkeypatch.setenv(core.ALLOW_LATEST_ENV, "1")
    assert (
        core.pinned_checksum_index_digest(
            PINNED_WHISPER["repo"], PINNED_WHISPER["release_tag"]
        )
        is None
    )


def test_allow_unverified_turns_the_digest_pin_off(monkeypatch):
    monkeypatch.setenv(core.ALLOW_UNVERIFIED_ENV, "1")
    assert (
        core.pinned_checksum_index_digest(
            PINNED_LLAMA["repo"], PINNED_LLAMA["release_tag"]
        )
        is None
    )


def test_an_unreadable_pins_file_is_not_a_pin_rather_than_a_failure(tmp_path, monkeypatch):
    # The default install path already fails closed on a missing pins file long
    # before this point (pinned_release_tag), so treating it as "no pin" here
    # cannot downgrade a default install -- but raising would break an install
    # that legitimately overrode the tag and never needed the file.
    monkeypatch.setattr(core, "release_pins_path", lambda: tmp_path / "gone.json")
    assert (
        core.pinned_checksum_index_digest(
            PINNED_WHISPER["repo"], PINNED_WHISPER["release_tag"]
        )
        is None
    )


def test_a_pin_bump_that_forgot_the_digest_fails_closed(tmp_path, monkeypatch):
    bad = tmp_path / "prebuilt_release_pins.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": core.RELEASE_PINS_SCHEMA_VERSION,
                "components": {
                    "whisper_cpp": {
                        "repo": PINNED_WHISPER["repo"],
                        "release_tag": "v9.9.9-unsloth.1",
                        "checksum_index_asset": whisper.SHA256_ASSET_NAME,
                    }
                },
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(core, "release_pins_path", lambda: bad)
    with pytest.raises(PrebuiltFallback) as exc:
        core.pinned_checksum_index_digest(
            PINNED_WHISPER["repo"], "v9.9.9-unsloth.1",
            checksum_index_asset = whisper.SHA256_ASSET_NAME,
        )
    assert "checksum_index_sha256" in str(exc.value)


# ── Functional: real bytes through the real verification path ──
_PINNED_TAG = PINNED_WHISPER["release_tag"]
_PINNED_REPO = PINNED_WHISPER["repo"]
_ASSET = "whisper-v1.9.2-unsloth.11-linux-x64-cpu.tar.gz"
_ARCHIVE_SHA = "a" * 64


def _index_bytes(release_tag: str = _PINNED_TAG) -> bytes:
    """A checksum index shaped exactly like a published one."""
    return json.dumps(
        {
            "schema_version": whisper.SCHEMA_VERSION,
            "component": "whisper.cpp",
            "release_tag": release_tag,
            "upstream_tag": "v1.9.2",
            "artifacts": {_ASSET: {"sha256": _ARCHIVE_SHA}},
        }
    ).encode("utf-8")


def _pin_the_bytes(monkeypatch, tmp_path, raw: bytes, *, tag: str = _PINNED_TAG) -> None:
    """Re-point the pins file at a manifest recording sha256(raw) for repo@tag, so
    the real published digest is not needed to exercise the real code path."""
    pins = tmp_path / "prebuilt_release_pins.json"
    pins.write_text(
        json.dumps(
            {
                "schema_version": core.RELEASE_PINS_SCHEMA_VERSION,
                "components": {
                    "whisper_cpp": {
                        "repo": _PINNED_REPO,
                        "release_tag": tag,
                        "checksum_index_asset": whisper.SHA256_ASSET_NAME,
                        "checksum_index_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                },
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(core, "release_pins_path", lambda: pins)


def _serve(monkeypatch, raw: bytes) -> None:
    monkeypatch.setattr(whisper, "download_bytes", lambda url, **kwargs: raw)


def _bundle(repo: str = _PINNED_REPO, tag: str = _PINNED_TAG):
    return whisper.ReleaseBundle(
        repo = repo,
        release_tag = tag,
        manifest = {},
        asset_urls = {
            whisper.SHA256_ASSET_NAME: core.release_asset_download_url(
                repo, tag, whisper.SHA256_ASSET_NAME
            )
        },
    )


def test_the_pinned_index_is_accepted_and_its_checksums_are_used(monkeypatch, tmp_path):
    raw = _index_bytes()
    _pin_the_bytes(monkeypatch, tmp_path, raw)
    _serve(monkeypatch, raw)
    assert whisper.fetch_release_checksums(_bundle()) == {_ASSET: _ARCHIVE_SHA}


def test_a_one_byte_change_to_the_index_fails_closed(monkeypatch, tmp_path):
    """The attack this closes: swap the index (and with it every archive sha256)
    for one the release host serves happily. Nothing downstream can notice --
    index, manifest and archives all agree with each other."""
    honest = _index_bytes()
    _pin_the_bytes(monkeypatch, tmp_path, honest)
    # Still valid JSON, still self-consistent, still the right schema/component/tag.
    tampered = honest.replace(b'"' + _ARCHIVE_SHA.encode() + b'"', b'"' + b"b" * 64 + b'"')
    assert tampered != honest and len(tampered) == len(honest)
    _serve(monkeypatch, tampered)
    with pytest.raises(PrebuiltFallback) as exc:
        whisper.fetch_release_checksums(_bundle())
    text = str(exc.value)
    assert core.RELEASE_PINS_FILENAME in text
    assert hashlib.sha256(honest).hexdigest() in text  # expected
    assert hashlib.sha256(tampered).hexdigest() in text  # actual
    assert whisper.SHA256_ASSET_NAME in text
    assert _PINNED_TAG in text and _PINNED_REPO in text
    assert core.ALLOW_UNVERIFIED_ENV in text and core.ALLOW_LATEST_ENV in text


def test_whitespace_only_reserialization_is_still_a_mismatch(monkeypatch, tmp_path):
    # The digest is over BYTES, not over the parsed object: a re-serialized index
    # with identical content is not the reviewed file and must not pass.
    honest = _index_bytes()
    _pin_the_bytes(monkeypatch, tmp_path, honest)
    _serve(monkeypatch, honest + b"\n")
    with pytest.raises(PrebuiltFallback):
        whisper.fetch_release_checksums(_bundle())


def test_a_tampered_index_is_accepted_for_an_env_overridden_tag(monkeypatch, tmp_path):
    # The installer was pointed at a release the tree never reviewed, so there is
    # no in-tree digest for it. Enforcing the pinned release's digest here would
    # break every UNSLOTH_WHISPER_RELEASE_TAG override.
    _pin_the_bytes(monkeypatch, tmp_path, _index_bytes())
    other_tag = "v1.9.2-unsloth.99"
    _serve(monkeypatch, _index_bytes(release_tag = other_tag))
    assert whisper.fetch_release_checksums(_bundle(tag = other_tag)) == {_ASSET: _ARCHIVE_SHA}


def test_a_tampered_index_is_accepted_under_allow_latest(monkeypatch, tmp_path):
    _pin_the_bytes(monkeypatch, tmp_path, _index_bytes())
    monkeypatch.setenv(core.ALLOW_LATEST_ENV, "1")
    _serve(monkeypatch, _index_bytes() + b"   ")
    assert whisper.fetch_release_checksums(_bundle()) == {_ASSET: _ARCHIVE_SHA}


def test_a_tampered_index_is_accepted_for_a_non_pinned_publisher(monkeypatch, tmp_path):
    # --published-repo someone/else: the pinned tag exists only in the pinned repo,
    # so a same-named tag elsewhere is a different release with no in-tree digest.
    _pin_the_bytes(monkeypatch, tmp_path, _index_bytes())
    _serve(monkeypatch, _index_bytes() + b"   ")
    assert whisper.fetch_release_checksums(_bundle(repo = "someone/else")) == {
        _ASSET: _ARCHIVE_SHA
    }


def test_verification_happens_before_the_index_is_parsed(monkeypatch, tmp_path):
    # Parsing is where schema/component/release_tag are cross-checked -- all of it
    # self-reported by the same file. A tampered index must be rejected on its
    # bytes, without its own claims ever being consulted.
    _pin_the_bytes(monkeypatch, tmp_path, _index_bytes())
    _serve(monkeypatch, b"not even json")
    monkeypatch.setattr(
        whisper,
        "parse_release_checksums",
        lambda *a, **k: pytest.fail("a tampered index was parsed before being verified"),
    )
    with pytest.raises(PrebuiltFallback, match = "does not match the digest pinned"):
        whisper.fetch_release_checksums(_bundle())


# ── The fast path is not a way around the pin ──
def test_the_download_host_path_reads_raw_bytes_only_when_a_pin_covers_the_release(
    monkeypatch, tmp_path
):
    """Both installers prefer a download-host fast path over the GitHub API, so
    enforcing only on the API path would leave the default route unchecked. The
    raw read happens exactly when there is an in-tree digest to compare against;
    otherwise the caller's ordinary JSON seam is used and nothing changes."""
    raw = _index_bytes()
    _pin_the_bytes(monkeypatch, tmp_path, raw)
    used: list[str] = []

    def _load(repo, tag):
        return core.load_verified_checksum_index(
            repo,
            tag,
            checksum_index_asset = whisper.SHA256_ASSET_NAME,
            fetch_json = lambda: (used.append("json"), json.loads(raw.decode()))[1],
            fetch_bytes = lambda: (used.append("bytes"), raw)[1],
        )

    assert _load(_PINNED_REPO, _PINNED_TAG)["release_tag"] == _PINNED_TAG
    assert used == ["bytes"]
    used.clear()
    assert _load("someone/else", _PINNED_TAG)["release_tag"] == _PINNED_TAG
    assert used == ["json"]


def test_the_download_host_path_rejects_a_tampered_index(monkeypatch, tmp_path):
    raw = _index_bytes()
    _pin_the_bytes(monkeypatch, tmp_path, raw)
    with pytest.raises(PrebuiltFallback, match = core.RELEASE_PINS_FILENAME):
        core.load_verified_checksum_index(
            _PINNED_REPO,
            _PINNED_TAG,
            checksum_index_asset = whisper.SHA256_ASSET_NAME,
            fetch_json = lambda: pytest.fail("the pinned path must read raw bytes"),
            fetch_bytes = lambda: raw + b"!",
        )
