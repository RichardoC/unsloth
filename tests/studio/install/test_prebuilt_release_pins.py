# SPDX-License-Identifier: AGPL-3.0-only
# Determinism-anchor tests for studio/prebuilt_release_pins.json and the two
# installers that read it. No network: nothing here resolves a real release.
#
# The property under test: the same installer installs the same release on any
# day. Before the pin both installers defaulted to "newest published release",
# resolved at install time, so two runs of one .dmg could disagree.

import json
import os
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
PINS_PATH = STUDIO_DIR / "prebuilt_release_pins.json"

if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))


# Plain imports (studio/ is on sys.path) rather than spec-loading: the installers
# resolve `prebuilt_core` through sys.modules, so a monkeypatch on `core` here has
# to land on the very object they hold.
import prebuilt_core as core  # noqa: E402
import install_llama_prebuilt as llama  # noqa: E402
import install_whisper_prebuilt as whisper  # noqa: E402

PrebuiltFallback = core.PrebuiltFallback

# (installer module, component key, env override name)
INSTALLERS = (
    (llama, "llama_cpp", "UNSLOTH_LLAMA_RELEASE_TAG"),
    (whisper, "whisper_cpp", "UNSLOTH_WHISPER_RELEASE_TAG"),
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    """Every test starts from "no overrides set" so a developer's shell cannot
    make a pinned-default assertion pass or fail by accident."""
    for name in ("UNSLOTH_LLAMA_RELEASE_TAG", "UNSLOTH_WHISPER_RELEASE_TAG", core.ALLOW_LATEST_ENV):
        monkeypatch.delenv(name, raising = False)


# ── The pins file itself ──
def test_pins_file_parses_and_has_the_expected_schema():
    data = json.loads(PINS_PATH.read_text(encoding = "utf-8"))
    assert data["schema_version"] == core.RELEASE_PINS_SCHEMA_VERSION
    assert isinstance(data.get("comment"), str) and data["comment"].strip()
    assert isinstance(data.get("pinned_at_utc"), str) and data["pinned_at_utc"].strip()
    assert set(data["components"]) == {"llama_cpp", "whisper_cpp"}


@pytest.mark.parametrize(
    ("component", "repo"),
    (("llama_cpp", "unslothai/llama.cpp"), ("whisper_cpp", "unslothai/whisper.cpp")),
)
def test_every_pinned_tag_is_a_non_empty_string(component, repo):
    entry = core.pinned_release(component)
    assert entry["repo"] == repo
    assert isinstance(entry["release_tag"], str) and entry["release_tag"].strip()
    # Recorded for human verification only (see the file's comment); still must
    # look like a digest if present, so a typo cannot masquerade as one.
    digest = entry.get("checksum_index_sha256")
    if digest is not None:
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_whisper_pin_stays_paired_with_the_llama_pin():
    # A whisper slim bundle needs the ggml runtime from its paired llama release;
    # bumping one pin without the other installs a mismatched pair.
    whisper_entry = core.pinned_release("whisper_cpp")
    assert whisper_entry["paired_llama_tag"] == core.pinned_release("llama_cpp")["release_tag"]


def test_pins_file_ships_as_package_data():
    # A pins file that is not packaged silently does nothing in production.
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    assert f'"{core.RELEASE_PINS_FILENAME}"' in pyproject


# ── Installer defaults ──
@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_installer_defaults_to_the_pinned_tag(module, component, env_var):
    pinned = core.pinned_release(component)["release_tag"]
    assert module.default_published_release_tag() == pinned
    # ...and is never the pre-pin "resolve latest at install time" sentinel.
    assert module.default_published_release_tag() not in (None, "")


@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_env_override_wins_over_the_pin(module, component, env_var, monkeypatch):
    monkeypatch.setenv(env_var, "some-other-release-tag")
    assert module.default_published_release_tag() == "some-other-release-tag"


@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_allow_latest_optout_restores_latest_resolution(module, component, env_var, monkeypatch):
    monkeypatch.setenv(core.ALLOW_LATEST_ENV, "1")
    assert module.default_published_release_tag() == ""


@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_optout_still_loses_to_an_explicit_env_tag(module, component, env_var, monkeypatch):
    monkeypatch.setenv(core.ALLOW_LATEST_ENV, "1")
    monkeypatch.setenv(env_var, "explicit-tag")
    assert module.default_published_release_tag() == "explicit-tag"


@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_pin_does_not_apply_to_another_publisher(module, component, env_var):
    # The pinned tag exists only in the repo the pin names; forwarding it to
    # another publisher would request a tag that repo never released.
    assert module.default_published_release_tag("ggml-org/llama.cpp") == ""
    own_repo = core.pinned_release(component)["repo"]
    assert module.default_published_release_tag(own_repo) != ""


# DEFAULT_PUBLISHED_TAG is evaluated at import, before the env-cleaning fixture
# runs, so these two only mean anything in an unpolluted shell.
@pytest.mark.skipif(
    any(os.environ.get(n) for n in ("UNSLOTH_LLAMA_RELEASE_TAG", core.ALLOW_LATEST_ENV)),
    reason = "an override was set before import, so the module constant is not the pin",
)
def test_llama_module_constant_carries_the_pin():
    assert llama.DEFAULT_PUBLISHED_TAG == core.pinned_release("llama_cpp")["release_tag"]


@pytest.mark.skipif(
    any(os.environ.get(n) for n in ("UNSLOTH_WHISPER_RELEASE_TAG", core.ALLOW_LATEST_ENV)),
    reason = "an override was set before import, so the module constant is not the pin",
)
def test_whisper_module_constant_carries_the_pin():
    assert whisper.DEFAULT_PUBLISHED_TAG == core.pinned_release("whisper_cpp")["release_tag"]


# ── CLI wiring ──
def test_whisper_cli_default_is_the_pinned_tag():
    args = whisper.build_arg_parser().parse_args(["--install-dir", "/tmp/whisper"])
    assert args.published_release_tag is None  # resolved in main(), scoped to --published-repo
    resolved = whisper.default_published_release_tag(args.published_repo)
    assert resolved == core.pinned_release("whisper_cpp")["release_tag"]


def test_whisper_cli_explicit_tag_survives_the_default_resolution():
    args = whisper.build_arg_parser().parse_args(
        ["--install-dir", "/tmp/whisper", "--published-release-tag", "v0.0.0-explicit"]
    )
    assert args.published_release_tag == "v0.0.0-explicit"


def test_llama_cli_default_is_the_pinned_tag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install_llama_prebuilt.py"])
    args = llama.parse_args()
    assert args.published_release_tag == core.pinned_release("llama_cpp")["release_tag"]


def test_llama_cli_explicit_tag_wins(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["install_llama_prebuilt.py", "--published-release-tag", "b0-explicit"]
    )
    assert llama.parse_args().published_release_tag == "b0-explicit"


def test_llama_cli_drops_the_pin_for_another_publisher(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["install_llama_prebuilt.py", "--published-repo", "ggml-org/llama.cpp"]
    )
    assert llama.parse_args().published_release_tag == ""


def test_whisper_main_resolves_the_pin_for_the_probe(monkeypatch):
    seen = {}

    def _fake_resolve(host, **kwargs):
        seen.update(kwargs)
        return {"prebuilt_available": False, "repo": kwargs["published_repo"]}

    monkeypatch.setattr(whisper, "resolve_prebuilt", _fake_resolve)
    assert whisper.main(["--resolve-prebuilt"]) == whisper.EXIT_SUCCESS
    assert seen["published_release_tag"] == core.pinned_release("whisper_cpp")["release_tag"]


def test_whisper_main_drops_the_pin_for_another_publisher(monkeypatch):
    seen = {}

    def _fake_resolve(host, **kwargs):
        seen.update(kwargs)
        return {"prebuilt_available": False, "repo": kwargs["published_repo"]}

    monkeypatch.setattr(whisper, "resolve_prebuilt", _fake_resolve)
    assert whisper.main(["--resolve-prebuilt", "--published-repo", "someone/else"]) == 0
    assert seen["published_release_tag"] is None


# ── Fail-closed loading ──
def test_missing_pins_file_raises_instead_of_tracking_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "release_pins_path", lambda: tmp_path / "does_not_exist.json")
    with pytest.raises(PrebuiltFallback):
        core.load_release_pins()
    for module, _component, _env in INSTALLERS:
        with pytest.raises(PrebuiltFallback):
            module.default_published_release_tag()


def test_malformed_pins_file_raises_instead_of_tracking_latest(tmp_path, monkeypatch):
    bad = tmp_path / "prebuilt_release_pins.json"
    bad.write_text("{not json", encoding = "utf-8")
    monkeypatch.setattr(core, "release_pins_path", lambda: bad)
    with pytest.raises(PrebuiltFallback):
        core.load_release_pins()
    for module, _component, _env in INSTALLERS:
        with pytest.raises(PrebuiltFallback):
            module.default_published_release_tag()


def test_wrong_schema_version_raises(tmp_path, monkeypatch):
    bad = tmp_path / "prebuilt_release_pins.json"
    bad.write_text(json.dumps({"schema_version": 999, "components": {}}), encoding = "utf-8")
    monkeypatch.setattr(core, "release_pins_path", lambda: bad)
    with pytest.raises(PrebuiltFallback):
        core.load_release_pins()


def test_empty_components_raises(tmp_path, monkeypatch):
    bad = tmp_path / "prebuilt_release_pins.json"
    bad.write_text(
        json.dumps({"schema_version": core.RELEASE_PINS_SCHEMA_VERSION, "components": {}}),
        encoding = "utf-8",
    )
    monkeypatch.setattr(core, "release_pins_path", lambda: bad)
    with pytest.raises(PrebuiltFallback):
        core.load_release_pins()


def test_blank_release_tag_raises(tmp_path, monkeypatch):
    bad = tmp_path / "prebuilt_release_pins.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": core.RELEASE_PINS_SCHEMA_VERSION,
                "components": {"llama_cpp": {"repo": "unslothai/llama.cpp", "release_tag": "  "}},
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(core, "release_pins_path", lambda: bad)
    with pytest.raises(PrebuiltFallback):
        core.pinned_release("llama_cpp")


def test_unknown_component_raises(tmp_path, monkeypatch):
    with pytest.raises(PrebuiltFallback):
        core.pinned_release("no_such_component")


# ── The pin is never silently walked past ──
def test_pinned_llama_release_yields_exactly_one_candidate(monkeypatch):
    """The older-release walk exists so an incompatible newest release cannot
    brick the install. It must not fire for a pinned tag: walking would install a
    different version than the one pinned, which is the drift this change removes.
    An incompatible pin has to surface as an error instead."""
    calls = []

    class _Bundle:
        release_tag = "pinned-tag"
        upstream_tag = "b99999"

    def _fake_pinned_bundle(repo, tag):
        calls.append((repo, tag))
        return _Bundle()

    monkeypatch.setattr(llama, "_download_host_resolve_enabled", lambda: False)
    monkeypatch.setattr(llama, "pinned_published_release_bundle", _fake_pinned_bundle)
    monkeypatch.setattr(llama, "validated_checksums_for_bundle", lambda repo, bundle: {})
    monkeypatch.setattr(
        llama,
        "iter_published_release_bundles",
        lambda *a, **k: pytest.fail("a pinned release must not scan other releases"),
    )
    resolved = list(
        llama.iter_resolved_published_releases("latest", llama.DEFAULT_PUBLISHED_REPO, "pinned-tag")
    )
    assert len(resolved) == 1
    assert resolved[0].bundle.release_tag == "pinned-tag"
    assert calls == [(llama.DEFAULT_PUBLISHED_REPO, "pinned-tag")]


@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_incompatible_pin_error_names_the_pin_and_the_optout(module, component, env_var):
    pinned = core.pinned_release(component)["release_tag"]
    hinted = module.pin_incompatibility_hint(
        PrebuiltFallback("no compatible prebuilt asset was found"), pinned
    )
    text = str(hinted)
    assert "no compatible prebuilt asset was found" in text
    assert pinned in text
    assert core.RELEASE_PINS_FILENAME in text
    assert core.ALLOW_LATEST_ENV in text
    assert env_var in text


@pytest.mark.parametrize(("module", "component", "env_var"), INSTALLERS)
def test_hint_left_alone_for_a_tag_that_is_not_the_pin(module, component, env_var):
    original = PrebuiltFallback("no compatible prebuilt asset was found")
    assert module.pin_incompatibility_hint(original, "some-users-own-tag") is original
    assert module.pin_incompatibility_hint(original, "") is original


def test_whisper_hint_preserves_the_exit_code_2_error_type():
    pinned = core.pinned_release("whisper_cpp")["release_tag"]
    hinted = whisper.pin_incompatibility_hint(
        whisper.ReleaseCompatibilityError("paired llama runtime missing"), pinned
    )
    assert isinstance(hinted, whisper.ReleaseCompatibilityError)


def test_pinned_llama_release_prefers_the_download_host(monkeypatch):
    # Pinning made this the default path; it must not start charging every
    # install an api.github.com call (60/hour unauthenticated).
    seen = []

    class _Bundle:
        release_tag = "pinned-tag"
        upstream_tag = "b99999"

    sentinel = llama.ResolvedPublishedRelease(bundle = _Bundle(), checksums = {})
    monkeypatch.setattr(llama, "_download_host_resolve_enabled", lambda: True)
    monkeypatch.setattr(
        llama,
        "_download_host_resolved_release",
        lambda repo, tag = None: (seen.append((repo, tag)), sentinel)[1],
    )
    monkeypatch.setattr(
        llama,
        "pinned_published_release_bundle",
        lambda *a, **k: pytest.fail("must not touch api.github.com when the download host answers"),
    )
    resolved = list(
        llama.iter_resolved_published_releases("latest", llama.DEFAULT_PUBLISHED_REPO, "pinned-tag")
    )
    assert resolved == [sentinel]
    assert seen == [(llama.DEFAULT_PUBLISHED_REPO, "pinned-tag")]
