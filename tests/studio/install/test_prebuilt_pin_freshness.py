# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The in-app freshness/update checks must respect studio/prebuilt_release_pins.json.

The pin makes each installer install one exact release, so a correct install is
deliberately OLDER than GitHub's newest published release. The freshness checks
resolved "latest" straight from GitHub, so every correctly pinned install showed
a permanent "update available" banner that no update could ever clear -- the
apply half would reinstall the pinned build and the banner would come straight
back.

What is pinned here:
  - with the pin in force, "latest" is the pinned tag and NO GitHub call is made;
  - UNSLOTH_PREBUILT_ALLOW_LATEST=1 restores the pre-pin behaviour exactly;
  - UNSLOTH_{LLAMA,WHISPER}_RELEASE_TAG compares against the override, since that
    is what the installer would install;
  - a publisher the pin does not name (a custom --published-repo) is unchanged;
  - an explicit Update click installs the pinned release and never past it.

No network: every GitHub fetcher is stubbed, and the pinned cases additionally
assert the fetcher was never called at all.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
BACKEND_DIR = STUDIO_DIR / "backend"
PINS_PATH = STUDIO_DIR / "prebuilt_release_pins.json"

# The backend modules import their siblings by top-level name (their sys.path
# root is studio/backend), exactly like test_managed_node_runtime.py does.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# structlog is a studio.txt requirement, so it is normally the real package here.
# Stub only a genuinely absent one -- a bare setdefault would shadow the real
# thing for every later test in the session.
if sys.modules.get("structlog") is None and importlib.util.find_spec("structlog") is None:
    _stub = sys.modules.setdefault("structlog", types.ModuleType("structlog"))
    _stub.get_logger = lambda *a, **k: logging.getLogger("structlog")

import prebuilt_core as core  # noqa: E402
import install_llama_prebuilt as llama_installer  # noqa: E402
import install_whisper_prebuilt as whisper_installer  # noqa: E402

lfresh = importlib.import_module("utils.llama_cpp_freshness")
lupd = importlib.import_module("utils.llama_cpp_update")
wfresh = importlib.import_module("utils.whisper_cpp_freshness")
wupd = importlib.import_module("utils.whisper_cpp_update")
release_pin = importlib.import_module("utils.prebuilt.release_pin")


PINS = json.loads(PINS_PATH.read_text(encoding = "utf-8"))
LLAMA_PIN = PINS["components"]["llama_cpp"]["release_tag"]
LLAMA_REPO = PINS["components"]["llama_cpp"]["repo"]
WHISPER_PIN = PINS["components"]["whisper_cpp"]["release_tag"]
WHISPER_REPO = PINS["components"]["whisper_cpp"]["repo"]

ALLOW_LATEST = release_pin.ALLOW_LATEST_ENV
LLAMA_TAG_ENV = lfresh.RELEASE_TAG_ENV
WHISPER_TAG_ENV = wfresh.RELEASE_TAG_ENV

# Tags that are unambiguously newer than the pins under each component's own
# comparator (llama: base build number; whisper: version + unsloth serial), so a
# banner appearing is never an artifact of a tie-break.
NEWER_THAN_LLAMA_PIN = "b99999-mix-deadbee"
NEWER_THAN_WHISPER_PIN = "v9.9.9-unsloth.99"
OLDER_THAN_LLAMA_PIN = "b9000"
OLDER_THAN_WHISPER_PIN = "v1.9.0-unsloth.1"


@pytest.fixture(autouse = True)
def _isolate(monkeypatch, tmp_path):
    """No inherited env, no shared cache, no real cache directory, no network.

    These modules memoize aggressively (24h in memory and on disk), so a warm
    cache from another test -- or from a developer's real ~/.unsloth -- could
    otherwise decide the verdict instead of the code under test.
    """
    for name in (ALLOW_LATEST, LLAMA_TAG_ENV, WHISPER_TAG_ENV):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(lfresh, "_cache_dir", lambda: tmp_path / ".llama_cache")
    monkeypatch.setattr(wfresh, "_cache_dir", lambda: tmp_path / ".whisper_cache")
    release_pin.reset_cache()
    lfresh.reset_caches()
    wfresh.reset_caches()
    lupd._resolve_memo.clear()
    wupd._resolve_memo.clear()
    yield
    release_pin.reset_cache()
    lfresh.reset_caches()
    wfresh.reset_caches()


class _Fetcher:
    """Stub GitHub fetcher that records whether it was consulted."""

    def __init__(self, tag):
        self.tag = tag
        self.calls = 0

    def __call__(self, repo, timeout = 5.0):
        self.calls += 1
        return self.tag


def _stub_github(monkeypatch, module, tag) -> _Fetcher:
    fetcher = _Fetcher(tag)
    monkeypatch.setattr(module, "_fetch_latest_release_tag", fetcher)
    return fetcher


def _installed_at(days_ago: int = 30) -> str:
    return (
        (datetime.now(tz = timezone.utc) - timedelta(days = days_ago))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _llama_install(install_dir: Path, tag: str, *, repo: str = LLAMA_REPO) -> str:
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("stub\n")
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(
            {
                "requested_tag": "latest",
                "tag": tag.split("-mix-")[0],
                "release_tag": tag,
                "published_repo": repo,
                "asset": f"app-{tag}-linux-x64-cuda13-newer.tar.gz",
                "source": "published",
                "installed_at_utc": _installed_at(),
            }
        )
    )
    return str(binary)


def _whisper_install(install_dir: Path, tag: str, *, repo: str = WHISPER_REPO) -> str:
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "whisper-server"
    binary.write_text("stub\n")
    (install_dir / "UNSLOTH_WHISPER_PREBUILT_INFO.json").write_text(
        json.dumps(
            {
                "requested_tag": "latest",
                "release_tag": tag,
                "published_repo": repo,
                "asset": f"whisper-{tag}-linux-x64-cpu.tar.gz",
                "source": "published",
                "installed_at_utc": _installed_at(),
            }
        )
    )
    return str(binary)


# ── The reader itself ──


def test_pins_path_resolves_the_in_tree_manifest():
    assert release_pin.pins_path() == PINS_PATH


def test_reader_constants_match_prebuilt_cores():
    # Two readers of one file (the backend one fails open, the installer one
    # fails closed -- see release_pin's module docstring). They must at least
    # agree on which file, which schema, and which opt-out.
    assert release_pin.PINS_FILENAME == core.RELEASE_PINS_FILENAME
    assert release_pin.PINS_SCHEMA_VERSION == core.RELEASE_PINS_SCHEMA_VERSION
    assert release_pin.ALLOW_LATEST_ENV == core.ALLOW_LATEST_ENV


@pytest.mark.parametrize(
    ("installer", "component", "env_var", "repo"),
    (
        (llama_installer, "llama_cpp", LLAMA_TAG_ENV, LLAMA_REPO),
        (whisper_installer, "whisper_cpp", WHISPER_TAG_ENV, WHISPER_REPO),
    ),
)
@pytest.mark.parametrize("mode", ("default", "allow_latest", "override", "foreign_repo"))
def test_target_tag_is_exactly_what_the_installer_would_install(
    monkeypatch, installer, component, env_var, repo, mode
):
    """The whole point: detection and apply must name the same release.

    Compared against the installer's own default resolver rather than a literal,
    so a pin bump or a precedence change cannot leave the banner behind.
    """
    published_repo = repo
    if mode == "allow_latest":
        monkeypatch.setenv(ALLOW_LATEST, "1")
    elif mode == "override":
        monkeypatch.setenv(env_var, "v0.0.0-explicit")
    elif mode == "foreign_repo":
        published_repo = "someone-else/fork"
    release_pin.reset_cache()

    ours = release_pin.install_target_tag(
        component, env_var = env_var, published_repo = published_repo
    )
    theirs = installer.default_published_release_tag(published_repo)
    # The installer spells "resolve GitHub's newest" as "", the backend as None:
    # a status payload carries None, and "" would read as a real tag.
    assert (ours or "") == theirs


def test_a_foreign_publisher_has_no_pin():
    # The pinned tag exists only in the pinned repo, so applying it to a custom
    # --published-repo would compare against a tag that repo never published.
    assert release_pin.pinned_tag("llama_cpp", published_repo = "ggml-org/llama.cpp") is None
    assert release_pin.pinned_tag("llama_cpp", published_repo = LLAMA_REPO) == LLAMA_PIN


def test_an_unreadable_manifest_fails_open(monkeypatch, tmp_path):
    # Opposite policy to the installer, which must fail closed: a corrupt pins
    # file may cost the banner its pin awareness, never take out the status route.
    broken = tmp_path / release_pin.PINS_FILENAME
    broken.write_text("{ not json", encoding = "utf-8")
    monkeypatch.setattr(release_pin, "pins_path", lambda: broken)
    release_pin.reset_cache()
    assert release_pin.load_pins() is None
    assert release_pin.pinned_tag("llama_cpp", published_repo = LLAMA_REPO) is None


# ── llama.cpp freshness ──


def test_llama_pinned_install_is_up_to_date_and_never_asks_github(monkeypatch, tmp_path):
    binary = _llama_install(tmp_path / "llama.cpp", LLAMA_PIN)
    github = _stub_github(monkeypatch, lfresh, NEWER_THAN_LLAMA_PIN)

    info = lfresh.check_prebuilt_freshness(binary)

    assert info["installed_tag"] == LLAMA_PIN.split("-mix-")[0]
    assert info["latest_tag"] == LLAMA_PIN
    assert info["behind"] is False
    assert info["stale"] is False
    # Not merely "the answer was right": under a pin the newest published release
    # is irrelevant, so the check must be correct with no network at all.
    assert github.calls == 0


def test_llama_install_older_than_the_pin_is_offered_the_pin(monkeypatch, tmp_path):
    binary = _llama_install(tmp_path / "llama.cpp", OLDER_THAN_LLAMA_PIN)
    github = _stub_github(monkeypatch, lfresh, NEWER_THAN_LLAMA_PIN)

    info = lfresh.check_prebuilt_freshness(binary)

    assert info["behind"] is True
    # The offer is the pin, not GitHub's newest -- that is what apply installs.
    assert info["latest_tag"] == LLAMA_PIN
    assert github.calls == 0


def test_llama_allow_latest_restores_the_github_comparison(monkeypatch, tmp_path):
    monkeypatch.setenv(ALLOW_LATEST, "1")
    binary = _llama_install(tmp_path / "llama.cpp", LLAMA_PIN)
    github = _stub_github(monkeypatch, lfresh, NEWER_THAN_LLAMA_PIN)

    info = lfresh.check_prebuilt_freshness(binary)

    assert info["latest_tag"] == NEWER_THAN_LLAMA_PIN
    assert info["behind"] is True
    assert github.calls == 1


def test_llama_env_release_tag_override_wins_over_pin_and_github(monkeypatch, tmp_path):
    # An explicit override is what the installer would install, so it is what the
    # banner must compare against -- exactly the reason the pin is honoured.
    monkeypatch.setenv(LLAMA_TAG_ENV, "b12345-mix-abcdef0")
    binary = _llama_install(tmp_path / "llama.cpp", LLAMA_PIN)
    github = _stub_github(monkeypatch, lfresh, NEWER_THAN_LLAMA_PIN)

    info = lfresh.check_prebuilt_freshness(binary)

    assert info["latest_tag"] == "b12345-mix-abcdef0"
    assert info["behind"] is True
    assert github.calls == 0


def test_llama_custom_published_repo_is_unchanged(monkeypatch, tmp_path):
    binary = _llama_install(tmp_path / "llama.cpp", "b9500", repo = "ggml-org/llama.cpp")
    github = _stub_github(monkeypatch, lfresh, "b9600")

    info = lfresh.check_prebuilt_freshness(binary)

    assert info["published_repo"] == "ggml-org/llama.cpp"
    assert info["latest_tag"] == "b9600"
    assert info["behind"] is True
    assert github.calls == 1


# ── whisper.cpp freshness (the non-macOS path) ──


def test_whisper_pinned_install_is_up_to_date_and_never_asks_github(monkeypatch, tmp_path):
    binary = _whisper_install(tmp_path / "whisper.cpp", WHISPER_PIN)
    github = _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)

    info = wfresh.check_prebuilt_freshness(binary)

    assert info["latest_tag"] == WHISPER_PIN
    assert info["behind"] is False
    assert github.calls == 0


def test_whisper_allow_latest_restores_the_github_comparison(monkeypatch, tmp_path):
    monkeypatch.setenv(ALLOW_LATEST, "1")
    binary = _whisper_install(tmp_path / "whisper.cpp", WHISPER_PIN)
    github = _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)

    info = wfresh.check_prebuilt_freshness(binary)

    assert info["latest_tag"] == NEWER_THAN_WHISPER_PIN
    assert info["behind"] is True
    assert github.calls == 1


def test_whisper_env_release_tag_override_is_the_comparison(monkeypatch, tmp_path):
    monkeypatch.setenv(WHISPER_TAG_ENV, "v2.0.0-unsloth.3")
    binary = _whisper_install(tmp_path / "whisper.cpp", WHISPER_PIN)
    github = _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)

    info = wfresh.check_prebuilt_freshness(binary)

    assert info["latest_tag"] == "v2.0.0-unsloth.3"
    assert github.calls == 0


# ── the update-status surface (what actually draws the banner) ──


def test_llama_status_shows_no_banner_on_a_pinned_install(monkeypatch, tmp_path):
    binary = _llama_install(tmp_path / "llama.cpp", LLAMA_PIN)
    monkeypatch.setattr(lupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(lupd, "_whisper_chain_status", lambda **kwargs: None)
    github = _stub_github(monkeypatch, lfresh, NEWER_THAN_LLAMA_PIN)

    status = lupd.get_update_status()

    assert status["supported"] is True
    assert status["update_available"] is False
    assert status["latest_tag"] == LLAMA_PIN
    assert github.calls == 0


def test_llama_explicit_check_now_does_not_call_github_when_pinned(monkeypatch, tmp_path):
    # force_refresh exists to bypass the 24h cache. Under a pin there is nothing
    # to refresh, so "check now" must not spend a round trip on an unused answer.
    binary = _llama_install(tmp_path / "llama.cpp", LLAMA_PIN)
    monkeypatch.setattr(lupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(lupd, "_whisper_chain_status", lambda **kwargs: None)
    github = _stub_github(monkeypatch, lfresh, NEWER_THAN_LLAMA_PIN)

    status = lupd.get_update_status(force_refresh = True)

    assert status["update_available"] is False
    assert github.calls == 0


def test_whisper_status_shows_no_banner_on_a_pinned_install(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    binary = _whisper_install(tmp_path / "whisper.cpp", WHISPER_PIN)
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    github = _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)

    status = wupd.get_update_status(force_refresh = True)

    assert status["supported"] is True
    assert status["update_available"] is False
    assert status["latest_tag"] == WHISPER_PIN
    assert github.calls == 0


def test_whisper_macos_still_defers_to_the_host_aware_resolver(monkeypatch, tmp_path):
    """The macOS path was already correct and must stay that way.

    It re-asks install_whisper_prebuilt.py's own resolver, so the banner compares
    against what THIS host can install (the newest release can require a newer
    macOS). That resolver now answers with the pin, so the two agree -- and the
    resolver, not the freshness read, is still what decides.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    binary = _whisper_install(tmp_path / "whisper.cpp", WHISPER_PIN)
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)
    asked = {"n": 0}

    def _resolver(**kwargs):
        asked["n"] += 1
        return {"prebuilt_available": True, "release_tag": WHISPER_PIN}

    monkeypatch.setattr(wupd, "_resolve_prebuilt_for_host", _resolver)

    status = wupd.get_update_status()

    assert asked["n"] == 1
    assert status["update_available"] is False
    assert status["latest_tag"] == WHISPER_PIN


def test_whisper_macos_resolver_answer_still_wins(monkeypatch, tmp_path):
    # Same path, opposite verdict: when the resolver says this host can install a
    # newer release, that is what would be installed, so the banner appears.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv(ALLOW_LATEST, "1")
    binary = _whisper_install(tmp_path / "whisper.cpp", OLDER_THAN_WHISPER_PIN)
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)
    monkeypatch.setattr(
        wupd,
        "_resolve_prebuilt_for_host",
        lambda **kwargs: {"prebuilt_available": True, "release_tag": WHISPER_PIN},
    )

    status = wupd.get_update_status()

    assert status["update_available"] is True
    assert status["latest_tag"] == WHISPER_PIN


# ── the explicit Update click ──


def _plan(monkeypatch, tmp_path, *, installed: str, offered, backend_request = None) -> dict:
    """_plan_llama_phase with the status it reads stubbed, so the offered tag can
    be made to disagree with the pin the way a stale banner or a direct POST can."""
    binary = _llama_install(tmp_path / "llama.cpp", installed)
    monkeypatch.setattr(lupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(lupd, "_installer_script", lambda: tmp_path / "install.py")
    monkeypatch.setattr(
        lupd,
        "_llama_only_status",
        lambda **kwargs: {"update_available": True, "latest_tag": offered},
    )
    return lupd._plan_llama_phase(backend_request)


def test_update_click_installs_the_pin_not_githubs_newest(monkeypatch, tmp_path):
    # The chosen semantics: a user-initiated update moves to the pinned release
    # and never past it. Off-baseline updates stay possible, but only by asking
    # for them (UNSLOTH_PREBUILT_ALLOW_LATEST / UNSLOTH_LLAMA_RELEASE_TAG), so
    # the app never silently leaves the baseline it was built and tested against.
    monkeypatch.setattr(sys, "platform", "linux")
    plan = _plan(
        monkeypatch, tmp_path, installed = OLDER_THAN_LLAMA_PIN, offered = NEWER_THAN_LLAMA_PIN
    )
    assert plan["spec"]["pin_release_tag"] == LLAMA_PIN


def test_update_click_honours_the_offered_tag_when_unpinned(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(ALLOW_LATEST, "1")
    plan = _plan(
        monkeypatch, tmp_path, installed = OLDER_THAN_LLAMA_PIN, offered = NEWER_THAN_LLAMA_PIN
    )
    assert plan["spec"]["pin_release_tag"] == NEWER_THAN_LLAMA_PIN


def test_update_click_honours_an_explicit_release_tag_override(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(LLAMA_TAG_ENV, "b12345-mix-abcdef0")
    plan = _plan(
        monkeypatch, tmp_path, installed = OLDER_THAN_LLAMA_PIN, offered = NEWER_THAN_LLAMA_PIN
    )
    assert plan["spec"]["pin_release_tag"] == "b12345-mix-abcdef0"


def test_whisper_chained_phase_installs_the_pin(monkeypatch, tmp_path):
    # whisper applies only ever run as the second phase of the llama update, so
    # the same clamp has to hold there or the pair drifts apart.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(ALLOW_LATEST, "1")  # make the status offer GitHub's newest
    binary = _whisper_install(tmp_path / "whisper.cpp", OLDER_THAN_WHISPER_PIN)
    monkeypatch.setattr(wupd, "_find_binary", lambda: binary)
    monkeypatch.setattr(wupd, "_installer_script", lambda: tmp_path / "install.py")
    _stub_github(monkeypatch, wfresh, NEWER_THAN_WHISPER_PIN)

    offered = wupd.chained_phase_plan()
    assert offered["update_available"] is True
    assert offered["phase"]["pin_release_tag"] == NEWER_THAN_WHISPER_PIN

    # Same situation with the pin back in force: the phase installs the pin.
    monkeypatch.delenv(ALLOW_LATEST)
    release_pin.reset_cache()
    wfresh.reset_caches()
    wupd._resolve_memo.clear()
    pinned = wupd.chained_phase_plan()
    assert pinned["update_available"] is True
    assert pinned["phase"]["pin_release_tag"] == WHISPER_PIN


def test_backend_switch_still_reinstalls_the_markers_own_release(monkeypatch, tmp_path):
    # A switch replaces the backend at the SAME release (slim whisper bundles
    # require that exact one), so the clamp must not drag it to the pin.
    monkeypatch.setattr(sys, "platform", "linux")
    plan = _plan(
        monkeypatch,
        tmp_path,
        installed = OLDER_THAN_LLAMA_PIN,
        offered = NEWER_THAN_LLAMA_PIN,
        backend_request = "cpu",
    )
    assert plan["spec"]["pin_release_tag"] == OLDER_THAN_LLAMA_PIN
