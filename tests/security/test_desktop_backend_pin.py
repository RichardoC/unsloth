"""Checks that one desktop build can only ever install one backend version.

install.sh resolves setup.sh and every requirements/constraints file out of the
INSTALLED unsloth wheel, so whichever wheel a run resolves decides the whole pin
set. Left on a `>=` floor, the same .dmg installs a different Python stack every
month. The desktop app therefore stamps its backend version at build time and
hands it to install.sh, which turns every backend spec into `==`.

Source-level assertions: the .rs cannot be exercised from pytest, and the argv
install.sh produces is covered behaviourally in tests/sh/test_backend_version_pin.sh.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "install.rs"
VERSION_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "preflight" / "version.rs"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"


def _rust_code(path: Path) -> str:
    """Source with `//` comment lines dropped, so prose cannot satisfy a check."""
    return "\n".join(
        line for line in path.read_text(encoding = "utf-8").splitlines()
        if not line.lstrip().startswith("//")
    )


def test_install_rs_hands_the_stamped_version_to_the_installer():
    code = _rust_code(INSTALL_RS)
    assert 'option_env!("UNSLOTH_DESKTOP_BACKEND_VERSION")' in code, (
        "install.rs must read the build-time stamp to pin the backend"
    )
    assert 'cmd.env("UNSLOTH_BACKEND_VERSION", backend_version)' in code, (
        "install.rs must export UNSLOTH_BACKEND_VERSION so install.sh pins"
    )


def test_install_rs_leaves_an_unstamped_build_unpinned():
    """The floor helper is the wrong source here.

    preflight::version::expected_backend_version() is
    `option_env!(...).unwrap_or(MIN_DESKTOP_BACKEND_VERSION)`. That floor is right
    for staleness checks and wrong for pinning: an unstamped local or CI build
    would be pinned to a long-stale version instead of keeping today's
    track-the-newest behaviour. Only release builds carry a stamp.
    """
    code = _rust_code(INSTALL_RS)
    assert "expected_backend_version" not in code, (
        "install.rs must not pin through the MIN_DESKTOP_BACKEND_VERSION floor helper"
    )
    assert "MIN_DESKTOP_BACKEND_VERSION" not in code, (
        "install.rs must not fall back to the backend floor when unstamped"
    )
    # The helper itself keeps its floor semantics: other call sites depend on it.
    assert 'option_env!("UNSLOTH_DESKTOP_BACKEND_VERSION").unwrap_or(MIN_DESKTOP_BACKEND_VERSION)' \
        in VERSION_RS.read_text(encoding = "utf-8")


def test_release_workflow_still_stamps_the_backend_version():
    """The pin is inert unless the release build stamps the variable."""
    workflow = RELEASE_WORKFLOW.read_text(encoding = "utf-8")
    assert "UNSLOTH_DESKTOP_BACKEND_VERSION:" in workflow


def _install_sh() -> str:
    return INSTALL_SH.read_text(encoding = "utf-8")


def test_install_sh_reads_one_pin_input():
    script = _install_sh()
    assert '_BACKEND_VERSION="${UNSLOTH_BACKEND_VERSION:-}"' in script
    assert "--backend-version) _next_is_backend_version=true ;;" in script


def test_install_sh_defaults_to_todays_floors():
    """Unset, nothing changes: `curl | sh` users and CI still track the newest."""
    script = _install_sh()
    assert '_UNSLOTH_SPEC="$PACKAGE_NAME"' in script
    assert '_UNSLOTH_FLOOR_SPEC="unsloth>=2026.8.18"' in script
    assert '_UNSLOTH_ZOO_SPEC="unsloth-zoo>=2026.8.12"' in script


def test_install_sh_pins_exactly_when_asked():
    script = _install_sh()
    assert '_UNSLOTH_SPEC="$PACKAGE_NAME==$_BACKEND_VERSION"' in script
    assert '_UNSLOTH_FLOOR_SPEC="unsloth==$_BACKEND_VERSION"' in script


def test_package_name_stays_a_bare_package_name():
    """PACKAGE_NAME is a `case` match, a --package re-run hint and a `uv pip show`
    argument; a version specifier folded into it would break all three."""
    script = _install_sh()
    assert re.search(r'^PACKAGE_NAME="unsloth"$', script, re.MULTILINE)
    assert not re.search(r"^PACKAGE_NAME=.*(==|>=)", script, re.MULTILINE)


# The only `unsloth>=` literals allowed to survive: the two --local install lines.
# --local keeps floating on purpose, because the editable overlay installed right
# after is the version the developer asked for and an ==pin would only decide which
# wheel gets thrown away. Listed verbatim so a NEW floor anywhere fails this test.
ALLOWED_UNSLOTH_FLOOR_LINES = {
    '--upgrade-package unsloth "unsloth>=2026.8.18" "unsloth-zoo>=2026.8.12"',
    'run_install_cmd_retry "install unsloth (auto torch backend)" uv pip install'
    ' --python "$_VENV_PY" "unsloth-zoo>=2026.8.12" "unsloth>=2026.8.18" --torch-backend=auto',
}


def test_no_backend_install_site_hardcodes_the_unsloth_floor_outside_local():
    """Every backend install site outside --local resolves through the spec variables,
    so pinning cannot be half-applied and leave one path floating."""
    stray = []
    for number, line in enumerate(_install_sh().splitlines(), start = 1):
        stripped = line.strip()
        if "unsloth>=" not in stripped:
            continue
        if stripped.startswith("#") or "_UNSLOTH_FLOOR_SPEC=" in stripped:
            continue
        if stripped in ALLOWED_UNSLOTH_FLOOR_LINES:
            continue
        stray.append(f"{number}: {stripped}")
    assert not stray, "unsloth floor left on a non-local install site:\n" + "\n".join(stray)
