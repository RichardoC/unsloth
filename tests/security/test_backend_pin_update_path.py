"""The desktop backend pin must survive the UPDATE path, not just the install.

install.sh pins the FIRST install (see test_desktop_backend_pin.py) and then hands
install_python_stack.py SKIP_STUDIO_BASE=1, so the core-package branch there never
runs on a fresh install. `unsloth studio update` pops SKIP_STUDIO_BASE
(unsloth_cli/commands/studio.py), which is exactly the branch that reinstalls
unsloth + unsloth-zoo -- and it is what the desktop Update and Repair buttons run
(studio/src-tauri/src/update.rs). A pin that survives install but not update is not
a pin, so these tests drive the real branch and read back the argv it produces.

The value is also a command-argument injection surface: it is concatenated into a
requirement string that becomes an argv element of `uv pip install`, so a malformed
value must be refused rather than interpolated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips  # noqa: E402

UPDATE_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "update.rs"
CLI_STUDIO_PY = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"


# The exact argv the update branch produces today, unpinned. Spelled out rather
# than derived so a change to it has to be made here too, deliberately.
TODAYS_CORE_UPDATE_ARGS = (
    "--no-cache-dir",
    "--upgrade-package",
    "unsloth",
    "--upgrade-package",
    "unsloth-zoo",
    "unsloth",
    "unsloth-zoo",
)


class _StopAfterCorePackages(Exception):
    """Raised by the pip_install stub to end the run at the step under test."""


def _core_package_call(env: dict[str, str], *, no_torch: bool = False):
    """Run install_python_stack() far enough to capture the core-package install.

    Returns (label, args) of the first pip_install() the core-package phase makes.
    Everything before it is stubbed: the manifest writes touch the real venv root,
    and the pip/uv bootstrap shells out.
    """
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_pip_install(label: str, *args: str, **kwargs):
        calls.append((label, args))
        raise _StopAfterCorePackages

    full_env = {
        "SKIP_STUDIO_BASE": "0",
        "STUDIO_LOCAL_REPO": "",
        "STUDIO_PACKAGE_NAME": "unsloth",
        "UNSLOTH_BACKEND_VERSION": "",
        **env,
    }

    with (
        mock.patch.dict(os.environ, full_env, clear = False),
        mock.patch.object(ips, "pip_install", side_effect = fake_pip_install),
        mock.patch.object(ips, "_progress", return_value = None),
        mock.patch.object(ips, "_step", return_value = None),
        mock.patch.object(ips, "_bootstrap_uv", return_value = True),
        mock.patch.object(ips, "run", return_value = None),
        mock.patch.object(ips, "_bitsandbytes_installed", return_value = True),
        mock.patch.object(ips, "IS_MAC_ARM", False),
        mock.patch.object(ips, "NO_TORCH", no_torch),
        mock.patch.object(ips.install_manifest, "remove_manifest", return_value = True),
        mock.patch.object(ips.install_manifest, "set_no_torch_marker", return_value = None),
    ):
        with pytest.raises(_StopAfterCorePackages):
            ips.install_python_stack()

    assert calls, "the core-package phase made no pip_install call"
    return calls[0]


def test_update_branch_is_byte_for_byte_unchanged_when_unpinned():
    """`curl | sh` users, CI and unstamped dev builds keep tracking the newest."""
    label, args = _core_package_call({})
    assert label == "Updating core packages"
    assert args == TODAYS_CORE_UPDATE_ARGS


def test_update_branch_installs_the_exact_version_when_pinned():
    label, args = _core_package_call({"UNSLOTH_BACKEND_VERSION": "2026.8.18"})
    assert label == "Updating core packages"
    assert args == (
        "--no-cache-dir",
        # --upgrade-package unsloth is GONE: with an exact ==requirement the flag
        # decides nothing, and would read as "upgrade" and "hold" at once.
        "--upgrade-package",
        "unsloth-zoo",
        "unsloth==2026.8.18",
        "unsloth-zoo",
    )


def test_pinned_update_keeps_the_unsloth_zoo_floor_and_its_upgrade_flag():
    """No zoo version is stamped anywhere, so zoo stays a floor deliberately."""
    _, args = _core_package_call({"UNSLOTH_BACKEND_VERSION": "2026.8.18"})
    assert "unsloth-zoo" in args, "unsloth-zoo must still be installed"
    assert not any(arg.startswith("unsloth-zoo==") for arg in args), (
        "unsloth-zoo must not be pinned to an invented version"
    )
    zoo_flag = [
        index for index, arg in enumerate(args) if arg == "--upgrade-package"
    ]
    assert [args[index + 1] for index in zoo_flag] == ["unsloth-zoo"], (
        "unsloth-zoo must keep its --upgrade-package; unsloth must not have one"
    )


def test_no_torch_update_branch_honours_the_pin_too():
    """The other reachable core-package branch, same rules."""
    label, args = _core_package_call(
        {"UNSLOTH_BACKEND_VERSION": "2026.8.18"}, no_torch = True
    )
    assert label.startswith("Updating unsloth + unsloth-zoo")
    assert args == (
        "--no-cache-dir",
        "--no-deps",
        "--upgrade-package",
        "unsloth-zoo",
        "unsloth==2026.8.18",
        "unsloth-zoo",
    )


def test_no_torch_update_branch_is_unchanged_when_unpinned():
    _, args = _core_package_call({}, no_torch = True)
    assert args == (
        "--no-cache-dir",
        "--no-deps",
        "--upgrade-package",
        "unsloth",
        "--upgrade-package",
        "unsloth-zoo",
        "unsloth",
        "unsloth-zoo",
    )


def test_custom_package_branch_honours_the_pin():
    """--package pins too, exactly as install.sh's _UNSLOTH_SPEC does."""
    label, args = _core_package_call({
        "STUDIO_PACKAGE_NAME": "unsloth-test",
        "UNSLOTH_BACKEND_VERSION": "2026.8.18",
    })
    assert label == "Installing unsloth-test"
    assert args == ("--no-cache-dir", "unsloth-test==2026.8.18")


def test_local_repo_branch_stays_floating():
    """--local ignores the pin: the editable overlay right after is the version the
    developer asked for, so an ==pin would only decide which wheel gets discarded.
    Mirrors install.sh's --local branch."""
    label, args = _core_package_call({
        "STUDIO_LOCAL_REPO": "/some/checkout",
        "UNSLOTH_BACKEND_VERSION": "2026.8.18",
    })
    assert label == "Updating core packages"
    assert args == TODAYS_CORE_UPDATE_ARGS


# ── The injection surface ──────────────────────────────────────────────────────

# Each of these would otherwise be concatenated straight into a requirement
# string that becomes an argv element of `uv pip install`.
MALFORMED_VERSIONS = [
    "1.0 --index-url http://evil",          # a second argument
    "1.0\n--index-url http://evil",         # newline; `$` in a regex would allow a trailing one
    "2026.8.18\n",                          # bare trailing newline
    "1.0;rm -rf /",                         # shell metacharacters
    "unsloth @ git+https://evil/x",         # a whole different requirement
    "1.0 unsloth-zoo==0.0.1",               # a second requirement
    "-rrequirements.txt",                   # a flag, not a version
    "../../etc/passwd",                     # path traversal
    ">=1.0",                                # a specifier, not a version
    "1.0'",                                 # quote
    "  2026.8.18  ",                        # not stripped: install.sh does not strip either
    "2026.8.18\r",                          # carriage return
    "\t1.0",
]


@pytest.mark.parametrize("value", MALFORMED_VERSIONS)
def test_malformed_pin_never_reaches_pip_install(value: str):
    """Rejected outright -- not sanitised, not silently ignored."""
    with pytest.raises(ValueError) as excinfo:
        ips._backend_version_pin(value)
    assert "UNSLOTH_BACKEND_VERSION" in str(excinfo.value)


@pytest.mark.parametrize("value", MALFORMED_VERSIONS)
def test_malformed_pin_stops_the_install_instead_of_installing_anything(value: str):
    """The run must abort before pip_install is reached, and before the manifest
    is removed -- a half-built venv behind a bad pin is worse than no run."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_pip_install(label: str, *args: str, **kwargs):
        calls.append((label, args))
        return None

    with (
        mock.patch.dict(
            os.environ,
            {"SKIP_STUDIO_BASE": "0", "UNSLOTH_BACKEND_VERSION": value},
            clear = False,
        ),
        mock.patch.object(ips, "pip_install", side_effect = fake_pip_install),
        mock.patch.object(ips.install_manifest, "remove_manifest") as remove_manifest,
    ):
        assert ips.install_python_stack() == 1
    assert calls == [], f"pip_install was reached with a malformed pin: {calls!r}"
    remove_manifest.assert_not_called()


@pytest.mark.parametrize(
    "value", ["2026.8.18", "2026.8", "1.0.0rc1", "1.0.0b2", "1.2.3.post1", "1.2.3.dev4"]
)
def test_well_formed_versions_are_accepted(value: str):
    assert ips._backend_version_pin(value) == value


def test_unset_and_empty_mean_unpinned():
    assert ips._backend_version_pin("") == ""
    with mock.patch.dict(os.environ, {}, clear = True):
        assert ips._backend_version_pin() == ""


# ── The desktop update path actually carries the pin ───────────────────────────


def _rust_code(path: Path) -> str:
    """Source with `//` comment lines dropped, so prose cannot satisfy a check."""
    return "\n".join(
        line for line in path.read_text(encoding = "utf-8").splitlines()
        if not line.lstrip().startswith("//")
    )


def test_the_cli_update_command_pops_skip_studio_base():
    """This is why the update branch above is reachable at all."""
    assert 'os.environ.pop("SKIP_STUDIO_BASE", None)' in CLI_STUDIO_PY.read_text(
        encoding = "utf-8"
    )


def test_desktop_update_command_hands_the_stamped_version_to_the_update():
    """The desktop Update and Repair buttons both spawn `unsloth studio update`
    through update.rs, so the pin has to be exported there as well as in install.rs."""
    code = _rust_code(UPDATE_RS)
    assert '&["studio", "update"]' in code, "update.rs no longer runs `unsloth studio update`"
    assert 'option_env!("UNSLOTH_DESKTOP_BACKEND_VERSION")' in code
    assert 'cmd.env("UNSLOTH_BACKEND_VERSION", backend_version)' in code
    assert 'cmd.env_remove("UNSLOTH_BACKEND_VERSION")' in code, (
        "an inherited value must not out-rank the build-time stamp"
    )


def test_desktop_update_leaves_an_unstamped_build_unpinned():
    """Same reasoning as install.rs: the `.unwrap_or(MIN_DESKTOP_BACKEND_VERSION)`
    floor is right for staleness checks and wrong for pinning, because it would pin
    an unstamped local or CI build to a long-stale version."""
    code = _rust_code(UPDATE_RS)
    assert "expected_backend_version" not in code
    assert "MIN_DESKTOP_BACKEND_VERSION" not in code
