# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The first-run install must not name anything that changes underneath it.

Four steps of install_python_stack.py used to: triton_kernels from a git *branch*,
ROCm bitsandbytes from a GitHub tag that is republished in place, the MLX stack with
`--upgrade` and no ceiling, and pip itself at "newest". Each meant one desktop build
(and one `curl | sh`) installed different code on different days with nothing
reviewing the change.

These tests are the guard on all four: the general rule (no requirements file may
reference a moving git ref), the shape of each pin, that every bound actually admits
the version it was resolved against and excludes the next breaking one, and -- driving
the installer itself -- that the pinned values are what reach pip.

See studio/DETERMINISM.md.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from unittest import mock

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = REPO_ROOT / "studio"
REQ_ROOT = STUDIO_DIR / "backend" / "requirements"
TRITON_KERNELS_TXT = REQ_ROOT / "triton-kernels.txt"

if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_pins", STUDIO_DIR / "install_python_stack.py"
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
ips = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = ips
_STACK_SPEC.loader.exec_module(ips)

ALLOW_LATEST = "UNSLOTH_PREBUILT_ALLOW_LATEST"

# `<pkg> @ git+<url>@<ref>` / bare `git+<url>@<ref>`, with the optional #subdirectory
# fragment kept out of the captured ref.
_GIT_REF_RE = re.compile(r"git\+[^\s@]+(?:://|@)[^\s@]*@(?P<ref>[^\s#]+)")
_FULL_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def _requirements_files() -> list[Path]:
    files = sorted(REQ_ROOT.rglob("*.txt"))
    assert files, f"no requirements files found under {REQ_ROOT}"
    return files


def _git_refs(text: str) -> list[str]:
    return [m.group("ref") for m in _GIT_REF_RE.finditer(text)]


# ── 1. triton_kernels: a commit, never a branch ──────────────────────────────


class TestNoMovingGitRefs:
    """The general rule, so a future edit that reintroduces a branch fails here."""

    def test_every_git_requirement_pins_a_full_commit_sha(self):
        offenders: dict[str, list[str]] = {}
        for path in _requirements_files():
            for line in path.read_text(encoding = "utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                bad = [ref for ref in _git_refs(line) if not _FULL_SHA_RE.match(ref)]
                if bad:
                    offenders.setdefault(str(path.relative_to(REPO_ROOT)), []).extend(bad)
        assert not offenders, (
            "requirements files must pin git dependencies to a full 40-character commit "
            f"sha, not a moving branch or tag: {offenders}. Resolve the branch with "
            "`git ls-remote <url> refs/heads/<branch>` and paste the sha."
        )

    def test_the_guard_would_catch_a_branch_coming_back(self):
        """The regex, not the tree: a guard that cannot fail is not a guard."""
        branch_line = (
            "triton_kernels @ git+https://github.com/triton-lang/triton.git"
            "@release/3.6.x#subdirectory=python/triton_kernels"
        )
        assert _git_refs(branch_line) == ["release/3.6.x"]
        assert not _FULL_SHA_RE.match("release/3.6.x")
        # An abbreviated sha is not a stable ref either.
        assert not _FULL_SHA_RE.match("7c56a5e")

    def test_triton_kernels_pin_is_well_formed(self):
        text = TRITON_KERNELS_TXT.read_text(encoding = "utf-8")
        refs = _git_refs(text)
        assert len(refs) == 1, f"expected exactly one git requirement, got {refs}"
        assert _FULL_SHA_RE.match(refs[0]), refs[0]
        # Still the same package, repo and subdirectory -- a pin that moved the source
        # would be a different change wearing this one's clothes.
        assert "github.com/triton-lang/triton.git" in text
        assert "#subdirectory=python/triton_kernels" in text
        assert text.lstrip().startswith("#"), "keep the bump instructions at the top"
        assert "git ls-remote" in text, "the file must say how to re-resolve the pin"

    def test_the_comment_still_names_the_branch_the_pin_came_from(self):
        """A bare sha is unbumpable: the branch it tracks has to stay written down."""
        assert "release/3.6.x" in TRITON_KERNELS_TXT.read_text(encoding = "utf-8")


# ── 2. ROCm bitsandbytes: an immutable release, not a rolling tag ────────────


class TestBnbRocmPin:
    ROLLING_TAG = "continuous-release_main"

    def test_pinned_spec_is_an_exact_version(self):
        req = Requirement(ips._BNB_ROCM_PINNED_SPEC)
        assert req.name == "bitsandbytes"
        assert [(s.operator, s.version) for s in req.specifier] == [("==", "0.50.1")]

    def test_pinned_spec_is_at_or_above_the_shared_rocm_floor(self):
        """The floor exists because bnb <= 0.49.2 NaNs at 4-bit decode on every AMD
        GPU. A pin below it would reinstate that silently."""
        floor = Requirement(ips._BNB_ROCM_PYPI_FALLBACK).specifier
        pinned = Requirement(ips._BNB_ROCM_PINNED_SPEC).specifier
        version = next(iter(pinned)).version
        assert floor.contains(version), f"{version} is below {ips._BNB_ROCM_PYPI_FALLBACK}"

    def test_default_primary_is_the_pinned_release_not_the_rolling_tag(self):
        with mock.patch.dict(os.environ, {}, clear = False):
            os.environ.pop(ALLOW_LATEST, None)
            assert ips._bnb_rocm_prerelease_url() is None
            for arch_key in (None, "win_amd64"):
                spec = ips._bnb_rocm_primary_spec(arch_key)
                assert spec == ips._BNB_ROCM_PINNED_SPEC
                assert self.ROLLING_TAG not in spec

    def test_allow_latest_restores_the_rolling_wheel(self):
        with mock.patch.dict(os.environ, {ALLOW_LATEST: "1"}):
            url = ips._bnb_rocm_primary_spec("win_amd64")
            assert url == ips._BNB_ROCM_PRERELEASE_URLS["win_amd64"]
            assert self.ROLLING_TAG in url

    def test_allow_latest_on_an_arch_with_no_wheel_still_reaches_pypi(self):
        with mock.patch.dict(os.environ, {ALLOW_LATEST: "1"}):
            with mock.patch.object(ips, "_BNB_ROCM_PRERELEASE_URLS", {}):
                assert ips._bnb_rocm_primary_spec("win_amd64") is None
                assert ips._bnb_rocm_primary_spec() is None

    def test_windows_rocm_install_uses_the_pinned_release(self):
        """Functional: the argv Windows ROCm actually hands pip."""
        with mock.patch.dict(os.environ, {}, clear = False):
            os.environ.pop(ALLOW_LATEST, None)
            os.environ.pop("BNB_ROCM_VERSION", None)
            with (
                mock.patch.object(ips, "_persist_bnb_rocm_version", return_value = True),
                mock.patch.object(ips, "_detect_bnb_rocm_dll_ver", return_value = "72"),
                mock.patch.object(ips, "pip_install_try", return_value = True) as pip_try,
            ):
                assert ips._install_bnb_windows_rocm() is True
        assert pip_try.call_count == 1
        args = pip_try.call_args.args
        assert ips._BNB_ROCM_PINNED_SPEC in args
        assert not any("github.com" in str(arg) for arg in args), args

    def test_linux_rocm_install_uses_the_pinned_release(self):
        """Functional: same for the Linux ROCm branch of _ensure_rocm_torch()."""
        probe = mock.MagicMock()
        probe.returncode = 0
        probe.stdout = "\n"  # CPU-only torch, so the ROCm reinstall branch runs
        with mock.patch.dict(os.environ, {}, clear = False):
            os.environ.pop(ALLOW_LATEST, None)
            with (
                mock.patch.object(ips, "IS_WINDOWS", False),
                mock.patch.object(ips, "pip_install") as pip,
                mock.patch.object(ips, "pip_install_try", return_value = True) as pip_try,
                mock.patch.object(ips, "_has_usable_nvidia_gpu", return_value = False),
                mock.patch.object(ips, "_has_rocm_gpu", return_value = True),
                mock.patch.object(ips, "_detect_rocm_version", return_value = (7, 1)),
                mock.patch("os.path.isdir", return_value = True),
                mock.patch("subprocess.run", return_value = probe),
            ):
                ips._ensure_rocm_torch()
        bnb_calls = [c for c in pip_try.call_args_list if "bitsandbytes" in str(c)]
        assert len(bnb_calls) == 1, pip_try.call_args_list
        assert ips._BNB_ROCM_PINNED_SPEC in bnb_calls[0].args
        assert self.ROLLING_TAG not in str(bnb_calls[0])
        # The >=0.50.0 floor is the fallback only; it must not be what got installed.
        assert not any(ips._BNB_ROCM_PYPI_FALLBACK in str(c) for c in pip.call_args_list)

    def test_fallback_floor_is_untouched(self):
        """install.sh and pyproject.toml's amd extra share this constant; the pin must
        not have quietly moved the floor with it."""
        assert ips._BNB_ROCM_PYPI_FALLBACK == "bitsandbytes>=0.50.0"


# ── 3. MLX stack: bounded, and the bound has to mean something ───────────────


class TestMlxStackBounds:
    # The releases the bounds were resolved against, and the first version of each
    # that the ceiling must exclude. mlx is pre-1.0, so a MINOR bump is its breaking
    # release -- that is what <next-minor is protecting against.
    RESOLVED = {
        "mlx": "0.32.1",
        "mlx-metal": "0.32.1",
        "mlx-lm": "0.31.3",
        "mlx-vlm": "0.6.15",
    }
    NEXT_BREAKING = {
        "mlx": "0.33.0",
        "mlx-metal": "0.33.0",
        "mlx-lm": "0.32.0",
        "mlx-vlm": "0.7.0",
    }

    def _requirements(self) -> dict[str, Requirement]:
        reqs = [Requirement(spec) for spec in ips._MLX_STACK_SPECS]
        return {req.name: req for req in reqs}

    def test_all_four_packages_are_still_installed(self):
        assert set(self._requirements()) == set(self.RESOLVED)
        assert set(ips._MLX_STACK_UNBOUNDED) == set(self.RESOLVED)

    def test_every_spec_has_a_ceiling(self):
        for name, req in self._requirements().items():
            operators = {s.operator for s in req.specifier}
            assert "<" in operators, f"{name} has no ceiling: {req}"

    def test_no_spec_adds_a_floor(self):
        """Deliberate: with --upgrade a floor cannot change what a healthy install
        resolves to, so it would only change the failing case -- and there it turns a
        degraded-but-installed stack (which the health probe reports and the self-heal
        retries) into a fatal macOS install error, because without uv there is no
        UV_OVERRIDE to relax mlx-vlm's transformers floor against the pinned
        transformers==5.5.0. If a floor is ever wanted, that trade has to be made
        knowingly, so it fails here first."""
        for name, req in self._requirements().items():
            operators = {s.operator for s in req.specifier}
            assert operators == {"<"}, f"{name} gained a bound beyond its ceiling: {req}"

    @pytest.mark.parametrize("name", sorted(RESOLVED))
    def test_bound_admits_the_resolved_version(self, name):
        req = self._requirements()[name]
        current = self.RESOLVED[name]
        assert req.specifier.contains(current), f"{req} excludes {current}"

    @pytest.mark.parametrize("name", sorted(RESOLVED))
    def test_bound_admits_a_patch_release_but_excludes_the_next_minor(self, name):
        """A patch inside the window must still reach users -- an exact == would strand
        Apple Silicon on a broken combination as readily as no ceiling does."""
        req = self._requirements()[name]
        resolved = Version(self.RESOLVED[name])
        patch = f"{resolved.major}.{resolved.minor}.{resolved.micro + 1}"
        assert req.specifier.contains(patch), f"{req} excludes the patch {patch}"
        breaking = self.NEXT_BREAKING[name]
        assert not req.specifier.contains(breaking), f"{req} admits {breaking}"

    def test_ceiling_is_the_next_minor_of_the_resolved_release(self):
        """Not <=current and not the next *major*: mlx is pre-1.0, so a minor bump is
        its breaking release."""
        for name, req in self._requirements().items():
            ceiling = Version(next(iter(req.specifier)).version)
            resolved = Version(self.RESOLVED[name])
            assert (ceiling.major, ceiling.minor) == (resolved.major, resolved.minor + 1), (
                f"{name}: ceiling {ceiling} is not the minor after {resolved}"
            )

    def test_mlx_and_mlx_metal_share_one_window(self):
        """mlx declares `mlx-metal==<its own version>` on Darwin, so two different
        windows here would be unsatisfiable rather than merely wrong."""
        reqs = self._requirements()
        assert str(reqs["mlx"].specifier) == str(reqs["mlx-metal"].specifier)

    def test_allow_latest_restores_the_unbounded_names(self):
        with mock.patch.dict(os.environ, {ALLOW_LATEST: "1"}):
            assert ips._mlx_stack_specs() == ips._MLX_STACK_UNBOUNDED
        with mock.patch.dict(os.environ, {}, clear = False):
            os.environ.pop(ALLOW_LATEST, None)
            assert ips._mlx_stack_specs() == ips._MLX_STACK_SPECS


# ── 4. pip: an exact version ─────────────────────────────────────────────────


class TestPipBootstrapPin:
    def test_version_is_a_release_version(self):
        version = Version(ips._PIP_BOOTSTRAP_VERSION)
        assert not version.is_prerelease and not version.is_devrelease
        assert str(version) == ips._PIP_BOOTSTRAP_VERSION

    def test_spec_is_exact_by_default(self):
        with mock.patch.dict(os.environ, {}, clear = False):
            os.environ.pop(ALLOW_LATEST, None)
            spec = ips._pip_bootstrap_spec()
        req = Requirement(spec)
        assert req.name == "pip"
        assert [(s.operator, s.version) for s in req.specifier] == [
            ("==", ips._PIP_BOOTSTRAP_VERSION)
        ]

    def test_allow_latest_restores_the_bare_name(self):
        with mock.patch.dict(os.environ, {ALLOW_LATEST: "1"}):
            assert ips._pip_bootstrap_spec() == "pip"


# ── The installer end to end (mac arm64: pip bootstrap + MLX step) ───────────


class _StopAfterMlx(Exception):
    """Raised by the pip_install stub to end the run at the step under test."""


def _mac_arm_install(env: dict[str, str]) -> tuple[list[list[str]], tuple[str, ...]]:
    """Run install_python_stack() on a simulated Apple Silicon host.

    Returns (argv of every run() call before the MLX step, args of the MLX
    pip_install). The MLX step is the first pip_install a mac-arm fresh install makes,
    so stopping there also captures the pip bootstrap that precedes it.
    """
    runs: list[list[str]] = []
    mlx_args: list[tuple[str, ...]] = []

    def fake_run(_label, cmd, *a, **kw):
        runs.append(list(cmd))
        return None

    def fake_pip_install(_label, *args, **kwargs):
        mlx_args.append(args)
        raise _StopAfterMlx

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
        mock.patch.object(ips, "run", side_effect = fake_run),
        mock.patch.object(ips, "_progress", return_value = None),
        mock.patch.object(ips, "_step", return_value = None),
        mock.patch.object(ips, "_bootstrap_uv", return_value = False),
        mock.patch.object(ips, "IS_MAC_ARM", True),
        mock.patch.object(ips, "IS_MACOS", True),
        mock.patch.object(ips, "IS_WINDOWS", False),
        mock.patch.object(ips, "NO_TORCH", False),
        mock.patch.object(ips, "_bitsandbytes_installed", return_value = True),
        mock.patch.object(ips.install_manifest, "remove_manifest", return_value = True),
        mock.patch.object(ips.install_manifest, "set_no_torch_marker", return_value = None),
        # The non-uv bootstrap branch shells out to check for pip; say it is there so
        # the pinned-install branch (not ensurepip) is the one exercised.
        mock.patch.object(
            ips.subprocess, "run", return_value = mock.MagicMock(returncode = 0)
        ),
    ):
        with pytest.raises(_StopAfterMlx):
            ips.install_python_stack()

    assert mlx_args, "the MLX step made no pip_install call"
    return runs, mlx_args[0]


def _pip_bootstrap_argv(runs: list[list[str]]) -> list[str]:
    matches = [cmd for cmd in runs if "install" in cmd and any("pip" in part for part in cmd)]
    assert matches, f"no pip bootstrap command in {runs}"
    return matches[0]


class TestInstallerAppliesThePins:
    def test_pip_is_bootstrapped_at_the_pinned_version(self):
        runs, _ = _mac_arm_install({})
        argv = _pip_bootstrap_argv(runs)
        assert f"pip=={ips._PIP_BOOTSTRAP_VERSION}" in argv, argv
        assert argv[-1] != "pip", f"bootstrap still installs unpinned pip: {argv}"

    def test_mlx_step_installs_the_bounded_specs(self):
        _, args = _mac_arm_install({})
        assert set(ips._MLX_STACK_SPECS) <= set(args), args
        # --upgrade stays: with ceilings it means "newest inside the window".
        assert "--upgrade" in args
        # No bare package name may survive alongside the bounded spec, or the
        # resolver is free to satisfy the bare one with anything.
        assert not (set(ips._MLX_STACK_UNBOUNDED) & set(args)), args

    def test_allow_latest_reproduces_the_pre_pin_argv(self):
        runs, args = _mac_arm_install({ALLOW_LATEST: "1"})
        assert _pip_bootstrap_argv(runs)[-1] == "pip"
        assert set(ips._MLX_STACK_UNBOUNDED) <= set(args), args
        assert not any("<" in arg for arg in args), args
