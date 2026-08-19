# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The hash-verified Python locks, and the installer's use of them.

Every requirements file this repo ships is version-pinned, but a pin is not a
digest: `pip install -r studio.txt` re-resolves the transitive closure from PyPI at
install time and installs whatever bytes come back. studio/backend/requirements/locks/
closes that gap for the torch-independent steps -- one machine-generated, fully
hashed resolution per step, installed with `--require-hashes`.

These tests are the guard on the parts that can rot silently:

  1. the locks themselves -- every requirement pinned with `==`, every requirement
     carrying at least one sha256, no range specifier anywhere, and the entries that
     deliberately cannot be locked absent from them;
  2. the wiring -- a locked step really does pass `--require-hashes` and really does
     drop `-c constraints.txt` (which `--require-hashes` rejects outright), an
     unlocked step is untouched, the escape hatch reverts everything, and a missing
     lock warns and falls back rather than failing an older wheel's install;
  3. the invariants a future requirements edit could break -- a new requirements file
     that is neither locked nor recorded as deliberately unlocked, a lock that
     contradicts a version pinned elsewhere in the same environment, or a skip filter
     that would leave orphaned `--hash=` lines behind in a lock.

Regenerate the locks with scripts/gen_python_locks.sh. See studio/DETERMINISM.md.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "studio"
REQ_ROOT = STUDIO_DIR / "backend" / "requirements"
LOCK_ROOT = REQ_ROOT / "locks"
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_python_locks.sh"

if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_locks", STUDIO_DIR / "install_python_stack.py"
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
ips = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = ips
_STACK_SPEC.loader.exec_module(ips)

NO_LOCK_ENV = "UNSLOTH_PYTHON_NO_LOCK"

# A requirement line in a lock: `name==version [; marker] \`. uv writes the hashes as
# indented continuation lines below it, so anything indented is not a requirement.
_REQ_LINE = re.compile(r"^[A-Za-z0-9]")
# Anything that is not an exact pin. `--require-hashes` refuses all of them, and uv
# says so plainly: "all requirements must have their versions pinned with '=='".
_RANGE_OPS = ("<", ">", "!=", "~=", "===")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _lock_files() -> list[Path]:
    files = sorted(LOCK_ROOT.glob("*.lock.txt"))
    assert files, f"no locks found under {LOCK_ROOT}; run scripts/gen_python_locks.sh"
    return files


def _entries(lock: Path) -> list[tuple[str, str, list[str]]]:
    """(name, spec, hashes) for every requirement in a lock."""
    out: list[tuple[str, str, list[str]]] = []
    current: tuple[str, str, list[str]] | None = None
    for raw in lock.read_text(encoding = "utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if _REQ_LINE.match(raw):
            spec = raw.split("\\", 1)[0].strip()
            name = re.split(r"[<>=!~;\[\s]", spec, maxsplit = 1)[0]
            current = (_normalise(name), spec, [])
            out.append(current)
        elif current is not None:
            for token in raw.replace("\\", " ").split():
                if token.startswith("--hash="):
                    current[2].append(token)
    return out


def _exact_pins(req: Path) -> dict[str, str]:
    """{normalised name: version} for every UNCONDITIONAL `name==version` in a file.

    Marked pins are skipped on both sides of the comparison below. A marked pin is a
    claim about one fork only -- `huggingface-hub==0.36.2; python_version < "3.10"`
    says nothing about the 3.10+ resolution a lock records -- so comparing it to an
    unconditional lock entry would report a contradiction that is not one.
    """
    pins: dict[str, str] = {}
    for raw in req.read_text(encoding = "utf-8").splitlines():
        spec = raw.split("#", 1)[0].strip()
        if not spec or spec.startswith("-") or ";" in spec:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s,]+)", spec)
        if match:
            pins[_normalise(match.group(1))] = match.group(2)
    return pins


# ── 1. the locks themselves ──────────────────────────────────────────────────


class TestLockContents:
    def test_a_lock_exists_for_every_file_the_installer_expects_one(self):
        missing = sorted(
            name
            for name in ips.LOCKED_REQUIREMENTS
            if not (LOCK_ROOT / f"{Path(name).stem}.lock.txt").is_file()
        )
        assert not missing, (
            f"LOCKED_REQUIREMENTS names {missing} but no lock is committed for them. "
            f"Run: bash scripts/gen_python_locks.sh"
        )

    def test_every_requirement_in_every_lock_carries_a_sha256(self):
        """The whole point. One unhashed line makes the file unusable under
        --require-hashes, so a partial lock is not a weaker lock, it is a failed
        install -- and a lock generated any way other than by uv is how that happens.
        """
        offenders: dict[str, list[str]] = {}
        for lock in _lock_files():
            bad = [spec for _, spec, hashes in _entries(lock) if not hashes]
            if bad:
                offenders[lock.name] = bad
        assert not offenders, f"requirements with no --hash: {offenders}"

    def test_every_hash_is_a_well_formed_sha256(self):
        offenders: dict[str, list[str]] = {}
        for lock in _lock_files():
            bad = [
                h
                for _, _, hashes in _entries(lock)
                for h in hashes
                if not re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", h)
            ]
            if bad:
                offenders[lock.name] = bad
        assert not offenders, f"malformed hashes: {offenders}"

    def test_no_lock_contains_a_range_specifier(self):
        """`uv pip install --require-hashes` fails the whole step on the first one:
        "all requirements must have their versions pinned with '=='". A range in a
        lock is therefore not a loose pin, it is a broken install."""
        offenders: dict[str, list[str]] = {}
        for lock in _lock_files():
            bad = []
            for _, spec, _ in _entries(lock):
                requirement = spec.split(";", 1)[0]
                if "==" not in requirement or any(op in requirement for op in _RANGE_OPS):
                    bad.append(spec)
            if bad:
                offenders[lock.name] = bad
        assert not offenders, f"non-exact requirements in a lock: {offenders}"

    def test_no_lock_names_a_url_or_vcs_requirement(self):
        """A `git+` requirement cannot carry a hash at all, and a direct URL would
        pin bytes off a host that is not the index the digests were taken from."""
        offenders: dict[str, list[str]] = {}
        for lock in _lock_files():
            bad = [spec for _, spec, _ in _entries(lock) if "://" in spec or "git+" in spec]
            if bad:
                offenders[lock.name] = bad
        assert not offenders, f"URL/VCS requirements in a lock: {offenders}"

    def test_every_lock_records_the_generator_and_the_python_floor(self):
        for lock in _lock_files():
            text = lock.read_text(encoding = "utf-8")
            assert "GENERATED FILE -- DO NOT EDIT" in text, lock.name
            assert "scripts/gen_python_locks.sh" in text, lock.name
            assert re.search(r"^# source: ", text, re.M), lock.name
            assert re.search(r"^# generated by: uv \d+\.\d+\.\d+", text, re.M), lock.name
            assert ips._lock_python_floor(text) is not None, (
                f"{lock.name} has no `# unsloth-lock-python-floor:` line, so "
                f"install_python_stack.py cannot tell whether this interpreter may use it"
            )

    def test_the_locks_were_generated_by_the_uv_the_installer_pins(self):
        """A lock built by a different resolver is a lock for an install nobody
        performs, and nothing in the file's contents would show it."""
        pinned = re.search(
            r'^UV_PINNED_VERSION="([^"]+)"',
            (REPO_ROOT / "install.sh").read_text(encoding = "utf-8"),
            re.M,
        )
        assert pinned, "could not read UV_PINNED_VERSION out of install.sh"
        for lock in _lock_files():
            assert f"# generated by: uv {pinned.group(1)}" in lock.read_text(encoding = "utf-8"), (
                f"{lock.name} was not generated by uv {pinned.group(1)}; "
                f"regenerate it with that uv"
            )


# ── 2. the carve-outs ────────────────────────────────────────────────────────


class TestCarveOuts:
    """Entries whose real bound is not expressible as a PEP 508 marker.

    `pytorch_tokenizers<=1.4.1` is the case: 1.4.1 ships no musllinux wheel and its
    arm64 wheel is macosx_14_0, so a musl host needs 1.1.0 and a macOS 13 arm64 host
    needs 1.2.0. A universal lock names one version, and naming 1.4.1 hands both of
    those hosts the single release carrying an sdist -- the cmake build
    extras-no-deps.txt is written to avoid. So the cap stays out of the lock.
    """

    CARVED_OUT = ("pytorch_tokenizers",)

    def test_the_carve_outs_are_absent_from_every_lock(self):
        carved = {_normalise(n) for n in self.CARVED_OUT}
        offenders: dict[str, list[str]] = {}
        for lock in _lock_files():
            bad = [spec for name, spec, _ in _entries(lock) if name in carved]
            if bad:
                offenders[lock.name] = bad
        assert not offenders, (
            f"a carve-out was pinned into a lock: {offenders}. A universal lock names "
            f"one version, and for these entries that breaks a supported platform."
        )

    def test_the_carve_outs_are_still_installed_from_the_side_file(self):
        """Carving an entry out of the lock must not drop it from the install."""
        side = LOCK_ROOT / "extras-no-deps.unlocked.txt"
        assert side.is_file(), f"{side} is missing; run scripts/gen_python_locks.sh"
        specs = [
            line.split("#", 1)[0].strip()
            for line in side.read_text(encoding = "utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        ]
        for name in self.CARVED_OUT:
            assert any(_normalise(spec.split(";")[0]).startswith(_normalise(name)) for spec in specs), (
                f"{name} is carved out of the lock but not present in {side.name}, so "
                f"nothing installs it"
            )

    def test_the_side_file_keeps_the_cap_that_is_the_reason_for_the_carve_out(self):
        side = LOCK_ROOT / "extras-no-deps.unlocked.txt"
        text = side.read_text(encoding = "utf-8")
        assert "pytorch_tokenizers<=1.4.1" in text, (
            "the carve-out is only worth anything as a CAP: pinned exactly it has the "
            "same effect as leaving it in the lock"
        )

    def test_the_carved_entry_is_still_declared_in_its_requirements_file(self):
        """The source file stays the single place the requirement is written; the
        generator splits it, so a reader of extras-no-deps.txt still sees it."""
        text = (REQ_ROOT / "extras-no-deps.txt").read_text(encoding = "utf-8")
        assert "pytorch_tokenizers<=1.4.1" in text

    def test_the_installer_finds_the_side_file_for_the_lock_that_has_one(self):
        lock = LOCK_ROOT / "extras-no-deps.lock.txt"
        assert ips._lock_carve_out(lock) == LOCK_ROOT / "extras-no-deps.unlocked.txt"
        # And reports None for the locks that have no carve-outs, so no phantom
        # second install is scheduled.
        assert ips._lock_carve_out(LOCK_ROOT / "studio.lock.txt") is None


# ── 3. coverage: locked, or recorded as deliberately unlocked ────────────────


class TestCoverage:
    def test_every_requirements_file_is_either_locked_or_has_a_recorded_reason(self):
        """Fails when a new requirements file lands with neither a lock nor an entry
        in UNLOCKED_REQUIREMENTS saying why it has none. The reason is the point: the
        three that exist today (a torch-bound closure, a GitHub archive, a git ref)
        are each a real obstacle, and "nobody got round to it" should not read the same.
        """
        installed_with_r = {
            path.name
            for path in sorted(REQ_ROOT.glob("*.txt"))
            # `.{stem}-filtered-*.txt` is _filter_requirements' temp, left beside the
            # source on purpose and possibly present on a machine that has installed.
            if not re.fullmatch(r"\..+-filtered-\w+\.txt", path.name)
        }
        # single-env/ holds two files applied with -r and two that are only ever -c
        # / --overrides inputs.
        installed_with_r |= {"data-designer-deps.txt", "data-designer.txt"}
        accounted = set(ips.LOCKED_REQUIREMENTS) | set(ips.UNLOCKED_REQUIREMENTS)
        unaccounted = sorted(installed_with_r - accounted)
        assert not unaccounted, (
            f"{unaccounted} is applied with -r but is neither in LOCKED_REQUIREMENTS "
            f"nor recorded in UNLOCKED_REQUIREMENTS with a reason "
            f"(studio/install_python_stack.py)"
        )

    def test_no_file_is_both_locked_and_recorded_as_unlocked(self):
        both = sorted(set(ips.LOCKED_REQUIREMENTS) & set(ips.UNLOCKED_REQUIREMENTS))
        assert not both, both

    def test_each_recorded_reason_is_an_actual_reason(self):
        for name, reason in ips.UNLOCKED_REQUIREMENTS.items():
            assert len(reason) >= 20, f"{name}: {reason!r} does not say why"

    def test_the_git_pinned_file_stays_unlocked(self):
        """triton-kernels.txt is a `git+https` requirement. A VCS requirement cannot
        carry a hash, so locking it would break the step, not harden it."""
        assert "triton-kernels.txt" in ips.UNLOCKED_REQUIREMENTS
        assert "git+" in (REQ_ROOT / "triton-kernels.txt").read_text(encoding = "utf-8")


# ── 4. cross-file consistency ────────────────────────────────────────────────


class TestNoLockContradictsAnotherPin:
    """The install applies these files one after another into ONE environment.

    An unlocked step leaves an already-satisfied range alone; a lock pins, so it can
    silently override a pin from another file. Measured before the generator started
    passing the other files as constraints: an independent resolution of
    data-designer-deps.txt wanted pymupdf 1.28.2 (the lockstep line studio.txt's
    comment says to stay off, because it makes pymupdf-layout -> onnxruntime a hard
    dep), tiktoken 0.14.0 over extras.txt's 0.13.0 and uvicorn 0.52.3 over
    studio.txt's 0.52.1.
    """

    @staticmethod
    def _lock_pins(lock: Path) -> dict[str, str]:
        pins: dict[str, str] = {}
        for name, spec, _ in _entries(lock):
            requirement = spec.split(";", 1)[0].strip()
            marker = spec.split(";", 1)[1].strip() if ";" in spec else ""
            if "==" not in requirement:
                continue
            # A forked entry (`foo==1 ; python_full_version < '3.12'`) is a different
            # claim per fork, so only the unconditional ones are compared.
            if marker:
                continue
            pins[name] = requirement.split("==", 1)[1]
        return pins

    def test_no_lock_pins_a_version_another_requirements_file_pins_differently(self):
        sources = [
            REQ_ROOT / "studio.txt",
            REQ_ROOT / "extras.txt",
            REQ_ROOT / "extras-no-deps.txt",
            REQ_ROOT / "no-torch-runtime.txt",
            REQ_ROOT / "single-env" / "constraints.txt",
            REQ_ROOT / "single-env" / "data-designer-deps.txt",
        ]
        offenders: list[str] = []
        for lock in _lock_files():
            lock_pins = self._lock_pins(lock)
            for source in sources:
                if lock.name == f"{source.stem}.lock.txt":
                    continue
                for name, version in _exact_pins(source).items():
                    locked = lock_pins.get(name)
                    if locked is not None and locked != version:
                        offenders.append(
                            f"{lock.name} pins {name}=={locked} but {source.name} "
                            f"pins {version}"
                        )
        assert not offenders, (
            "a lock contradicts a version pinned elsewhere in the same environment: "
            + "; ".join(offenders)
        )

    def test_two_locks_never_pin_the_same_package_differently(self):
        pins: dict[str, tuple[str, str]] = {}
        offenders: list[str] = []
        for lock in _lock_files():
            for name, version in self._lock_pins(lock).items():
                seen = pins.get(name)
                if seen is not None and seen[1] != version:
                    offenders.append(
                        f"{name}: {seen[0]} pins {seen[1]}, {lock.name} pins {version}"
                    )
                else:
                    pins[name] = (lock.name, version)
        assert not offenders, (
            "the later step would install over the earlier one: " + "; ".join(offenders)
        )


# ── 5. the generator ─────────────────────────────────────────────────────────


class TestGenerator:
    def test_the_generator_exists_and_is_syntactically_valid(self):
        assert GEN_SCRIPT.is_file()
        result = subprocess.run(
            ["bash", "-n", str(GEN_SCRIPT)],
            capture_output = True,
            text = True,
        )
        assert result.returncode == 0, result.stderr

    def test_the_generator_refuses_a_uv_that_is_not_the_pinned_one(self, tmp_path):
        """The version check is the load-bearing part of "machine-generated": with it
        off, whatever uv happens to be on PATH decides what the locks say."""
        fake = tmp_path / "uv"
        fake.write_text("#!/bin/sh\necho 'uv 0.1.0 (fake)'\n", encoding = "utf-8")
        fake.chmod(0o755)
        result = subprocess.run(
            ["bash", str(GEN_SCRIPT)],
            capture_output = True,
            text = True,
            cwd = REPO_ROOT,
            env = {"PATH": "/usr/bin:/bin", "UV": str(fake), "HOME": str(tmp_path)},
        )
        assert result.returncode != 0
        assert "install.sh pins uv" in result.stderr, result.stderr

    def test_the_generator_reads_its_uv_pin_out_of_install_sh(self):
        text = GEN_SCRIPT.read_text(encoding = "utf-8")
        assert "UV_PINNED_VERSION" in text
        assert 'install.sh"' in text, "the pin must be read, not copied"

    def test_the_generator_pins_the_index_so_regeneration_is_reproducible(self):
        """Without --exclude-newer the freshness lane fails on any day upstream
        publishes anything, which trains everyone to ignore it."""
        text = GEN_SCRIPT.read_text(encoding = "utf-8")
        assert "--exclude-newer" in text
        assert re.search(r'EXCLUDE_NEWER="\d{4}-\d{2}-\d{2}T', text)

    def test_the_generator_asks_uv_for_hashes_and_a_universal_resolution(self):
        text = GEN_SCRIPT.read_text(encoding = "utf-8")
        assert "--generate-hashes" in text
        assert "--universal" in text

    def test_the_torch_bound_and_vcs_files_are_not_fed_to_the_generator(self):
        """extras.txt / diffusers-pin.txt / triton-kernels.txt must not become
        compile inputs. They may appear as constraints or in comments, so this checks
        the compile_lock call sites."""
        text = GEN_SCRIPT.read_text(encoding = "utf-8")
        for name in ("extras.txt", "diffusers-pin.txt", "triton-kernels.txt"):
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("compile_lock") and name in stripped:
                    pytest.fail(f"{name} is a compile input: {stripped}")


class TestFreshnessLane:
    WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python-lock-freshness.yml"

    def test_the_lane_exists_and_runs_the_generator_in_check_mode(self):
        assert self.WORKFLOW.is_file()
        text = self.WORKFLOW.read_text(encoding = "utf-8")
        assert "scripts/gen_python_locks.sh --check" in text

    def test_the_lane_fires_on_a_requirements_edit(self):
        yaml = pytest.importorskip("yaml")
        doc = yaml.safe_load(self.WORKFLOW.read_text(encoding = "utf-8"))
        on = doc.get(True) or doc.get("on")  # PyYAML reads a bare `on:` as True
        assert "pull_request" in on and "push" in on
        for event in ("pull_request", "push"):
            paths = on[event]["paths"]
            assert "studio/backend/requirements/**" in paths, event
            assert "scripts/gen_python_locks.sh" in paths, event
        assert doc["permissions"] == {"contents": "read"}
        assert "concurrency" in doc


# ── 6. the wiring: what argv a step actually gets ────────────────────────────


@pytest.fixture
def install_calls(monkeypatch):
    """Record the argv of every install command pip_install would run."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = b""

    def _fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(ips.subprocess, "run", _fake_run)
    monkeypatch.setattr(ips, "USE_UV", True)
    monkeypatch.setattr(ips, "UV_NEEDS_SYSTEM", False)
    monkeypatch.setattr(ips, "IS_WINDOWS", False)
    monkeypatch.setattr(ips, "NO_TORCH", False)
    monkeypatch.setattr(ips, "PLATFORM_LACKS_TORCHCODEC_WHEEL", False)
    monkeypatch.delenv(NO_LOCK_ENV, raising = False)
    monkeypatch.delenv("UV_TORCH_BACKEND", raising = False)
    return calls


class TestLockedStepArgv:
    def test_a_locked_step_installs_the_lock_with_require_hashes_and_no_constraint(
        self, install_calls
    ):
        ips.pip_install("studio deps", "--no-cache-dir", req = REQ_ROOT / "studio.txt")
        assert len(install_calls) == 1
        cmd = install_calls[0]
        assert "--require-hashes" in cmd
        assert "-r" in cmd
        assert cmd[cmd.index("-r") + 1].endswith("locks/studio.lock.txt")
        # -c constraints.txt would fail the step outright: constraints.txt carries
        # `packaging<27`, `av<16` and `anyio<4.14.0`, and --require-hashes rejects a
        # range. The compile applies the constraints instead.
        assert "-c" not in cmd, cmd
        assert not any(arg.endswith("studio.txt") for arg in cmd), cmd

    def test_a_no_deps_locked_step_keeps_no_deps(self, install_calls):
        ips.pip_install(
            "extras (no-deps)",
            "--no-deps",
            "--no-cache-dir",
            req = REQ_ROOT / "extras-no-deps.txt",
        )
        assert all("--no-deps" in cmd for cmd in install_calls), install_calls

    def test_a_locked_step_with_carve_outs_installs_the_side_file_unlocked(
        self, install_calls
    ):
        ips.pip_install(
            "extras (no-deps)",
            "--no-deps",
            "--no-cache-dir",
            req = REQ_ROOT / "extras-no-deps.txt",
        )
        assert len(install_calls) == 2, install_calls
        locked, unlocked = install_calls
        assert "--require-hashes" in locked
        assert locked[locked.index("-r") + 1].endswith("locks/extras-no-deps.lock.txt")
        # The carve-out is the one entry that cannot be hashed into a universal lock,
        # so it is installed the old way -- WITH the constraints, which is what bounds
        # it, and WITHOUT --require-hashes, which would reject the cap.
        assert "--require-hashes" not in unlocked
        assert unlocked[unlocked.index("-r") + 1].endswith("locks/extras-no-deps.unlocked.txt")
        assert "-c" in unlocked
        assert unlocked[unlocked.index("-c") + 1].endswith("single-env/constraints.txt")

    def test_an_unlocked_step_is_completely_unchanged(self, install_calls):
        """extras.txt is torch-bound, so it must still go through the old path:
        constraints applied, no --require-hashes, the source file itself."""
        ips.pip_install("extras", "--no-cache-dir", req = REQ_ROOT / "extras.txt")
        assert len(install_calls) == 1
        cmd = install_calls[0]
        assert "--require-hashes" not in cmd
        assert cmd[cmd.index("-c") + 1].endswith("single-env/constraints.txt")
        assert cmd[cmd.index("-r") + 1].endswith("requirements/extras.txt")

    def test_a_step_with_no_requirements_file_is_unchanged(self, install_calls):
        ips.pip_install("core packages", "--no-cache-dir", "unsloth")
        cmd = install_calls[0]
        assert "--require-hashes" not in cmd
        assert "unsloth" in cmd

    def test_every_locked_requirements_file_resolves_to_its_lock(self, install_calls):
        for name in sorted(ips.LOCKED_REQUIREMENTS):
            install_calls.clear()
            source = REQ_ROOT / name
            if not source.is_file():
                source = REQ_ROOT / "single-env" / name
            ips.pip_install(name, "--no-cache-dir", req = source)
            first = install_calls[0]
            assert "--require-hashes" in first, name
            assert first[first.index("-r") + 1].endswith(
                f"locks/{Path(name).stem}.lock.txt"
            ), name


class TestEscapeHatch:
    def test_the_escape_hatch_restores_the_pre_lock_command(
        self, install_calls, monkeypatch
    ):
        monkeypatch.setenv(NO_LOCK_ENV, "1")
        ips.pip_install("studio deps", "--no-cache-dir", req = REQ_ROOT / "studio.txt")
        assert len(install_calls) == 1
        cmd = install_calls[0]
        assert "--require-hashes" not in cmd
        assert cmd[cmd.index("-c") + 1].endswith("single-env/constraints.txt")
        assert cmd[cmd.index("-r") + 1].endswith("requirements/studio.txt")
        assert not any("locks/" in arg for arg in cmd), cmd

    def test_the_escape_hatch_also_drops_the_pip_bootstrap_lock(self, monkeypatch):
        monkeypatch.setenv(NO_LOCK_ENV, "1")
        assert ips._pip_bootstrap_lock() is None

    def test_only_the_documented_value_disables_the_locks(self, monkeypatch):
        monkeypatch.setenv(NO_LOCK_ENV, "0")
        assert ips._python_locks_enabled()
        monkeypatch.setenv(NO_LOCK_ENV, "1")
        assert not ips._python_locks_enabled()


class TestMissingLockFallsBack:
    def test_a_missing_lock_warns_and_installs_the_unlocked_file(
        self, install_calls, monkeypatch, tmp_path, capsys
    ):
        """A wheel built before the locks shipped must still install. Failing closed
        here would turn a hardening change into an outage for every older install."""
        monkeypatch.setattr(ips, "LOCK_ROOT", tmp_path / "locks")
        ips.pip_install("studio deps", "--no-cache-dir", req = REQ_ROOT / "studio.txt")
        out = capsys.readouterr().out
        assert "no hash-verified lock" in out
        # It has to say the install went ahead unverified, and where to fix it. The
        # generator is named in the code comment rather than the message: the
        # installer-helper scan in tests/test_installer_interactive_prompts.py treats
        # any *.sh named outside a comment as a script the installer invokes.
        assert "unverified" in out
        assert "need regenerating" in out
        assert len(install_calls) == 1
        cmd = install_calls[0]
        assert "--require-hashes" not in cmd
        assert cmd[cmd.index("-r") + 1].endswith("requirements/studio.txt")

    def test_a_file_we_do_not_lock_warns_about_nothing(
        self, install_calls, monkeypatch, capsys
    ):
        """The warning has to mean something. extras.txt has no lock on purpose, so a
        warning for it would be noise and would train people past the real one."""
        ips.pip_install("extras", "--no-cache-dir", req = REQ_ROOT / "extras.txt")
        assert "no hash-verified lock" not in capsys.readouterr().out


class TestPythonFloor:
    def test_a_lock_is_skipped_below_the_python_it_was_resolved_for(
        self, install_calls, monkeypatch
    ):
        """The locks are universal across platforms but resolved from a 3.10 floor, so
        their `python_version < "3.10"` branches were resolved away. Handing that to a
        3.9 host would quietly install a subset of the file."""
        monkeypatch.setattr(ips.sys, "version_info", (3, 9, 18, "final", 0))
        ips.pip_install("studio deps", "--no-cache-dir", req = REQ_ROOT / "studio.txt")
        cmd = install_calls[0]
        assert "--require-hashes" not in cmd
        assert cmd[cmd.index("-r") + 1].endswith("requirements/studio.txt")

    def test_the_floor_is_read_out_of_the_header(self):
        assert ips._lock_python_floor("# unsloth-lock-python-floor: 3.10\nfoo==1\n") == (3, 10)
        # A pin before the marker means the header ended; do not scan a 3000-line body.
        assert ips._lock_python_floor("foo==1\n# unsloth-lock-python-floor: 3.10\n") is None
        assert ips._lock_python_floor("# nothing here\nfoo==1\n") is None


class TestPipBootstrapLock:
    def test_the_bootstrap_lock_pins_exactly_the_version_the_module_asks_for(self):
        lock = ips._pip_bootstrap_lock()
        assert lock is not None, "the pip bootstrap lock is missing"
        assert f"pip=={ips._PIP_BOOTSTRAP_VERSION}" in lock.read_text(encoding = "utf-8")

    def test_a_stale_bootstrap_lock_is_refused_rather_than_installed(
        self, monkeypatch, tmp_path, capsys
    ):
        """A _PIP_BOOTSTRAP_VERSION bump with the lock left behind must not silently
        install the old pip: the bump is the thing being asked for."""
        stale = tmp_path / "locks"
        stale.mkdir()
        (stale / "pip-bootstrap.lock.txt").write_text(
            "# unsloth-lock-python-floor: 3.10\npip==1.2.3 \\\n"
            f"    --hash=sha256:{'0' * 64}\n",
            encoding = "utf-8",
        )
        monkeypatch.setattr(ips, "LOCK_ROOT", stale)
        assert ips._pip_bootstrap_lock() is None
        assert "does not pin pip==" in capsys.readouterr().out

    def test_allow_latest_gives_up_the_lock_on_purpose(self, monkeypatch):
        """ALLOW_LATEST asks for "whatever pip is newest today", which no lock can
        express. It must revert to the spec rather than pin through the escape hatch."""
        monkeypatch.setattr(ips, "_allow_latest_pins", lambda: True)
        assert ips._pip_bootstrap_lock() is None


# ── 7. the filters that also run over the locks ──────────────────────────────


class TestFilterRequirementsOnALock:
    """NO_TORCH / Windows / torchcodec skips run over whatever file is installed, and
    that is now sometimes a lock. uv writes hashes as backslash-continued lines below
    the requirement, so dropping only the first line leaves orphaned `--hash=` lines
    and pip rejects the file -- taking down the step the skip exists to rescue."""

    LOCK_SHAPED = (
        "torchcodec==0.10.0 \\\n"
        "    --hash=sha256:aa \\\n"
        "    --hash=sha256:bb\n"
        "    # via -r extras-no-deps.txt\n"
        "transformers==5.5.0 \\\n"
        "    --hash=sha256:cc\n"
    )

    def test_dropping_an_entry_drops_its_hash_lines_too(self, tmp_path):
        source = tmp_path / "extras-no-deps.lock.txt"
        source.write_text(self.LOCK_SHAPED, encoding = "utf-8")
        filtered = ips._filter_requirements(source, {"torchcodec"})
        try:
            text = filtered.read_text(encoding = "utf-8")
        finally:
            filtered.unlink(missing_ok = True)
        assert "torchcodec" not in text
        assert "sha256:aa" not in text and "sha256:bb" not in text
        # The surviving entry keeps its own hash.
        assert "transformers==5.5.0" in text and "sha256:cc" in text

    def test_the_real_lock_survives_the_no_torch_skips(self):
        lock = LOCK_ROOT / "extras-no-deps.lock.txt"
        filtered = ips._filter_requirements(lock, ips.NO_TORCH_SKIP_PACKAGES)
        try:
            entries = _entries(filtered)
        finally:
            filtered.unlink(missing_ok = True)
        assert entries, "filtering emptied the lock"
        assert all(hashes for _, _, hashes in entries), (
            "an entry lost its hashes to the filter, so --require-hashes would reject "
            "the whole file"
        )
        names = {name for name, _, _ in entries}
        for skipped in ips.NO_TORCH_SKIP_PACKAGES:
            assert _normalise(skipped) not in names, skipped

    def test_a_plain_requirements_file_still_filters_as_before(self, tmp_path):
        source = tmp_path / "plain.txt"
        source.write_text("timm==1.0.28\neinops==0.8.2\n", encoding = "utf-8")
        filtered = ips._filter_requirements(source, {"timm"})
        try:
            text = filtered.read_text(encoding = "utf-8")
        finally:
            filtered.unlink(missing_ok = True)
        assert text == "einops==0.8.2\n"


# ── 8. the hardened-pip relaxation must not undo the lock ────────────────────


class TestHashRelaxationDoesNotDefeatTheLock:
    """_relaxed_pip_policy_env sets PIP_REQUIRE_HASHES=0 so a user pip.conf with
    `require-hashes = true` cannot fail the pip FALLBACK on the unlocked steps (#8530).
    A hashed install must not inherit it."""

    def test_a_hashed_pip_command_gets_no_hash_relaxation(self):
        cmd = [sys.executable, "-m", "pip", "install", "--require-hashes", "-r", "lock.txt"]
        assert ips._relaxed_pip_policy_env(cmd) == {}
        assert ips._install_env_for_cmd(cmd) is None  # inherit, nothing overridden

    def test_an_unlocked_pip_command_still_gets_it(self):
        cmd = [sys.executable, "-m", "pip", "install", "-r", "extras.txt"]
        assert ips._relaxed_pip_policy_env(cmd) == {"PIP_REQUIRE_HASHES": "0"}

    def test_a_uv_command_is_untouched_either_way(self):
        assert ips._relaxed_pip_policy_env(["uv", "pip", "install", "-r", "x"]) == {}


class TestPipFallbackKeepsTheLock:
    def test_the_pip_fallback_carries_require_hashes_and_the_lock(
        self, install_calls, monkeypatch
    ):
        """pip verifies a hashed requirements file natively, so a uv failure must not
        silently downgrade the step to an unverified install."""
        monkeypatch.setattr(ips, "USE_UV", False)
        recorded: list[list[str]] = []
        monkeypatch.setattr(ips, "run", lambda label, cmd, **kw: recorded.append(list(cmd)))
        ips.pip_install("studio deps", "--no-cache-dir", req = REQ_ROOT / "studio.txt")
        assert len(recorded) == 1
        cmd = recorded[0]
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"]
        assert "--require-hashes" in cmd
        assert cmd[cmd.index("-r") + 1].endswith("locks/studio.lock.txt")
        assert "-c" not in cmd

    def test_a_failing_locked_install_is_not_retried_unlocked(
        self, install_calls, monkeypatch
    ):
        """Otherwise anyone who can make the verified install fail gets the
        unverified one instead."""
        calls: list[list[str]] = []

        class _Failed:
            returncode = 1
            stdout = b"boom"

        monkeypatch.setattr(ips.subprocess, "run", lambda cmd, *a, **k: (
            calls.append(list(cmd)) or _Failed()
        ))
        recorded: list[list[str]] = []
        monkeypatch.setattr(ips, "run", lambda label, cmd, **kw: recorded.append(list(cmd)))
        ips.pip_install("studio deps", "--no-cache-dir", req = REQ_ROOT / "studio.txt")
        # uv failed, so pip runs -- with the lock and the flag, never the source file.
        assert len(recorded) == 1
        assert "--require-hashes" in recorded[0]
        assert recorded[0][recorded[0].index("-r") + 1].endswith("locks/studio.lock.txt")
