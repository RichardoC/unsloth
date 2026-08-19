# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The self-contained macOS arm64 runtime that ships inside Unsloth.app.

The .dmg is meant to be complete: copy Unsloth.app across and it runs, with no
first-run installation. That makes the app bundle a distribution channel for
executable code -- a CPython interpreter, ~260 Python distributions, llama.cpp,
whisper.cpp, Node -- assembled at build time by scripts/build_macos_runtime.sh and
signed on the way out. Four things can rot silently in that arrangement, and this
file is the guard on each:

  1. THE LOCK. studio/backend/requirements/locks/darwin-arm64-bundle.lock.txt is the
     one resolution the bundle installs for the parts scripts/gen_python_locks.sh
     deliberately leaves unlocked. It is not universal, so
     tests/security/test_python_locks.py excludes it (see PLATFORM_LOCKS there); the
     properties that matter are asserted here instead, and then some: exactly one
     index, because a second one is a dependency-confusion surface nobody would
     notice in a 4000-line diff.

  2. THE TRUST ANCHORS. Every archive that is not a wheel -- CPython, llama.cpp,
     whisper.cpp, Node -- must be verified against a digest committed in this tree
     before it is extracted, with no environment variable able to turn that off. A
     bundled runtime has no user to fall back for.

  3. THE LAYOUT CONTRACT. studio/src-tauri resolves the bundled runtime through fixed
     paths. They live in studio/macos_runtime_pins.json as data, and the build script,
     the injection script and the dev-build workflow must all still agree with it.

  4. THE PRUNE LIST. Pruning is where a size win quietly becomes a runtime failure on
     somebody else's Mac. The list is closed: every entry carries a reason, no entry
     may match anything the contract requires, and the build script may not delete
     anything the list does not name.

Nothing here builds a payload or touches the network (the suite's autouse blocker
would refuse anyway) -- these are assertions about committed files. The claim that
the payload actually WORKS is a different claim and belongs on a Mac: the
desktop-dev-build workflow mounts the .dmg and runs the bundled interpreter, and the
last section of this file asserts that it still does.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a suite dependency
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "studio"
REQ_ROOT = STUDIO_DIR / "backend" / "requirements"
LOCK_ROOT = REQ_ROOT / "locks"

PINS_JSON = STUDIO_DIR / "macos_runtime_pins.json"
BUNDLE_LOCK = LOCK_ROOT / "darwin-arm64-bundle.lock.txt"
LOCK_GENERATOR = REPO_ROOT / "scripts" / "gen_macos_bundle_lock.sh"
UNIVERSAL_GENERATOR = REPO_ROOT / "scripts" / "gen_python_locks.sh"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_macos_runtime.sh"
INJECT_SCRIPT = REPO_ROOT / "scripts" / "inject_macos_runtime.sh"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_macos_prebuilts.py"
DEV_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "desktop-dev-build.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# A requirement line starts at column zero; uv indents hashes and annotations.
_REQ_LINE = re.compile(r"^[A-Za-z0-9]")
_RANGE_OPS = ("<", ">", "!=", "~=", "===")


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _entries(lock: Path) -> list[tuple[str, str, list[str]]]:
    """(normalised name, spec, hashes) for every requirement in a lock."""
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


@pytest.fixture(scope = "module")
def pins() -> dict:
    assert PINS_JSON.is_file(), f"{PINS_JSON} is missing"
    return json.loads(PINS_JSON.read_text(encoding = "utf-8"))


@pytest.fixture(scope = "module")
def lock_text() -> str:
    assert BUNDLE_LOCK.is_file(), (
        f"{BUNDLE_LOCK} is missing. Run: bash scripts/gen_macos_bundle_lock.sh"
    )
    return BUNDLE_LOCK.read_text(encoding = "utf-8")


@pytest.fixture(scope = "module")
def dev_workflow() -> dict:
    assert yaml is not None, "PyYAML is required"
    assert DEV_WORKFLOW.is_file(), f"{DEV_WORKFLOW} is missing"
    return yaml.safe_load(DEV_WORKFLOW.read_text(encoding = "utf-8"))


def _dev_steps(workflow: dict) -> list[dict]:
    steps: list[dict] = []
    for job in (workflow.get("jobs") or {}).values():
        steps.extend(job.get("steps") or [])
    return steps


# ── 1. the bundle lock ───────────────────────────────────────────────────────


class TestBundleLock:
    def test_every_requirement_is_pinned_with_a_double_equals(self, lock_text):
        """--require-hashes refuses anything else outright, so an unpinned line does
        not make the payload weaker, it makes the build fail. Asserting it here means
        the failure is a red test rather than a red CI job twenty minutes in.
        """
        offenders = [
            spec
            for _, spec, _ in _entries(BUNDLE_LOCK)
            if "==" not in spec.split(";", 1)[0]
        ]
        assert not offenders, f"requirements that are not exact pins: {offenders}"

    def test_every_requirement_carries_at_least_one_sha256(self):
        offenders = [spec for _, spec, hashes in _entries(BUNDLE_LOCK) if not hashes]
        assert not offenders, (
            f"requirements with no --hash: {offenders}. Every digest in this file must "
            f"come from uv; a hand-written hash either fails every build or asserts "
            f"bytes nobody verified."
        )

    def test_every_hash_is_a_well_formed_sha256(self):
        offenders = []
        for _, spec, hashes in _entries(BUNDLE_LOCK):
            for token in hashes:
                algorithm, _, digest = token[len("--hash="):].partition(":")
                if algorithm != "sha256" or not SHA256_RE.match(digest):
                    offenders.append((spec, token))
        assert not offenders, f"malformed hashes: {offenders}"

    def test_no_requirement_carries_a_range_specifier(self, lock_text):
        offenders = []
        for _, spec, _ in _entries(BUNDLE_LOCK):
            head = spec.split(";", 1)[0]
            if any(operator in head for operator in _RANGE_OPS):
                offenders.append(spec)
        assert not offenders, f"range specifiers in a lock: {offenders}"

    def test_no_requirement_is_a_url_or_vcs_entry(self, lock_text):
        """A URL or git requirement cannot carry an index hash, so one here would make
        the whole file unusable under --require-hashes. It is also why the diffusers
        pin is installed in its own step rather than locked.
        """
        offenders = [
            spec
            for _, spec, _ in _entries(BUNDLE_LOCK)
            if " @ " in spec or "://" in spec
        ]
        assert not offenders, f"URL/VCS requirements in a lock: {offenders}"

    def test_the_header_records_how_to_regenerate_it_and_what_it_is_for(self, lock_text):
        for needle in (
            "GENERATED FILE -- DO NOT EDIT",
            "scripts/build_macos_runtime.sh",
            "scripts/gen_macos_bundle_lock.sh",
            "NOT universal",
            "--require-hashes",
        ):
            assert needle in lock_text, f"the lock header no longer says {needle!r}"

    def test_the_header_records_the_one_platform_it_resolves_for(self, lock_text, pins):
        """These three header lines are not documentation: build_macos_runtime.sh reads
        them and refuses to install a lock resolved for a different platform, Python or
        deployment target. Installing the wrong one would silently select the wrong
        wheels and surface as an ImportError on a user's Mac.
        """
        found = {
            key: value
            for key, value in (
                (
                    line.split(":", 1)[0].strip("# ").strip(),
                    line.split(":", 1)[1].strip(),
                )
                for line in lock_text.splitlines()
                if line.startswith("# unsloth-lock-")
            )
        }
        assert found.get("unsloth-lock-python-platform") == pins["target"]["uv_python_platform"]
        assert found.get("unsloth-lock-python-version") == pins["components"]["cpython"]["python_minor"]
        assert (
            found.get("unsloth-lock-macos-deployment-target")
            == pins["target"]["macos_deployment_target"]
        )

    def test_it_was_generated_by_the_uv_the_installer_pins(self, lock_text):
        pinned = re.search(
            r'^UV_PINNED_VERSION="([^"]+)"',
            (REPO_ROOT / "install.sh").read_text(encoding = "utf-8"),
            re.MULTILINE,
        )
        assert pinned is not None, "could not read UV_PINNED_VERSION out of install.sh"
        assert f"generated by: uv {pinned.group(1)}" in lock_text, (
            f"the lock records a different uv than install.sh pins ({pinned.group(1)}). "
            f"A lock generated by a resolver no user runs is a lock for an install "
            f"nobody performs."
        )

    def test_it_resolves_from_exactly_one_index(self, lock_text):
        """The reason --emit-index-annotation is on.

        install.sh installs torch with `--default-index download.pytorch.org/whl/cpu`,
        whose macOS arm64 torch wheel has a different sha256 from PyPI's -- but not
        different code: measured on torch 2.10.0 cp313 macosx_11_0_arm64 the two
        wheels agree on all 12338 members except dist-info/METADATA and RECORD, and
        the METADATA delta is only the Linux-x86_64-only CUDA Requires-Dist lines. So
        this lock uses PyPI alone, and that is worth asserting: with a second index in
        play uv either fails outright (its stale numpy cannot satisfy scikit-learn) or,
        under --index-strategy unsafe-best-match, quietly sources jinja2 and markupsafe
        from download.pytorch.org as well. One index, no confusion surface.
        """
        indexes = set(re.findall(r"^\s*#\s*from (\S+)$", lock_text, re.MULTILINE))
        assert indexes == {"https://pypi.org/simple"}, (
            f"the bundle lock resolves from {sorted(indexes)}; it must resolve from "
            f"PyPI alone. If a second index is genuinely needed, say why here first."
        )
        # And every requirement must actually carry the annotation, or the check above
        # could pass on a file where uv stopped emitting them.
        annotated = len(re.findall(r"^\s*#\s*from \S+$", lock_text, re.MULTILINE))
        requirements = len(_entries(BUNDLE_LOCK))
        assert annotated == requirements, (
            f"{requirements} requirements but {annotated} index annotations; "
            f"--emit-index-annotation must stay on so this file is auditable."
        )

    def test_it_covers_the_steps_gen_python_locks_cannot(self):
        """The bundle lock exists to close exactly the gap the universal generator
        documents. If one of these stops being present the gap is open again and the
        payload is installing something unhashed.
        """
        names = {name for name, _, _ in _entries(BUNDLE_LOCK)}
        for required in (
            "torch",
            "torchvision",
            "torchaudio",
            "torchao",
            "unsloth",
            "unsloth-zoo",
            "mlx",
            "mlx-metal",
            "mlx-lm",
            "mlx-vlm",
        ):
            assert _normalise(required) in names, f"the bundle lock does not pin {required}"

    def test_the_carve_out_is_absorbed_on_purpose(self):
        """gen_python_locks.sh carves pytorch_tokenizers out of its universal lock
        because the axis that decides the version (musl vs glibc, the macOS deployment
        target a wheel was built against) has no PEP 508 marker. This lock is one
        platform at one deployment target, where that axis is a constant -- so the cap
        resolves to one version and gets hashed like everything else, and the build
        never installs anything unhashed for it. That is the whole reason it may
        appear here and not there.
        """
        pins_in_lock = {
            name: spec for name, spec, _ in _entries(BUNDLE_LOCK)
        }
        assert "pytorch-tokenizers" in pins_in_lock, (
            "pytorch_tokenizers is missing from the bundle lock, so the build would "
            "have to install extras-no-deps.unlocked.txt without hashes"
        )

    def test_it_does_not_contradict_the_universal_locks(self):
        """The bundle installs this lock and the six universal ones into ONE
        site-packages directory. Two locks naming different versions of the same
        distribution would mean whichever installs last wins, silently.
        """
        mine = {name: spec.split(";", 1)[0] for name, spec, _ in _entries(BUNDLE_LOCK)}
        conflicts: dict[str, list[str]] = {}
        for other in sorted(LOCK_ROOT.glob("*.lock.txt")):
            if other.name == BUNDLE_LOCK.name:
                continue
            for name, spec, _ in _entries(other):
                head = spec.split(";", 1)[0]
                # Marked entries describe one fork of a universal resolution and say
                # nothing about this platform, so only unconditional pins compare.
                if ";" in spec or name not in mine or head == mine[name]:
                    continue
                conflicts.setdefault(name, []).append(f"{other.name}: {head} vs {mine[name]}")
        assert not conflicts, f"locks disagree about a version: {conflicts}"

    def test_it_does_not_contradict_a_version_pinned_elsewhere(self):
        """Same property against the shipped requirements files, which are also applied
        into that one directory.
        """
        mine = {}
        for name, spec, _ in _entries(BUNDLE_LOCK):
            head = spec.split(";", 1)[0]
            if "==" in head:
                mine[name] = head.split("==", 1)[1]
        conflicts: dict[str, str] = {}
        for req in (
            REQ_ROOT / "studio.txt",
            REQ_ROOT / "extras.txt",
            REQ_ROOT / "extras-no-deps.txt",
            REQ_ROOT / "no-torch-runtime.txt",
            REQ_ROOT / "single-env" / "constraints.txt",
            REQ_ROOT / "single-env" / "data-designer-deps.txt",
            REQ_ROOT / "single-env" / "data-designer.txt",
        ):
            for raw in req.read_text(encoding = "utf-8").splitlines():
                spec = raw.split("#", 1)[0].strip()
                if not spec or spec.startswith("-") or ";" in spec:
                    continue
                match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s,]+)", spec)
                if not match:
                    continue
                name = _normalise(match.group(1))
                if name in mine and mine[name] != match.group(2):
                    conflicts[name] = f"{req.name} pins {match.group(2)}, lock pins {mine[name]}"
        assert not conflicts, f"the bundle lock contradicts a pin in the tree: {conflicts}"

    def test_it_is_a_real_closure_and_not_a_stub(self):
        entries = _entries(BUNDLE_LOCK)
        hashes = sum(len(h) for _, _, h in entries)
        assert len(entries) >= 100, (
            f"only {len(entries)} requirements; the torch + extras + unsloth + MLX "
            f"closure cannot be that small, so the lock was truncated"
        )
        assert hashes >= 2 * len(entries), f"only {hashes} hashes for {len(entries)} requirements"


# ── 2. the lock generator ────────────────────────────────────────────────────


class TestLockGenerator:
    def test_it_exists_and_is_syntactically_valid(self):
        assert LOCK_GENERATOR.is_file(), f"{LOCK_GENERATOR} is missing"
        result = subprocess.run(
            ["bash", "-n", str(LOCK_GENERATOR)], capture_output = True, text = True
        )
        assert result.returncode == 0, result.stderr

    def test_it_reads_its_uv_pin_out_of_install_sh(self):
        text = LOCK_GENERATOR.read_text(encoding = "utf-8")
        assert "UV_PINNED_VERSION" in text and "install.sh" in text
        assert re.search(r"UV_ACTUAL.*!=.*UV_PINNED_VERSION", text) or (
            '"$UV_ACTUAL" != "$UV_PINNED_VERSION"' in text
        ), "the generator must refuse a uv that is not the pinned one"

    def test_it_freezes_the_index_at_the_same_instant_as_the_universal_generator(self):
        """The bundle applies both lock sets into one directory. Two cutoffs would put
        two different days' worth of upstream in one environment.
        """
        def cutoff(path: Path) -> str | None:
            match = re.search(
                r'^EXCLUDE_NEWER="([^"]+)"', path.read_text(encoding = "utf-8"), re.MULTILINE
            )
            return match.group(1) if match else None

        mine, theirs = cutoff(LOCK_GENERATOR), cutoff(UNIVERSAL_GENERATOR)
        assert mine, "the bundle lock generator does not pin --exclude-newer"
        assert mine == theirs, (
            f"the two lock generators freeze the index at different instants "
            f"({mine} vs {theirs}); move them together or the bundle mixes two days."
        )

    def test_it_asks_uv_for_hashes_and_for_one_platform(self):
        text = LOCK_GENERATOR.read_text(encoding = "utf-8")
        for flag in (
            "--generate-hashes",
            "--python-platform",
            "--python-version",
            "--emit-index-annotation",
            "--only-binary :all:",
        ):
            assert flag in text, f"the generator no longer passes {flag}"
        assert "--universal" not in text, (
            "a universal resolution is the one thing this lock must not be"
        )

    def test_it_exports_the_deployment_target_rather_than_prefixing_the_pipeline(self):
        """`VAR=x a | b` sets VAR only for `a`, and it is uv (the right-hand side) that
        has to see MACOSX_DEPLOYMENT_TARGET. Getting this wrong is silent: uv falls back
        to a lower default, bitsandbytes has no installable macOS arm64 wheel there, and
        the whole resolution walks backwards to a 2024 unsloth.
        """
        text = LOCK_GENERATOR.read_text(encoding = "utf-8")
        assert re.search(r"^export MACOSX_DEPLOYMENT_TARGET=", text, re.MULTILINE), (
            "MACOSX_DEPLOYMENT_TARGET must be exported, not prefixed onto a pipeline"
        )
        assert not re.search(r"MACOSX_DEPLOYMENT_TARGET=\S+ compile_input", text)

    def test_it_does_not_feed_uv_the_files_that_cannot_be_locked(self):
        text = LOCK_GENERATOR.read_text(encoding = "utf-8")
        input_block = text.split("compile_input()", 1)[-1].split("DISPLAY_SOURCE", 1)[0]
        for unlockable in ("diffusers-pin.txt", "triton-kernels.txt"):
            assert unlockable not in input_block, (
                f"{unlockable} is in the compile input; it is a URL/VCS requirement and "
                f"cannot carry an index hash"
            )

    def test_it_applies_the_macos_override_so_the_mlx_stack_resolves(self):
        """Without overrides-darwin-arm64.txt, mlx-vlm's own transformers floor fights
        constraints.txt's transformers==5.5.0 and the resolver drags the whole MLX
        stack backwards (measured: mlx-vlm 0.6.15 -> 0.3.9).
        """
        text = LOCK_GENERATOR.read_text(encoding = "utf-8")
        assert "overrides-darwin-arm64.txt" in text
        assert "--override" in text

    def test_the_universal_generators_check_mode_knows_this_lock_is_not_its_own(self):
        """gen_python_locks.sh --check diffs the whole locks/ directory against a fresh
        regeneration, so a lock it does not produce reads as "the committed locks are
        stale" and reds a lane on a day nobody touched requirements. Its allowlist must
        name this lock, and must stay an allowlist rather than becoming a wildcard --
        any OTHER unexpected file in locks/ still has to fail that diff.
        """
        text = UNIVERSAL_GENERATOR.read_text(encoding = "utf-8")
        assert "FOREIGN_LOCKS=(" in text, (
            "gen_python_locks.sh --check has no exclusion for locks it does not generate"
        )
        listed = set(text.split("FOREIGN_LOCKS=(", 1)[1].split(")", 1)[0].split())
        assert listed == {BUNDLE_LOCK.name}, (
            f"gen_python_locks.sh excludes {sorted(listed)} from its freshness diff; it "
            f"must exclude exactly {BUNDLE_LOCK.name}"
        )
        assert "--exclude=$foreign" in text and "diff -ru" in text
        # The exclusion list and test_python_locks.py's PLATFORM_LOCKS describe the same
        # set from two directions; a divergence means one of them is stale.
        other = (Path(__file__).parent / "test_python_locks.py").read_text(encoding = "utf-8")
        platform_locks = set(
            re.findall(r'"([^"]+\.lock\.txt)"', other.split("PLATFORM_LOCKS", 1)[1].split("\n\n", 1)[0])
        )
        assert platform_locks == listed, (
            f"test_python_locks.PLATFORM_LOCKS is {sorted(platform_locks)} but "
            f"gen_python_locks.sh excludes {sorted(listed)}"
        )

    def test_it_supports_check_mode_without_mutating_the_tree(self):
        text = LOCK_GENERATOR.read_text(encoding = "utf-8")
        assert "--check" in text and "mktemp -d" in text, (
            "check mode must write to a temp directory and diff, never mutate the tree"
        )


# ── 3. the pins file and its trust anchors ───────────────────────────────────


class TestPins:
    def test_the_schema_is_the_one_the_scripts_expect(self, pins):
        assert pins["schema_version"] == 1
        for key in ("components", "layout", "prune", "target"):
            assert key in pins, f"macos_runtime_pins.json is missing {key!r}"

    def test_the_cpython_pin_is_a_complete_trust_anchor(self, pins):
        entry = pins["components"]["cpython"]
        for field in ("repo", "release", "python_version", "python_minor", "asset", "sha256"):
            assert isinstance(entry.get(field), str) and entry[field].strip(), field
        assert SHA256_RE.match(entry["sha256"]), entry["sha256"]
        assert entry["python_version"].startswith(entry["python_minor"] + "."), (
            f"{entry['python_version']} is not a patch of {entry['python_minor']}"
        )
        assert entry["python_minor"] in entry["asset"], (
            "the asset name must name the version it claims to be"
        )
        assert "aarch64-apple-darwin" in entry["asset"], (
            "the bundled interpreter must be the macOS arm64 build"
        )

    def test_the_bundled_python_is_not_the_patch_install_sh_skips(self, pins):
        """install.sh's PYTHON_SKIP exists because python/cpython#139783 makes 3.13.8
        break `import torch`. Shipping that patch inside the app would ship a Studio
        whose Train and Export are dead on arrival.
        """
        skip = re.search(
            r'^PYTHON_SKIP="([^"]+)"',
            (REPO_ROOT / "install.sh").read_text(encoding = "utf-8"),
            re.MULTILINE,
        )
        assert skip is not None, "could not read PYTHON_SKIP out of install.sh"
        skipped = {part.strip() for part in skip.group(1).split() if part.strip()}
        assert pins["components"]["cpython"]["python_version"] not in skipped, (
            f"the bundle pins CPython {pins['components']['cpython']['python_version']}, "
            f"which install.sh refuses to install for users"
        )

    def test_the_bundled_python_minor_matches_what_install_sh_pins_on_apple_silicon(self, pins):
        """Not a style rule. A different minor in the bundle than on a user's own
        install means two different wheel sets, and the macOS arm64 wheel coverage is
        not the same across minors -- extras.txt already routes darwin + >=3.14 to a
        MeCab release that ships only an sdist.
        """
        install_sh = (REPO_ROOT / "install.sh").read_text(encoding = "utf-8")
        # The Apple Silicon branch: `else PYTHON_VERSION="3.13"` after the Intel case.
        candidates = re.findall(r'^\s*PYTHON_VERSION="(\d+\.\d+)"', install_sh, re.MULTILINE)
        assert candidates, "could not read PYTHON_VERSION out of install.sh"
        assert pins["components"]["cpython"]["python_minor"] in candidates, (
            f"the bundle pins Python {pins['components']['cpython']['python_minor']}, "
            f"which install.sh never selects (it selects {sorted(set(candidates))})"
        )

    def test_the_deployment_target_is_recorded_and_explained(self, pins):
        target = pins["target"]["macos_deployment_target"]
        assert re.fullmatch(r"\d+\.\d+", target), target
        assert float(target) >= 14.0, (
            "below macOS 14 there is no installable bitsandbytes for macOS arm64 (every "
            "wheel from 0.49.0 on is macosx_14_0), and the resolver silently backtracks "
            "unsloth to a 2024 release. Lowering this requires re-reading that comment."
        )
        assert "bitsandbytes" in pins["target"]["comment"], (
            "the reason this number is what it is must travel with it"
        )

    def test_the_other_components_reuse_the_trust_anchors_already_in_the_tree(self):
        """No second copy of a digest that already exists in this repo: llama.cpp and
        whisper.cpp go through prebuilt_release_pins.json, Node through
        node_prebuilt_pins.json. A duplicated digest is a digest that will drift.
        """
        pins = json.loads(PINS_JSON.read_text(encoding = "utf-8"))
        assert set(pins["components"]) == {"cpython"}, (
            f"macos_runtime_pins.json carries pins for {sorted(pins['components'])}. Only "
            f"CPython belongs here; llama.cpp and whisper.cpp are anchored by "
            f"prebuilt_release_pins.json and Node by node_prebuilt_pins.json, and a "
            f"second copy of a digest is a digest that will drift."
        )
        digests = re.findall(r"\b[0-9a-f]{64}\b", PINS_JSON.read_text(encoding = "utf-8"))
        assert digests == [pins["components"]["cpython"]["sha256"]], (
            f"macos_runtime_pins.json holds {len(digests)} sha256 values; it may hold "
            f"exactly one, CPython's"
        )
        fetch = FETCH_SCRIPT.read_text(encoding = "utf-8")
        assert "prebuilt_release_pins.json" in fetch or "RELEASE_PINS_FILENAME" in fetch
        assert "node_prebuilt_pins.json" in fetch


# ── 4. the layout contract ───────────────────────────────────────────────────


class TestLayoutContract:
    def test_the_contract_names_every_directory_the_rust_side_reads(self, pins):
        assert set(pins["layout"]["required_dirs"]) == {
            "python",
            "site-packages",
            "llama.cpp",
            "whisper.cpp",
            "node",
            "oxc-node-modules",
        }, (
            "the payload layout is an interface studio/src-tauri resolves. Changing it "
            "is changing that interface, so change it deliberately and tell the Rust side."
        )
        assert pins["layout"]["root"] == "Contents/Resources/runtime"

    def test_the_contract_names_the_entry_points_the_app_invokes(self, pins):
        required = set(pins["layout"]["required_files"])
        for path in (
            "BUNDLE_MANIFEST.json",
            "python/bin/python3",
            "llama.cpp/build/bin/llama-server",
            "whisper.cpp/build/bin/whisper-server",
            "node/bin/node",
        ):
            assert path in required, f"{path} is not in required_files"

    def test_the_contract_covers_what_the_rust_health_check_refuses_to_start_without(self, pins):
        """bundled_runtime.rs::health() names two site-packages files explicitly. A
        payload missing either would build clean, ship, and then refuse to launch, so
        the build asserts them too rather than leaving the .dmg to find out.
        """
        required = set(pins["layout"]["required_files"])
        for path in (
            "site-packages/unsloth_cli/__init__.py",
            "site-packages/studio/backend/run.py",
        ):
            assert path in required, f"{path} is not in required_files"
        health = STUDIO_DIR / "src-tauri" / "src" / "bundled_runtime.rs"
        if health.is_file():
            text = health.read_text(encoding = "utf-8")
            for needle in ('join("unsloth_cli")', 'join("run.py")'):
                assert needle in text, (
                    f"bundled_runtime.rs no longer checks for {needle}; the contract in "
                    f"macos_runtime_pins.json should follow it"
                )

    def test_the_ggml_prebuilts_keep_the_layout_the_installer_uses(self, pins):
        """`build/bin/<server>` is not an arbitrary choice: it is what
        prebuilt_core.assemble_install_tree and install_whisper_prebuilt.runtime_bin_dir
        produce under ~/.unsloth, so the app finds the bundled copy through paths it
        already knows.
        """
        for path in pins["layout"]["required_files"]:
            if path.startswith(("llama.cpp/", "whisper.cpp/")):
                assert "/build/bin/" in path, path
        whisper = STUDIO_DIR / "install_whisper_prebuilt.py"
        assert '"build" / "bin"' in whisper.read_text(encoding = "utf-8"), (
            "install_whisper_prebuilt.py no longer uses build/bin; the bundle layout "
            "must follow it rather than diverge"
        )

    def test_the_build_script_asserts_the_contract_rather_than_assuming_it(self):
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        assert "required_dirs" in text and "required_files" in text
        assert "required_executables" in text
        assert "violates the layout contract" in text, (
            "the build script must fail on a payload that does not satisfy the contract; "
            "a .dmg is the wrong place to discover a missing path"
        )

    def test_the_injection_script_re_asserts_it_against_the_app_bundle(self):
        text = INJECT_SCRIPT.read_text(encoding = "utf-8")
        for path in (
            "runtime/BUNDLE_MANIFEST.json",
            "runtime/python/bin/python3",
            "runtime/llama.cpp/build/bin/llama-server",
            "runtime/whisper.cpp/build/bin/whisper-server",
            "runtime/node/bin/node",
            "runtime/oxc-node-modules",
        ):
            assert path in text, f"the injection script does not check for {path}"

    def test_the_injection_script_refuses_to_break_an_existing_signature(self):
        text = INJECT_SCRIPT.read_text(encoding = "utf-8")
        assert "codesign -dv" in text and "already signed" in text, (
            "injecting into a signed bundle invalidates its signature; the script must "
            "refuse rather than do it quietly"
        )

    def test_the_manifest_is_part_of_the_contract(self, pins):
        assert "BUNDLE_MANIFEST.json" in pins["layout"]["required_files"]
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        for field in (
            "generator_version",
            "sizes_bytes",
            "distributions",
            "locks",
            "pruned",
            "incomplete",
        ):
            assert field in text, f"BUNDLE_MANIFEST.json no longer records {field}"


# ── 5. the prune list ────────────────────────────────────────────────────────


class TestPruneList:
    def test_every_entry_carries_a_kind_a_path_and_a_real_reason(self, pins):
        entries = pins["prune"]["entries"]
        assert entries, "the prune list is empty"
        for entry in entries:
            assert entry.get("kind") in {"dir", "glob", "dir-name"}, entry
            assert isinstance(entry.get("path"), str) and entry["path"].strip(), entry
            assert isinstance(entry.get("optional"), bool), entry
            reason = entry.get("reason", "")
            assert isinstance(reason, str) and len(reason) >= 60, (
                f"prune entry {entry['path']!r} has no substantive reason. Deleting "
                f"something from a shipped runtime needs an argument, not a note."
            )

    def test_no_entry_can_remove_anything_the_contract_requires(self, pins):
        """The guard that keeps the list from growing into something the app needs. The
        build script enforces the same rule at build time; this catches it in review.
        """
        required = set(pins["layout"]["required_dirs"]) | set(pins["layout"]["required_files"])
        for entry in pins["prune"]["entries"]:
            path = entry["path"]
            if entry["kind"] == "dir-name":
                assert path not in required, entry
                assert not any(
                    part == path for keep in required for part in keep.split("/")
                ), f"prune entry {path!r} would match a path component of a required path"
                continue
            for keep in required:
                assert keep != path, f"prune entry {path!r} is a required path"
                assert not keep.startswith(path.rstrip("/") + "/"), (
                    f"prune entry {path!r} contains the required path {keep!r}"
                )

    def test_no_entry_touches_a_top_level_component_wholesale(self, pins):
        """A prune entry naming a component root would delete a whole half of the
        payload. Every entry must be a path INSIDE one.
        """
        roots = set(pins["layout"]["required_dirs"])
        for entry in pins["prune"]["entries"]:
            if entry["kind"] == "dir-name":
                continue
            head = entry["path"].split("/", 1)[0]
            assert entry["path"] not in roots, entry
            assert head in roots or head == "site-packages", (
                f"prune entry {entry['path']!r} is not inside a known component"
            )

    def test_the_build_script_prunes_only_what_the_list_names(self):
        """No hardcoded deletions. The one exception is loose *.pyc files, which are
        the same category as __pycache__ and are named in the script right beside it.
        """
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        # Deletions of a path INSIDE the staging tree. `rm -rf "$STAGE"` itself is the
        # staging reset that makes the build re-runnable, not a prune.
        offenders = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"\brm -rf\b.*\$\{?(STAGE|SITE)\}?/", line)
        ]
        assert not offenders, (
            f"the build script deletes from the payload outside the prune list: "
            f"{offenders}. Add an entry with a reason to studio/macos_runtime_pins.json."
        )
        assert 'pins["prune"]["entries"]' in text, (
            "the pruning step must be driven by the committed list"
        )
        assert "*.pyc" in text, "loose .pyc files should still be swept"

    def test_the_duplicate_frontend_is_pruned_and_the_reason_is_the_api_only_invariant(self, pins):
        """The single biggest prune (108 MiB) rests on the desktop always launching the
        backend with --api-only, because backend/run.py raises SystemExit when it is
        asked to serve a UI it cannot find. If that invariant ever changes, this prune
        becomes a crash on launch, so the reason must name it and the Rust side must
        still hold it.
        """
        entry = next(
            (item for item in pins["prune"]["entries"] if item["path"] == "site-packages/studio/frontend"),
            None,
        )
        assert entry is not None, "the wheel-embedded duplicate frontend is no longer pruned"
        assert "api-only" in entry["reason"], entry["reason"]
        process_rs = STUDIO_DIR / "src-tauri" / "src" / "process.rs"
        if process_rs.is_file():
            assert "--api-only" in process_rs.read_text(encoding = "utf-8"), (
                "studio/src-tauri no longer passes --api-only, so pruning "
                "site-packages/studio/frontend would break the bundled backend"
            )

    def test_node_headers_are_pruned_and_python_headers_are_not(self, pins):
        paths = {entry["path"] for entry in pins["prune"]["entries"]}
        assert "node/include" in paths, (
            "Node's 65 MiB of addon headers are dead weight in a bundle that compiles "
            "no addons"
        )
        assert not any(path.startswith("python/include") for path in paths), (
            "python/include must stay: torch.utils.cpp_extension reads it, and 2.4 MiB "
            "is not worth that class of bug"
        )

    def test_per_package_tests_and_docs_are_not_pruned(self, pins):
        """Deliberate, and recorded as such. A package that imports its own test
        helpers or reads its own data files fails at runtime on somebody else's Mac,
        which is a far worse trade than the megabytes.
        """
        for entry in pins["prune"]["entries"]:
            path = entry["path"]
            assert not re.search(r"site-packages/.*/(tests?|docs?)\b", path), (
                f"prune entry {path!r} removes a package's own tests/docs"
            )
        assert "test suites and docs inside site-packages" in pins["prune"]["comment"]


# ── 6. the build and fetch scripts ───────────────────────────────────────────


class TestBuildScripts:
    @pytest.mark.parametrize("script", [BUILD_SCRIPT, INJECT_SCRIPT, LOCK_GENERATOR])
    def test_the_shell_scripts_are_syntactically_valid(self, script):
        assert script.is_file(), f"{script} is missing"
        result = subprocess.run(["bash", "-n", str(script)], capture_output = True, text = True)
        assert result.returncode == 0, result.stderr

    def test_the_fetch_script_compiles(self):
        # compile(), not py_compile: this must not leave a __pycache__ entry behind in
        # scripts/ as a side effect of running the suite.
        assert FETCH_SCRIPT.is_file(), f"{FETCH_SCRIPT} is missing"
        compile(FETCH_SCRIPT.read_text(encoding = "utf-8"), str(FETCH_SCRIPT), "exec")

    def test_the_build_script_installs_every_lock_with_require_hashes(self):
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        # The install loop is the LAST iteration over LOCK_STEPS (the first is the
        # preflight that checks each lock exists), and it must carry the flag.
        install_block = text.rsplit('for lock in "${LOCK_STEPS[@]}"', 1)[-1].split("done", 1)[0]
        assert "pip install" in install_block, install_block
        assert "--require-hashes" in install_block, (
            "the lock install loop must pass --require-hashes; without it the payload "
            "installs whatever bytes the index hands back"
        )
        # And no uv pip install anywhere may reach the index without hashes: the only
        # unhashed installs are the two local plugin directories and the diffusers
        # archive, all of which are local files by then.
        for block in text.split("$UV_BIN\" pip install")[1:]:
            head = block.split("|| die", 1)[0]
            if "--require-hashes" in head:
                continue
            assert "$plugin" in head or "file://$ARCHIVE" in head, (
                f"an unhashed uv install that is not a local path: {head.strip()[:200]}"
            )

    def test_the_build_script_installs_every_lock_this_repo_ships(self):
        """A lock that exists but is never installed is a lock that is not in the
        bundle, and the missing distributions would only show up as an ImportError.
        """
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        steps = text.split("LOCK_STEPS=(", 1)[1].split(")", 1)[0]
        named = {line.strip() for line in steps.splitlines() if line.strip()}
        on_disk = {path.name for path in LOCK_ROOT.glob("*.lock.txt")}
        assert named == on_disk, (
            f"the build script installs {sorted(named)} but the tree ships "
            f"{sorted(on_disk)}. Every lock must be either installed or removed."
        )

    def test_the_build_script_builds_no_venv(self):
        """A venv bakes absolute paths into pyvenv.cfg and into every console-script
        shebang, and a relocated venv is a broken one. --target plus PYTHONPATH is the
        whole reason the payload survives being copied to /Applications.
        """
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        assert "uv venv" not in text and "-m venv" not in text
        assert "--target" in text

    def test_the_build_script_cross_builds_explicitly(self):
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        for flag in ("--python-platform", "--python-version", "--only-binary :all:"):
            assert flag in text, f"the build script no longer passes {flag}"
        assert re.search(r"^export MACOSX_DEPLOYMENT_TARGET=", text, re.MULTILINE), (
            "MACOSX_DEPLOYMENT_TARGET decides which wheels exist and must be exported "
            "explicitly, not inherited from the build machine"
        )

    def test_the_build_script_refuses_a_uv_that_is_not_the_pinned_one(self):
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        assert "UV_PINNED_VERSION" in text
        assert '"$UV_ACTUAL" != "$UV_PINNED_VERSION"' in text

    def test_the_build_script_verifies_the_lock_matches_the_pins(self):
        """Installing a lock resolved for another platform selects the wrong wheels and
        surfaces as an ImportError on a user's Mac, so the header is compared, not
        trusted.
        """
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        assert "unsloth-lock-python-platform" in text
        assert "unsloth-lock-macos-deployment-target" in text
        assert "regenerate it" in text

    def test_the_build_script_makes_console_scripts_relocatable(self):
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        assert "shebang" in text.lower()
        assert "#!/bin/sh" in text and "realpath" in text, (
            "console scripts must resolve their interpreter relative to themselves, the "
            "way python-build-standalone's own bin/pip already does"
        )

    def test_the_build_script_prefetches_the_oxc_bindings_for_the_right_platform(self):
        text = BUILD_SCRIPT.read_text(encoding = "utf-8")
        assert "--os=darwin" in text and "--cpu=arm64" in text, (
            "oxc-parser's parser is a per-platform native binding; without --os/--cpu a "
            "Linux runner installs a Linux binding and the validator fails on every Mac"
        )
        assert "npm ci" in text and "--ignore-scripts" in text
        assert "binding-darwin-arm64" in text, (
            "the build must fail closed when the darwin-arm64 binding is absent"
        )

    def test_the_fetch_script_fails_closed_and_cannot_be_switched_off(self):
        """prebuilt_core's ALLOW_LATEST / ALLOW_UNVERIFIED environment variables turn
        digest verification into a no-op at install time, where there is a user to fall
        back for. A build that ships signed inside an app has none, so the expected
        digest is passed explicitly rather than looked up through that path.
        """
        text = FETCH_SCRIPT.read_text(encoding = "utf-8")
        assert "verify_pinned_checksum_index" in text
        assert re.search(r"expected\s*=\s*entry\[.checksum_index_sha256.\]", text), (
            "the expected digest must be passed explicitly so no environment variable "
            "can skip verification"
        )
        assert "There is no unverified path" in text
        # And no code path may consult those variables to decide whether to verify.
        body = text.split('"""', 2)[-1]
        assert "ALLOW_UNVERIFIED" not in body and "ALLOW_LATEST" not in body, (
            "the fetch script reads a verification escape hatch; a build has no user to "
            "fall back for, so there must be no way to skip the digest check"
        )

    def test_the_fetch_script_refuses_plain_http(self):
        text = FETCH_SCRIPT.read_text(encoding = "utf-8")
        assert text.count('startswith("https://")') >= 2, (
            "every download must refuse a non-https URL"
        )

    def test_the_fetch_script_picks_the_macos_arm64_artifact_by_kind(self):
        """Matching on the release's own `kind` rather than on a filename pattern: the
        filename carries the release tag, so a tag bump would silently stop matching and
        the build would fall back to whatever else was there.
        """
        text = FETCH_SCRIPT.read_text(encoding = "utf-8")
        assert 'LLAMA_ARTIFACT_KIND = "macos-arm64-app"' in text
        assert 'WHISPER_ARTIFACT_KIND = "macos-arm64-slim-bundle"' in text
        assert "refusing to guess" in text, (
            "two artifacts claiming the same kind must fail rather than be picked from"
        )

    def test_sd_cpp_stays_opt_in_because_it_has_no_in_tree_digest(self):
        """The one prebuilt in this tree whose digest arrives over the same channel as
        the asset. That is fine for an opportunistic install and is not the standard the
        rest of the payload is held to, so a build only ships it when asked.
        """
        text = FETCH_SCRIPT.read_text(encoding = "utf-8")
        assert "--with-sd-cpp" in text
        assert "no in-tree digest" in text.lower()
        assert 'default = ["cpython", "llama.cpp", "whisper.cpp", "node"]' in text or (
            '"sd.cpp" not in wanted' in text
        )


# ── 7. the dev-build workflow ────────────────────────────────────────────────


class TestDevBuildWorkflow:
    def test_the_workflow_is_loadable_and_its_triggers_are_unchanged(self, dev_workflow):
        # `on` is parsed by PyYAML as True; accept either spelling.
        triggers = dev_workflow.get("on", dev_workflow.get(True))
        assert isinstance(triggers, dict), triggers
        assert set(triggers) == {"push", "workflow_dispatch"}, (
            f"the dev build's triggers changed to {sorted(triggers)}; adding a trigger "
            f"here is a change to what runs on untrusted input"
        )
        assert dev_workflow.get("permissions") == {"contents": "read"}

    def test_the_trigger_linter_still_passes_on_it(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "lint_workflow_triggers.py")],
            cwd = str(REPO_ROOT),
            capture_output = True,
            text = True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_payload_is_assembled_before_the_tauri_build(self, dev_workflow):
        """A payload failure is the likeliest failure and the cheapest to hit early;
        paying for a 20-minute Rust compile first buys nothing.
        """
        names = [str(step.get("name", "")) for step in _dev_steps(dev_workflow)]
        assemble = next(i for i, name in enumerate(names) if "Assemble the bundled runtime" in name)
        build = next(i for i, name in enumerate(names) if name == "Build the .dmg")
        assert assemble < build, f"payload assembly runs after tauri build: {names}"

    def test_the_workflow_uses_the_pinned_uv(self, dev_workflow):
        steps = _dev_steps(dev_workflow)
        uv_step = next(
            step for step in steps if "Install the pinned uv" in str(step.get("name", ""))
        )
        run = str(uv_step.get("run", ""))
        assert "UV_PINNED_VERSION" in run and "install.sh" in run, (
            "the uv version must be read out of install.sh, not written into the workflow"
        )
        assemble = next(
            step for step in steps if "Assemble the bundled runtime" in str(step.get("name", ""))
        )
        assert "UV" in (assemble.get("env") or {}), (
            "the assembly step must be handed the pinned uv"
        )

    def test_the_workflow_checks_the_bundle_lock_is_fresh(self, dev_workflow):
        runs = " ".join(str(step.get("run", "")) for step in _dev_steps(dev_workflow))
        assert "gen_macos_bundle_lock.sh --check" in runs, (
            "a stale bundle lock means the payload ships versions nobody reviewed"
        )

    def test_the_workflow_injects_the_payload_into_the_app_and_the_dmg(self, dev_workflow):
        runs = " ".join(str(step.get("run", "")) for step in _dev_steps(dev_workflow))
        assert "inject_macos_runtime.sh" in runs
        assert "--app" in runs and "--dmg" in runs

    def test_the_workflow_runs_the_bundled_interpreter_from_the_mounted_dmg(self, dev_workflow):
        """The assertion the whole payload exists for. Checking that files landed is
        not the same claim as the app being able to start, and only the second one
        matters to somebody who just copied Unsloth.app across.
        """
        step = next(
            item
            for item in _dev_steps(dev_workflow)
            if "Prove the bundled runtime works" in str(item.get("name", ""))
        )
        run = str(step.get("run", ""))
        assert "hdiutil attach" in run, "the check must run against the .dmg's own copy"
        assert "Contents/Resources/runtime" in run
        assert "-I -P -c 'import fastapi, transformers; print(\"ok\")'" in run, (
            "the workflow must import the stack with the bundled interpreter"
        )
        for module in ("torch", "mlx", "diffusers", "datasets", "uvicorn", "pymupdf", "bitsandbytes"):
            assert module in run, f"the import check no longer covers {module}"
        assert "PYTHONDONTWRITEBYTECODE=1" in run, (
            "the app runs the payload read-only; anything needing to write a .pyc must "
            "fail here rather than on a user's Mac"
        )
        assert "unsloth_cli --help" in run

    def test_the_workflow_asserts_llama_server_is_an_executable_arm64_binary(self, dev_workflow):
        step = next(
            item
            for item in _dev_steps(dev_workflow)
            if "Prove the bundled runtime works" in str(item.get("name", ""))
        )
        run = str(step.get("run", ""))
        assert "llama.cpp/build/bin/llama-server" in run
        assert "test -x" in run
        assert "grep -q 'arm64'" in run
        assert "--version" in run

    def test_the_workflow_parses_the_manifest_and_refuses_an_incomplete_payload(self, dev_workflow):
        step = next(
            item
            for item in _dev_steps(dev_workflow)
            if "Prove the bundled runtime works" in str(item.get("name", ""))
        )
        run = str(step.get("run", ""))
        assert "BUNDLE_MANIFEST.json" in run
        assert "json.load" in run
        assert 'manifest["incomplete"]' in run, (
            "a payload built with --skip-diffusers-pin must not pass as shippable"
        )

    def test_the_workflow_asserts_the_prune_actually_applied(self, dev_workflow):
        step = next(
            item
            for item in _dev_steps(dev_workflow)
            if "Prove the bundled runtime works" in str(item.get("name", ""))
        )
        assert "test ! -d \"$runtime/node/include\"" in str(step.get("run", "")), (
            "a reappearance of node/include means the prune list stopped applying"
        )

    def test_the_workflow_exercises_the_oxc_binding_with_the_bundled_node(self, dev_workflow):
        step = next(
            item
            for item in _dev_steps(dev_workflow)
            if "Prove the bundled runtime works" in str(item.get("name", ""))
        )
        run = str(step.get("run", ""))
        assert "node/bin/node" in run and "oxc-parser" in run, (
            "a Linux oxc binding in the payload would fail exactly here, and nowhere else"
        )

    def test_the_dmg_size_is_printed_in_the_job_summary(self, dev_workflow):
        """The payload costs hundreds of megabytes per build. Making that visible on
        every run is how it stays a decision rather than a drift.
        """
        runs = " ".join(str(step.get("run", "")) for step in _dev_steps(dev_workflow))
        assert "dmg_mib" in runs
        summary = next(
            step
            for step in _dev_steps(dev_workflow)
            if "GITHUB_STEP_SUMMARY" in str(step.get("run", ""))
        )
        rendered = str(summary.get("run", "")) + json.dumps(summary.get("env") or {})
        assert "dmg_mib" in rendered, (
            "the .dmg size must reach the job summary, not only a log line"
        )
        assert "dmg size" in str(summary.get("run", ""))
        # And the per-component breakdown, so a jump can be attributed.
        assert "sizes_bytes" in str(summary.get("run", ""))
        assert "bytes_freed" in str(summary.get("run", ""))

    def test_the_dmg_digest_is_recorded_after_injection(self, dev_workflow):
        """The uploaded .dmg is the injected one, so a digest taken before injection
        would describe a file nobody receives.
        """
        names = [str(step.get("name", "")) for step in _dev_steps(dev_workflow)]
        inject = next(i for i, name in enumerate(names) if "Inject the bundled runtime" in name)
        digest = next(i for i, name in enumerate(names) if "Record the .dmg digest" in name)
        assert inject < digest, f"the digest is taken before injection: {names}"

    def test_the_release_workflow_is_untouched_by_this_stage(self):
        """Release wiring and code signing are a separate stage. If the release
        workflow starts referencing these scripts without the signing question being
        answered, it would ship an app whose signature the injection invalidated.
        """
        if not RELEASE_WORKFLOW.is_file():
            pytest.skip("release-desktop.yml is not present")
        text = RELEASE_WORKFLOW.read_text(encoding = "utf-8")
        for script in (
            "build_macos_runtime.sh",
            "inject_macos_runtime.sh",
            "gen_macos_bundle_lock.sh",
        ):
            assert script not in text, (
                f"release-desktop.yml references {script}; wiring the payload into a "
                f"SIGNED build needs the signing order settled first (inject, then sign)."
            )
