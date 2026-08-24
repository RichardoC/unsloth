#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Regenerate the hash-verified Python locks under
# studio/backend/requirements/locks/.
#
# WHY THESE FILES EXIST
#
# Every requirements file this repo ships is version-pinned, but a pin is not a
# digest: `pip install -r studio.txt` re-resolves the transitive closure from PyPI at
# install time and takes whatever bytes the index hands back. The locks close that
# gap for the torch-INDEPENDENT steps -- one fully-pinned, fully-hashed resolution
# per step, generated here, reviewed in the diff, installed with `--require-hashes`.
# See studio/install_python_stack.py:pip_install and studio/DETERMINISM.md.
#
# WHAT IS *NOT* LOCKED, AND WHY
#
#   extras.txt         Not torch-independent. timm / openai-whisper / torch-stoi /
#                      torchcodec all declare `torch`, so a with-deps universal
#                      resolution of this file pins torch, torchvision, torchaudio,
#                      triton and the whole nvidia-* CUDA set from PyPI. That would
#                      override the hardware-detected torch index the installer
#                      picks, which is a separate and much more fragile tier.
#   diffusers-pin.txt  A source ARCHIVE off github.com; uv has to fetch and build it
#                      to resolve it at all.
#   triton-kernels.txt A `git+https` requirement. A VCS requirement cannot carry a
#                      hash and uv rejects it under `--require-hashes`.
#   base.txt           Comment-only today; nothing to lock.
#   overrides.txt      Comment-only today; nothing to lock.
#
# DETERMINISM
#
# This script must produce byte-identical output on every run, or the CI freshness
# lane (.github/workflows/python-lock-freshness.yml) becomes a coin flip. Four
# things make that true:
#
#   1. uv is pinned to install.sh's UV_PINNED_VERSION -- the same uv the installer
#      runs, so a lock cannot encode a resolver no user ever uses.
#   2. `--exclude-newer` freezes the index at a fixed instant. Without it every
#      unbounded range (`urllib3>=2.3.0`) re-resolves upward the moment upstream
#      publishes, and the freshness lane fails on days nobody touched the repo.
#      Bumping EXCLUDE_NEWER is therefore the deliberate, reviewed act of taking
#      upstream releases: do it on purpose and read the diff.
#   3. `--universal` resolves for every platform and Python at once, so the lock does
#      not depend on the machine that generated it.
#   4. Nothing is fed to uv from a temp path. The two synthesized inputs (the pip
#      bootstrap, and the carve-out-stripped extras-no-deps) arrive on stdin, so no
#      `mktemp` name can leak into a `# via` annotation.
#
# PYTHON FLOOR
#
# pyproject's requires-python is >=3.9, but studio.txt, extras-no-deps.txt,
# no-torch-runtime.txt and the data-designer files each carry at least one
# UNCONDITIONAL pin requiring >=3.10 (matplotlib==3.10.9, peft==0.18.1,
# cut_cross_entropy==25.1.1, data-designer==0.5.4), so a 3.9 resolution of them is
# already unsatisfiable today, lock or no lock. The locks are compiled at a 3.10
# floor and record it in a machine-readable header line; install_python_stack.py
# reads that line and leaves a 3.9 interpreter on exactly today's unlocked path
# rather than handing it a lock whose `python_version < "3.10"` branches were
# resolved away.
#
# CROSS-FILE CONSTRAINTS
#
# The install applies these files one after another into ONE environment, and an
# unlocked step leaves an already-satisfied range alone. A lock does not -- it pins,
# so it can silently override a pin from another file. Measured, on this tree: an
# independent resolution of single-env/data-designer-deps.txt wanted pymupdf 1.28.2
# (exactly the lockstep line studio.txt's comment says to stay off, because it makes
# pymupdf-layout -> onnxruntime a hard dep), tiktoken 0.14.0 over extras.txt's
# 0.13.0, uvicorn 0.52.3 over studio.txt's 0.52.1, and fsspec 2025.12.0 over the cap
# datasets==4.3.0 imposes. So every compile passes the other shipped requirements
# files as CONSTRAINTS, and data-designer-deps.txt additionally takes the studio lock.
# Constraints bound versions without requiring anything, so this adds nothing to any
# closure; it only removes disagreement. tests/security/test_python_locks.py asserts
# the resulting property: no lock contradicts a version this repo pins elsewhere.
#
# CARVE-OUTS
#
# A universal lock names ONE version per package per marker fork, so an entry whose
# real bound cannot be written as a PEP 508 marker must stay out of it. Carve-outs
# are stripped from the compile input and written to `<name>.unlocked.txt`, which
# install_python_stack.py installs unlocked (with constraints) right after the lock.
# See CARVE_OUT_* below for the list and the reason.
#
# USAGE
#
#   bash scripts/gen_python_locks.sh              # regenerate in place
#   bash scripts/gen_python_locks.sh --check      # fail if the committed tree is stale
#   UV=/path/to/uv bash scripts/gen_python_locks.sh
#
# --check writes to a temp directory and diffs, so it never mutates the tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_DIR="$REPO_ROOT/studio/backend/requirements"
LOCK_DIR="$REQ_DIR/locks"
STACK_PY="$REPO_ROOT/studio/install_python_stack.py"

# The lowest Python every lock is resolved for, also written into each lock as
# `# unsloth-lock-python-floor:`.
PYTHON_FLOOR="3.10"

# Index cutoff. Bump deliberately; see DETERMINISM above.
EXCLUDE_NEWER="2026-08-19T00:00:00Z"

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=1
elif [ $# -gt 0 ]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"

# ---------------------------------------------------------------- uv resolution

# Read the pin out of install.sh rather than duplicating it: install.sh is the file
# that downloads uv for the user, so its version is the only one that matters.
UV_PINNED_VERSION="$(
    sed -n 's/^UV_PINNED_VERSION="\([^"]*\)".*/\1/p' "$REPO_ROOT/install.sh" | head -n 1
)"
if [ -z "$UV_PINNED_VERSION" ]; then
    echo "error: could not read UV_PINNED_VERSION out of install.sh" >&2
    exit 1
fi

UV_BIN="${UV:-uv}"
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo "error: '$UV_BIN' not found. Install uv $UV_PINNED_VERSION, or pass UV=/path/to/uv." >&2
    exit 1
fi
UV_ACTUAL="$("$UV_BIN" --version 2>/dev/null | awk '{print $2}')"
if [ "$UV_ACTUAL" != "$UV_PINNED_VERSION" ]; then
    # Refuse rather than warn: a lock generated by a different resolver is a lock for
    # an install nobody performs, and the difference is invisible in review.
    echo "error: '$UV_BIN' is uv ${UV_ACTUAL:-unknown}, but install.sh pins uv $UV_PINNED_VERSION." >&2
    echo "       Locks must be generated with the uv the installer actually runs." >&2
    echo "       e.g.  python3 -m venv /tmp/uvpin && /tmp/uvpin/bin/pip install uv==$UV_PINNED_VERSION" >&2
    echo "             UV=/tmp/uvpin/bin/uv bash scripts/gen_python_locks.sh" >&2
    exit 1
fi

# ------------------------------------------------------------------- carve-outs

# extras-no-deps.txt: `pytorch_tokenizers<=1.4.1` is a CAP on purpose. 1.4.1 ships no
# musllinux wheel (1.1.0 is the newest that does) and its arm64 wheel is macosx_14_0,
# so a musl host needs 1.1.0 and a macOS 13 arm64 host needs 1.2.0 while macOS 14+
# takes 1.4.1. PEP 508 can express neither axis, so a universal compile pins the
# newest and hands both of those hosts the one release carrying an sdist -- the cmake
# build that file is written to avoid. Leaving the cap unlocked keeps the resolver's
# fallback, which is the whole point of writing it as a cap.
CARVE_OUT_EXTRAS_NO_DEPS="pytorch_tokenizers"

# ------------------------------------------------------------------------ paths

if [ "$CHECK_ONLY" = "1" ]; then
    OUT_DIR="$(mktemp -d)"
    trap 'rm -rf "$OUT_DIR"' EXIT
else
    OUT_DIR="$LOCK_DIR"
fi
mkdir -p "$OUT_DIR"

# Every compile runs from the requirements directory so each constraint path, and
# each `# via` annotation, is a short repo-relative string rather than this machine's
# absolute layout.
cd "$REQ_DIR"

# Every shipped requirements file that lands in the same environment, passed as
# constraints to every compile. Excludes diffusers-pin.txt (URL requirement) and
# triton-kernels.txt (git requirement), which are not version specifiers uv can
# constrain against, and the comment-only base.txt / overrides.txt.
CROSS_CONSTRAINTS=(
    single-env/constraints.txt
    studio.txt
    extras.txt
    extras-no-deps.txt
    no-torch-runtime.txt
    single-env/data-designer-deps.txt
    single-env/data-designer.txt
)

# --------------------------------------------------------------------- compiler

# compile_lock <name> <compile-input> <display-source> [extra uv args...]
#
# <compile-input> is a path relative to REQ_DIR, or "-" to read the requirements
# from stdin. Writes $OUT_DIR/<name>.lock.txt. Every requirement line and every
# hash comes from uv verbatim; this only prepends a fixed header.
compile_lock() {
    local name="$1" input="$2" display="$3"
    shift 3

    local lock="$OUT_DIR/$name.lock.txt"
    local body
    body="$(mktemp)"

    local -a constraint_args=()
    local c
    for c in "${CROSS_CONSTRAINTS[@]}"; do
        constraint_args+=(-c "$c")
    done

    echo "  $name.lock.txt  <-  $display" >&2
    if ! "$UV_BIN" pip compile \
        --universal \
        --generate-hashes \
        --python-version "$PYTHON_FLOOR" \
        --exclude-newer "$EXCLUDE_NEWER" \
        --no-header \
        "${constraint_args[@]}" \
        "$@" \
        "$input" \
        -o "$body" >/dev/null
    then
        rm -f "$body"
        echo "error: uv pip compile failed for $display" >&2
        exit 1
    fi

    {
        echo "# GENERATED FILE -- DO NOT EDIT."
        echo "#"
        echo "# Hash-verified lock installed with --require-hashes by"
        echo "# studio/install_python_stack.py. Every version and every digest below came"
        echo "# from uv. A hand-written or hand-edited hash is worse than no lock at all:"
        echo "# it either fails every install, or asserts bytes nobody verified."
        echo "#"
        echo "# source:       $display"
        echo "# generated by: uv $UV_PINNED_VERSION (install.sh UV_PINNED_VERSION)"
        echo "# regenerate:   bash scripts/gen_python_locks.sh"
        echo "# index cutoff: $EXCLUDE_NEWER"
        echo "#"
        echo "# unsloth-lock-python-floor: $PYTHON_FLOOR"
        echo "#"
        cat "$body"
    } > "$lock"

    rm -f "$body"
}

# split_carve_outs <source> <side-file> <names...>
#
# Prints the source with the named requirements removed (that is what gets
# compiled), and writes those requirements, with the comment block above each one
# so the reason travels with them, to <side-file>.
split_carve_outs() {
    local source="$1" side="$2"
    shift 2
    "$PYTHON_BIN" - "$source" "$side" "$@" <<'PY'
import pathlib
import re
import sys

source, side, *names = sys.argv[1:]
wanted = {re.sub(r"[-_.]+", "-", n).lower() for n in names}


def name_of(line):
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:[<>=!~;\[]|$)", line.strip())
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else None


lines = pathlib.Path(source).read_text(encoding = "utf-8").splitlines()
kept, carved, block, found = [], [], [], set()
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        block.append(line)
        continue
    if name_of(line) in wanted:
        # The comment block directly above a requirement documents it; move it with
        # the requirement instead of stranding it in the compile input.
        carved.extend(block)
        carved.append(line)
        found.add(name_of(line))
    else:
        kept.extend(block)
        kept.append(line)
    block = []
kept.extend(block)

missing = wanted - found
if missing:
    sys.exit(f"error: carve-out {sorted(missing)} not found in {source}")

pathlib.Path(side).write_text(
    "# GENERATED FILE -- DO NOT EDIT. See scripts/gen_python_locks.sh.\n"
    "#\n"
    f"# Carved out of {source} and its lock: a universal lock names one version per\n"
    "# package, and these bounds are deliberately NOT exact because the axis that\n"
    "# decides the right version (musl vs glibc, the macOS deployment target a wheel\n"
    "# was built against) has no PEP 508 marker. Pinning them would push those hosts\n"
    "# onto the one release carrying an sdist and into a source build.\n"
    "#\n"
    "# install_python_stack.py installs this file UNCONSTRAINED BY HASHES, with the\n"
    "# shared constraints applied, immediately after the lock for the same step.\n"
    "#\n" + "\n".join(carved) + "\n",
    encoding = "utf-8",
)
print("\n".join(kept))
PY
}

# ---------------------------------------------------------------------- the run

echo "Regenerating Python locks with uv $UV_PINNED_VERSION (index cutoff $EXCLUDE_NEWER)" >&2

# pip itself, the resolver every non-uv step runs through: the one step where an
# unverified download decides what every later download means. Not a requirements
# file -- the version lives in install_python_stack.py, and the input is synthesized
# from it so the two cannot drift.
PIP_BOOTSTRAP_VERSION="$(
    sed -n 's/^_PIP_BOOTSTRAP_VERSION = "\([^"]*\)".*/\1/p' "$STACK_PY" | head -n 1
)"
if [ -z "$PIP_BOOTSTRAP_VERSION" ]; then
    echo "error: could not read _PIP_BOOTSTRAP_VERSION out of install_python_stack.py" >&2
    exit 1
fi
printf 'pip==%s\n' "$PIP_BOOTSTRAP_VERSION" | compile_lock \
    pip-bootstrap - \
    "install_python_stack.py _PIP_BOOTSTRAP_VERSION (pip==$PIP_BOOTSTRAP_VERSION)" \
    --no-deps

# studio.txt first: its lock constrains the data-designer compile below.
compile_lock studio studio.txt studio.txt

split_carve_outs extras-no-deps.txt "$OUT_DIR/extras-no-deps.unlocked.txt" \
    "$CARVE_OUT_EXTRAS_NO_DEPS" |
    compile_lock extras-no-deps - \
        "extras-no-deps.txt (minus carve-outs: $CARVE_OUT_EXTRAS_NO_DEPS)" \
        --no-deps

compile_lock no-torch-runtime no-torch-runtime.txt no-torch-runtime.txt --no-deps

# The one extra constraint. single-env/data-designer-deps.txt is applied AFTER
# studio.txt into the same environment and its closure overlaps studio's heavily, so
# it is resolved against the studio lock; without this it pins its own newer
# pymupdf / fsspec / tiktoken / uvicorn over what the studio step just installed.
compile_lock data-designer-deps single-env/data-designer-deps.txt \
    single-env/data-designer-deps.txt \
    -c locks/studio.lock.txt

compile_lock data-designer single-env/data-designer.txt single-env/data-designer.txt --no-deps

# ------------------------------------------------------------------------ check

if [ "$CHECK_ONLY" = "1" ]; then
    # Locks that live in LOCK_DIR but are NOT produced here, excluded from the diff so
    # their presence is not read as "the committed locks are stale".
    #
    # scripts/gen_macos_bundle_lock.sh owns darwin-arm64-bundle.lock.txt: one
    # single-platform resolution for the macOS app bundle, covering exactly the steps
    # the WHAT IS *NOT* LOCKED section above explains this generator cannot lock.
    # It has its own --check, run by the same lanes.
    #
    # This is an allowlist, not a wildcard: any OTHER unexpected file in LOCK_DIR still
    # fails the diff, which is the property this check exists to protect.
    FOREIGN_LOCKS=(darwin-arm64-bundle.lock.txt)
    diff_args=()
    for foreign in "${FOREIGN_LOCKS[@]}"; do
        if [ ! -f "$LOCK_DIR/$foreign" ]; then
            echo "error: $FOREIGN_LOCKS references $foreign, which is not committed." >&2
            echo "       Regenerate it (bash scripts/gen_macos_bundle_lock.sh) or drop it" >&2
            echo "       from FOREIGN_LOCKS here." >&2
            exit 1
        fi
        diff_args+=("--exclude=$foreign")
    done

    if diff -ru "${diff_args[@]}" "$LOCK_DIR" "$OUT_DIR"; then
        echo "locks are up to date" >&2
    else
        echo >&2
        echo "error: the committed locks differ from a fresh regeneration." >&2
        echo "       A requirements file changed without its lock being regenerated," >&2
        echo "       or EXCLUDE_NEWER / the uv pin moved. Run:" >&2
        echo "           bash scripts/gen_python_locks.sh" >&2
        exit 1
    fi
fi
