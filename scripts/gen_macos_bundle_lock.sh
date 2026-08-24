#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Regenerate studio/backend/requirements/locks/darwin-arm64-bundle.lock.txt -- the
# hash-verified lock for the parts of the macOS arm64 app bundle that
# scripts/gen_python_locks.sh deliberately leaves unlocked.
#
# WHY A SECOND GENERATOR AND NOT A SEVENTH LOCK IN gen_python_locks.sh
#
# gen_python_locks.sh produces UNIVERSAL locks: one resolution valid on every OS,
# arch and Python, for the steps that are torch-INDEPENDENT. Its header explains
# what it cannot lock and why -- extras.txt because a with-deps universal
# resolution of it pins torch and the whole nvidia-* CUDA set, overriding the
# hardware-detected torch index; and the core unsloth/unsloth-zoo closure and the
# Apple MLX stack for the same reason.
#
# The app bundle is the one case where that reason does not apply. A macOS arm64
# bundle has NO GPU matrix: install.sh's get_torch_index_url hard-routes Darwin to
# a single cpu index, install.sh applies one static overrides-darwin-arm64.txt, and
# the release matrix builds only aarch64-apple-darwin. There is exactly ONE
# resolution to make, so it can be made once, hashed, reviewed in the diff, and
# installed with --require-hashes -- which is the whole point of baking a runtime
# into a signed app: the bytes inside the .dmg must be bytes somebody reviewed.
#
# WHAT THIS LOCKS
#
#   torch / torchvision / torchaudio   the constraints install.sh resolves on macOS
#   torchao                            the torch-matched override
#   unsloth + unsloth-zoo (with deps)  the core closure
#   extras.txt                         with deps
#   the MLX stack                      mlx / mlx-metal / mlx-lm / mlx-vlm
#   pytorch_tokenizers                 gen_python_locks.sh's carve-out, see below
#
# WHAT IT STILL DOES NOT LOCK
#
#   diffusers-pin.txt   A source ARCHIVE off github.com. gen_python_locks.sh
#                       excludes it for that reason and this follows the same
#                       reviewed decision: a URL requirement is resolved by
#                       fetching and building it, and its only identity is the
#                       URL. build_macos_runtime.sh installs it in its own late
#                       step, exactly as install_python_stack.py does, and records
#                       the installed archive's sha256 in BUNDLE_MANIFEST.json so
#                       the payload stays auditable.
#   triton-kernels.txt  A git+https requirement, and skipped on macOS anyway.
#
# THE CARVE-OUT COMES BACK IN
#
# gen_python_locks.sh carves pytorch_tokenizers out of extras-no-deps.lock.txt
# because the axis that decides the right version (musl vs glibc, the macOS
# deployment target a wheel was built against) has no PEP 508 marker, so a
# universal lock cannot express it. This lock is not universal: it is one platform
# at one deployment target, where that axis is a constant. So the cap resolves to
# exactly one version here and is hashed like everything else, and
# build_macos_runtime.sh never has to install anything unhashed for it.
#
# ONE INDEX, ON PURPOSE
#
# install.sh installs torch with `--default-index https://download.pytorch.org/whl/cpu`,
# and that index's macOS arm64 torch wheel has a different sha256 from PyPI's. It is
# not a different build. Measured on torch 2.10.0 cp313 macosx_11_0_arm64: the two
# wheels agree on all 12338 members except dist-info/METADATA and dist-info/RECORD,
# and the METADATA delta is only the seventeen `platform_system == "Linux" and
# platform_machine == "x86_64"` CUDA Requires-Dist lines the pytorch index strips --
# inert on macOS arm64. PyPI's copy additionally carries zip directory entries. The
# torch CODE is byte-identical, so this lock resolves from PyPI alone rather than
# adding a second index. That matters: with the pytorch index in play uv either
# fails (its stale numpy wins under the default first-index strategy and cannot
# satisfy scikit-learn's numpy==2.5.2) or, under --index-strategy unsafe-best-match,
# quietly sources jinja2 and markupsafe from download.pytorch.org too. A single
# index means no dependency-confusion surface to reason about, and
# tests/security/test_macos_runtime_bundle.py asserts the lock has exactly one.
#
# DETERMINISM
#
# Same four rules as gen_python_locks.sh, for the same reason:
#   1. uv is pinned to install.sh's UV_PINNED_VERSION.
#   2. --exclude-newer freezes the index at a fixed instant, shared with
#      gen_python_locks.sh so the two lock sets cannot describe different days.
#   3. The resolution is pinned to one platform, one Python and one deployment
#      target, all read from studio/macos_runtime_pins.json, so it does not depend
#      on the machine that generated it. This is why it can be, and is, generated
#      and exercised from Linux CI.
#   4. Nothing is fed to uv from a temp path: the compile input is synthesized on
#      stdin from install.sh and install_python_stack.py, so no mktemp name can
#      leak into a `# via` annotation, and the torch window and MLX ceilings cannot
#      drift from the files that own them.
#
# CROSS-FILE CONSTRAINTS
#
# The bundle applies the six existing locks and this one into ONE site-packages
# directory, so this compile takes every shipped requirements file AND every
# existing lock as constraints. Constraints bound versions without requiring
# anything, so this adds nothing to the closure; it only removes disagreement.
#
# USAGE
#
#   bash scripts/gen_macos_bundle_lock.sh            # regenerate in place
#   bash scripts/gen_macos_bundle_lock.sh --check    # fail if the tree is stale
#   UV=/path/to/uv bash scripts/gen_macos_bundle_lock.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_DIR="$REPO_ROOT/studio/backend/requirements"
LOCK_DIR="$REQ_DIR/locks"
STACK_PY="$REPO_ROOT/studio/install_python_stack.py"
PINS_JSON="$REPO_ROOT/studio/macos_runtime_pins.json"

LOCK_NAME="darwin-arm64-bundle"

# Index cutoff. MUST stay equal to gen_python_locks.sh's EXCLUDE_NEWER: the bundle
# installs those locks and this one into one directory, and two cutoffs would mean
# two different days' worth of upstream in one environment.
EXCLUDE_NEWER="2026-08-19T00:00:00Z"

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=1
elif [ $# -gt 0 ]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "error: missing required input: $1" >&2
        exit 1
    fi
}

require_file "$REPO_ROOT/install.sh"
require_file "$STACK_PY"
require_file "$PINS_JSON"

# ---------------------------------------------------------------- uv resolution

# Read the pin out of install.sh rather than duplicating it, exactly as
# gen_python_locks.sh does: install.sh is the file that downloads uv for the user.
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
# `|| true`: a $UV_BIN that exists but is not uv exits non-zero, and under
# `set -e` a failing command substitution would abort here with no message at
# all. Let the empty result reach the comparison below, which explains itself.
UV_ACTUAL="$("$UV_BIN" --version 2>/dev/null | head -n 1 | awk '{print $2}' || true)"
if [ "$UV_ACTUAL" != "$UV_PINNED_VERSION" ]; then
    # Refuse rather than warn, for gen_python_locks.sh's reason: a lock generated by
    # a different resolver is a lock for an install nobody performs.
    echo "error: '$UV_BIN' is uv ${UV_ACTUAL:-unknown}, but install.sh pins uv $UV_PINNED_VERSION." >&2
    echo "       Locks must be generated with the uv the installer actually runs." >&2
    echo "       e.g.  python3 -m venv /tmp/uvpin && /tmp/uvpin/bin/pip install uv==$UV_PINNED_VERSION" >&2
    echo "             UV=/tmp/uvpin/bin/uv bash scripts/gen_macos_bundle_lock.sh" >&2
    exit 1
fi

# ------------------------------------------------------------- the one platform

read_pin() {
    "$PYTHON_BIN" - "$PINS_JSON" "$1" <<'PY'
import json
import pathlib
import sys

path, dotted = sys.argv[1:3]
node = json.loads(pathlib.Path(path).read_text(encoding = "utf-8"))
for part in dotted.split("."):
    node = node[part]
if not isinstance(node, str) or not node:
    sys.exit(f"error: {dotted} in {path} is not a non-empty string")
print(node)
PY
}

PYTHON_MINOR="$(read_pin components.cpython.python_minor)"
UV_PYTHON_PLATFORM="$(read_pin target.uv_python_platform)"
MACOS_TARGET="$(read_pin target.macos_deployment_target)"

# ------------------------------------------------- the specs the repo already owns

# The macOS arm64 torch window, read out of install.sh so it cannot drift. On
# Apple Silicon install.sh takes the `_PY_MINOR -ge 13` branch, so the tightened
# constraint is the one that applies; the companions are file-global.
TORCH_CONSTRAINT="$(
    sed -n 's/^ *TORCH_CONSTRAINT="\(torch>=2\.6[^"]*\)".*/\1/p' "$REPO_ROOT/install.sh" | head -n 1
)"
TORCHVISION_CONSTRAINT="$(
    sed -n 's/^TORCHVISION_CONSTRAINT="\([^"]*\)".*/\1/p' "$REPO_ROOT/install.sh" | head -n 1
)"
TORCHAUDIO_CONSTRAINT="$(
    sed -n 's/^TORCHAUDIO_CONSTRAINT="\([^"]*\)".*/\1/p' "$REPO_ROOT/install.sh" | head -n 1
)"
for pair in "TORCH_CONSTRAINT:$TORCH_CONSTRAINT" \
            "TORCHVISION_CONSTRAINT:$TORCHVISION_CONSTRAINT" \
            "TORCHAUDIO_CONSTRAINT:$TORCHAUDIO_CONSTRAINT"; do
    if [ -z "${pair#*:}" ]; then
        echo "error: could not read ${pair%%:*} out of install.sh" >&2
        exit 1
    fi
done

# The Apple MLX window, read out of install_python_stack.py's _MLX_STACK_SPECS.
MLX_SPECS="$(
    "$PYTHON_BIN" - "$STACK_PY" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding = "utf-8")
match = re.search(
    r"^_MLX_STACK_SPECS[^=]*=\s*\(\s*(.*?)\s*\)\s*$",
    text,
    re.DOTALL | re.MULTILINE,
)
if match is None:
    sys.exit("error: could not read _MLX_STACK_SPECS out of install_python_stack.py")
specs = re.findall(r'"([^"]+)"', match.group(1))
if not specs:
    sys.exit("error: _MLX_STACK_SPECS parsed empty")
print("\n".join(specs))
PY
)"

# The torch-matched torchao override. install_python_stack.py picks it from the
# INSTALLED torch version via _select_torchao_spec; torch resolves inside `<2.11`
# here, and the CUDA-13 branch cannot be reached on macOS, so the 2.10 constant is
# the one that applies. Read, not re-derived: importing install_python_stack.py to
# call the function would run its host detection on the build machine, which is the
# opposite of what a cross-built lock wants.
TORCHAO_SPEC="$(
    sed -n 's/^_TORCHAO_TORCH_210_SPEC = "\([^"]*\)".*/\1/p' "$STACK_PY" | head -n 1
)"
if [ -z "$TORCHAO_SPEC" ]; then
    echo "error: could not read _TORCHAO_TORCH_210_SPEC out of install_python_stack.py" >&2
    exit 1
fi

# gen_python_locks.sh's carve-out, brought back in (see header).
CARVE_OUT_LINE="$(
    sed -n 's/^\(pytorch_tokenizers[<>=!~][^;]*\);.*/\1/p' "$REQ_DIR/extras-no-deps.txt" | head -n 1
)"
if [ -z "$CARVE_OUT_LINE" ]; then
    echo "error: could not read the pytorch_tokenizers cap out of extras-no-deps.txt" >&2
    exit 1
fi

# ------------------------------------------------------------------------ paths

if [ "$CHECK_ONLY" = "1" ]; then
    OUT_DIR="$(mktemp -d)"
    trap 'rm -rf "$OUT_DIR"' EXIT
else
    OUT_DIR="$LOCK_DIR"
fi
mkdir -p "$OUT_DIR"

# Every compile runs from the requirements directory so each constraint path, and
# each `# via` annotation, is a short repo-relative string.
cd "$REQ_DIR"

# Every shipped requirements file that lands in the same environment, plus every
# existing lock. Same list gen_python_locks.sh uses, plus the locks.
CROSS_CONSTRAINTS=(
    single-env/constraints.txt
    studio.txt
    extras.txt
    extras-no-deps.txt
    no-torch-runtime.txt
    single-env/data-designer-deps.txt
    single-env/data-designer.txt
    locks/pip-bootstrap.lock.txt
    locks/studio.lock.txt
    locks/extras-no-deps.lock.txt
    locks/no-torch-runtime.lock.txt
    locks/data-designer-deps.lock.txt
    locks/data-designer.lock.txt
)
for c in "${CROSS_CONSTRAINTS[@]}"; do
    require_file "$REQ_DIR/$c"
done

# The one override install.sh exports on macOS arm64 (UV_OVERRIDE). Without it
# mlx-vlm's own `transformers>=5.14` fights constraints.txt's transformers==5.5.0
# and the resolver walks the whole MLX stack backwards.
OVERRIDES="single-env/overrides-darwin-arm64.txt"
require_file "$REQ_DIR/$OVERRIDES"

# --------------------------------------------------------------------- the input

# Synthesized on stdin, so `-r extras.txt` resolves against REQ_DIR and no temp
# path can appear in the output.
compile_input() {
    printf '%s\n' "$TORCH_CONSTRAINT" "$TORCHVISION_CONSTRAINT" "$TORCHAUDIO_CONSTRAINT"
    printf '%s\n' "$TORCHAO_SPEC"
    printf 'unsloth\nunsloth-zoo\n'
    printf -- '-r extras.txt\n'
    printf '%s\n' "$MLX_SPECS"
    printf '%s\n' "$CARVE_OUT_LINE"
}

DISPLAY_SOURCE="install.sh torch window + install_python_stack.py MLX/torchao specs + unsloth/unsloth-zoo + extras.txt + extras-no-deps.txt carve-out"

LOCK="$OUT_DIR/$LOCK_NAME.lock.txt"
BODY="$(mktemp)"
# Replaces the check-mode trap above and covers both temporaries. Written as an if
# rather than `[ ... ] && rm`, so the trap's own last command cannot end non-zero and
# muddy the script's exit status.
cleanup_temps() {
    rm -f "$BODY"
    if [ "$CHECK_ONLY" = "1" ]; then
        rm -rf "$OUT_DIR"
    fi
}
trap cleanup_temps EXIT

echo "Regenerating $LOCK_NAME.lock.txt with uv $UV_PINNED_VERSION" >&2
echo "  platform: $UV_PYTHON_PLATFORM  python: $PYTHON_MINOR  MACOSX_DEPLOYMENT_TARGET: $MACOS_TARGET" >&2
echo "  index cutoff: $EXCLUDE_NEWER" >&2

constraint_args=()
for c in "${CROSS_CONSTRAINTS[@]}"; do
    constraint_args+=(-c "$c")
done

# The wheel-less requirements in extras.txt, read out of install_python_stack.py's
# SDIST_ONLY_PACKAGES so the two lists cannot drift. All four are pure Python, which
# is the only reason a cross-build may build them at all: a pure-Python sdist builds
# to the same py3-none-any wheel on any host, so a Linux runner producing them for a
# macOS bundle is sound. The MeCab exemption that function adds is macOS-cp314-only
# and so unreachable here, where the pin is 3.13.
build_args=()
while IFS= read -r sdist_pkg; do
    [ -n "$sdist_pkg" ] || continue
    build_args+=(--no-binary "$sdist_pkg")
done < <(
    "$PYTHON_BIN" - "$STACK_PY" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding = "utf-8")
match = re.search(r"^SDIST_ONLY_PACKAGES = \(\s*(.*?)\s*\)\s*$", text, re.DOTALL | re.MULTILINE)
if match is None:
    sys.exit("error: could not read SDIST_ONLY_PACKAGES out of install_python_stack.py")
names = re.findall(r'"([^"]+)"', match.group(1))
if not names:
    sys.exit("error: SDIST_ONLY_PACKAGES parsed empty")
print("\n".join(names))
PY
)
if [ "${#build_args[@]}" -eq 0 ]; then
    echo "error: no sdist-only packages resolved from install_python_stack.py" >&2
    exit 1
fi

# --only-binary :all: is a hard requirement, not a preference, with the four
# pure-Python exemptions above carved back out per package. Anything else falling
# back to an sdist would compile C on the build machine against the build machine's
# headers -- precisely what a prebuilt payload exists to avoid, and on a Linux
# runner it would produce Linux objects for a macOS app. Failing the resolution is
# the correct outcome, and it is a resolution-time failure rather than a
# discovered-at-launch one.
# Exported, not prefixed onto the pipeline: in `VAR=x a | b` the assignment reaches
# only `a`, and it is uv that has to see it. Getting this wrong is silent -- uv falls
# back to its own default target, which is low enough that bitsandbytes has no
# installable macOS arm64 wheel and the whole resolution walks backwards (`unsloth`
# lands on a 2024 release). See target.comment in studio/macos_runtime_pins.json.
export MACOSX_DEPLOYMENT_TARGET="$MACOS_TARGET"

if ! compile_input | "$UV_BIN" pip compile \
    --python-platform "$UV_PYTHON_PLATFORM" \
    --python-version "$PYTHON_MINOR" \
    --only-binary :all: \
    "${build_args[@]}" \
    --generate-hashes \
    --emit-index-annotation \
    --exclude-newer "$EXCLUDE_NEWER" \
    --no-header \
    --override "$OVERRIDES" \
    "${constraint_args[@]}" \
    - \
    -o "$BODY" >/dev/null
then
    echo "error: uv pip compile failed for $LOCK_NAME" >&2
    exit 1
fi

{
    echo "# GENERATED FILE -- DO NOT EDIT."
    echo "#"
    echo "# Hash-verified lock installed with --require-hashes by"
    echo "# scripts/build_macos_runtime.sh, which bakes it into"
    echo "# Unsloth.app/Contents/Resources/runtime/site-packages. Every version and every"
    echo "# digest below came from uv. A hand-written or hand-edited hash is worse than no"
    echo "# lock at all: it either fails every build, or asserts bytes nobody verified."
    echo "#"
    echo "# This lock is NOT universal and must never be installed on another platform."
    echo "# It is the single macOS arm64 resolution the app bundle ships, which is possible"
    echo "# only because macOS has no GPU matrix -- see scripts/gen_macos_bundle_lock.sh."
    echo "#"
    echo "# source:       $DISPLAY_SOURCE"
    echo "# generated by: uv $UV_PINNED_VERSION (install.sh UV_PINNED_VERSION)"
    echo "# regenerate:   bash scripts/gen_macos_bundle_lock.sh"
    echo "# index cutoff: $EXCLUDE_NEWER"
    echo "#"
    echo "# unsloth-lock-python-platform: $UV_PYTHON_PLATFORM"
    echo "# unsloth-lock-python-version: $PYTHON_MINOR"
    echo "# unsloth-lock-macos-deployment-target: $MACOS_TARGET"
    echo "#"
    cat "$BODY"
} > "$LOCK"

rm -f "$BODY"

PKGS="$(grep -c '^[A-Za-z0-9]' "$LOCK" || true)"
HASHES="$(grep -c -- '--hash=sha256:' "$LOCK" || true)"
echo "  $LOCK_NAME.lock.txt: $PKGS packages, $HASHES hashes" >&2

# ------------------------------------------------------------------------ check

if [ "$CHECK_ONLY" = "1" ]; then
    if diff -u "$LOCK_DIR/$LOCK_NAME.lock.txt" "$LOCK"; then
        echo "$LOCK_NAME.lock.txt is up to date" >&2
    else
        echo >&2
        echo "error: the committed $LOCK_NAME.lock.txt differs from a fresh regeneration." >&2
        echo "       A requirements file, the torch window, the MLX ceilings, the pins or" >&2
        echo "       EXCLUDE_NEWER moved without the lock being regenerated. Run:" >&2
        echo "           bash scripts/gen_macos_bundle_lock.sh" >&2
        exit 1
    fi
fi
