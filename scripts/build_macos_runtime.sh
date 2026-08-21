#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Assemble the complete, self-contained macOS arm64 runtime that ships inside
# Unsloth.app/Contents/Resources/runtime, so copying the app off the .dmg is the
# whole installation. Today first launch builds a Python environment under
# ~/.unsloth; this replaces that with bytes that were downloaded, verified and
# reviewed at build time.
#
# THE PAYLOAD (the contract studio/src-tauri reads; see layout in
# studio/macos_runtime_pins.json, which is the machine-readable copy of this)
#
#   runtime/python/                relocatable CPython, so python/bin/python3 works
#   runtime/site-packages/         every installed distribution, flat
#   runtime/llama.cpp/             llama.cpp prebuilt, ~/.unsloth/llama.cpp layout
#   runtime/whisper.cpp/           whisper.cpp prebuilt, same layout
#   runtime/stable-diffusion.cpp/  sd-cli + sd-server prebuilt, same layout
#   runtime/node/                  trimmed Node (bin/node + npm, no include/)
#   runtime/oxc-node-modules/      prefetched node_modules for the OXC validator
#   runtime/BUNDLE_MANIFEST.json   what is inside, and where every byte came from
#
# The app runs `runtime/python/bin/python3 -I -P -m unsloth_cli ...` with
# PYTHONHOME/PYTHONPATH pointing into runtime/ and PYTHONDONTWRITEBYTECODE=1.
# ~/.unsloth/studio stays the writable data root (DBs, logs, caches, models);
# only code moves into the bundle.
#
# NO VENV, ON PURPOSE
#
# A venv bakes absolute paths into pyvenv.cfg and into every console script's
# shebang, and does not survive being moved -- which is the one thing this payload
# must do, since it is built in a CI checkout and read from /Applications. So the
# distributions go into a plain directory via `uv pip install --target`, and the
# interpreter finds them through PYTHONPATH. The console scripts uv writes into
# site-packages/bin DO carry an absolute shebang, so this script rewrites every one
# of them to the relative `#!/bin/sh` + exec form that python-build-standalone
# already uses for its own bin/pip and bin/idle3 -- same mechanism, not a new one.
#
# THE BACKEND IS THIS CHECKOUT, NOT A PUBLISHED WHEEL
#
# darwin-arm64-bundle.lock.txt pins `unsloth` like any other distribution, because
# resolving the closure needs a version of it to resolve against. But the payload must
# not KEEP that wheel: the app and the backend are released together from one commit,
# so the code inside the bundle has to be the code being built. Installing the index
# copy instead ships published code behind a new shell -- which is exactly how the
# bundled-runtime seam (unsloth_cli/_bundled_runtime.py, and _serves_backend_in_process
# in unsloth_cli/commands/studio.py) came to be missing from a payload whose Rust side
# depended on it: preflight said Ready, then the CLI took the old
# sys.prefix-under-STUDIO_HOME path and printed "Unsloth Studio not set up. Run
# install.sh first." on a machine where install.sh is never going to run.
#
# So step 3 below builds a wheel from this working tree and overlays it: the
# lock-installed `unsloth` files are removed through their own RECORD and the local
# wheel is installed with --no-deps in their place. Everything else in site-packages
# stays exactly as the hash-verified locks placed it -- unsloth-zoo included, which is
# a separate upstream project and stays at the version the lock pins. The overlay then
# ASSERTS the seam is present, so a payload built without it fails here rather than on
# somebody's Mac, and BUNDLE_MANIFEST.json records that this one distribution came from
# the checkout rather than from an index.
#
# CROSS-BUILDABLE, AND THEREFORE TESTABLE
#
# Every step here works from any host: uv installs for an explicit
# --python-platform / --python-version with MACOSX_DEPLOYMENT_TARGET set, npm
# fetches with --os/--cpu, and the prebuilts are picked by artifact `kind` rather
# than by probing the machine. That is not a nicety -- it is what lets the payload
# be assembled and inspected on a Linux runner instead of only being discovered to
# be wrong on a Mac. Two things it cannot do from a foreign host are noted where
# they appear: it cannot run the bundled interpreter, and it cannot verify Metal.
#
# WHAT IS HASH-VERIFIED, AND WHAT IS NOT
#
#   Python distributions   Every one, via --require-hashes against the six locks
#                          gen_python_locks.sh generates plus the macOS bundle lock
#                          gen_macos_bundle_lock.sh generates.
#   Prebuilt archives      Every one, fail-closed, through the in-tree digest chain
#                          (see scripts/fetch_macos_prebuilts.py). llama.cpp and
#                          whisper.cpp chain through prebuilt_release_pins.json;
#                          CPython, Node and stable-diffusion.cpp are compared
#                          directly against a digest committed in this tree.
#   OXC node_modules       npm ci against the committed package-lock.json, whose
#                          every entry carries an integrity digest.
#   The two local          Built from this checkout. Their identity is the commit.
#   data-designer plugins
#   unsloth itself         Built from this checkout too, over the lock-installed copy
#                          (see below). NOT index-verified, and it cannot be: it is the
#                          code being released. Its identity is the commit, recorded in
#                          BUNDLE_MANIFEST.json together with the wheel's own sha256.
#   The diffusers pin      NOT hash-verified, and the one thing here that is not.
#                          It is a source archive off github.com;
#                          gen_python_locks.sh excludes diffusers-pin.txt for
#                          exactly that reason and this follows the same reviewed
#                          decision rather than inventing a second one. The
#                          installed archive's sha256 IS recorded in
#                          BUNDLE_MANIFEST.json, so the payload stays auditable and
#                          a future pin has something to be compared against.
#
# USAGE
#
#   bash scripts/build_macos_runtime.sh --out <dir>       # assemble into <dir>
#   bash scripts/build_macos_runtime.sh --out <dir> --skip-diffusers-pin
#   bash scripts/build_macos_runtime.sh --out <dir> --skip-sd-cpp
#   UV=/path/to/uv bash scripts/build_macos_runtime.sh --out <dir>
#
# --skip-diffusers-pin exists for build hosts that cannot reach
# github.com/*/archive/*.zip (some egress policies block it while allowing
# releases). --skip-sd-cpp exists to shave a 45 MiB download off a local iteration
# on something else entirely. NEITHER may ship: both are recorded in the manifest's
# `incomplete` list, and the dev-build workflow refuses a payload whose `incomplete`
# is non-empty, so a build that used one cannot be mistaken for a shippable one.
# --skip-sd-cpp additionally removes the stable-diffusion.cpp paths from the layout
# assertion below -- narrowly, by name, only for the component that was skipped, so
# a component that was MEANT to be there and is missing still fails the build.

set -euo pipefail

# Bumped whenever the payload's shape or contents change in a way an auditor
# reading an old BUNDLE_MANIFEST.json would need to know about.
#   2: stable-diffusion.cpp joined the payload as a required slot.
#   3: `unsloth` no longer comes from the index. The payload's copy is built from the
#      checkout the build ran in, so site-packages carries the same commit as the app
#      around it; local_provenance in the manifest says so, and names the commit.
GENERATOR_VERSION="3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_DIR="$REPO_ROOT/studio"
REQ_DIR="$STUDIO_DIR/backend/requirements"
LOCK_DIR="$REQ_DIR/locks"
PINS_JSON="$STUDIO_DIR/macos_runtime_pins.json"
OXC_DIR="$STUDIO_DIR/backend/core/data_recipe/oxc-validator"

OUT_DIR=""
WORK_DIR=""
KEEP_WORK=0
SKIP_DIFFUSERS=0
# stable-diffusion.cpp is part of the payload, not an extra: the app is meant to need
# no first-run download, and image generation was the one feature still reaching for
# one. It has an in-tree digest anchor now (studio/macos_runtime_pins.json
# components.sd_cpp), which is what kept it out before.
SKIP_SD_CPP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT_DIR="${2:-}"; shift 2 ;;
        --work-dir) WORK_DIR="${2:-}"; KEEP_WORK=1; shift 2 ;;
        --skip-diffusers-pin) SKIP_DIFFUSERS=1; shift ;;
        --skip-sd-cpp) SKIP_SD_CPP=1; shift ;;
        -h|--help)
            sed -n '2,113p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUT_DIR" ]; then
    echo "error: --out <dir> is required" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"

die() { echo "error: $*" >&2; exit 1; }

require_file() { [ -f "$1" ] || die "missing required input: $1"; }
require_dir()  { [ -d "$1" ] || die "missing required input directory: $1"; }

# ─────────────────────────────────────────────────────────── preconditions

require_file "$REPO_ROOT/install.sh"
require_file "$PINS_JSON"
require_file "$STUDIO_DIR/prebuilt_release_pins.json"
require_file "$STUDIO_DIR/node_prebuilt_pins.json"
require_file "$REPO_ROOT/scripts/fetch_macos_prebuilts.py"
require_dir  "$OXC_DIR"
require_file "$OXC_DIR/package-lock.json"

# Every lock, in the order studio/install_python_stack.py applies the corresponding
# requirements file, so the payload is the same environment a user's install
# produces rather than a second opinion about it. The union of these seven files is
# the whole Python side of the bundle:
#
#   pip-bootstrap          pip itself
#   darwin-arm64-bundle    torch + torchao + unsloth/-zoo + extras.txt + MLX +
#                          the pytorch_tokenizers carve-out (this repo's macOS lock)
#   no-torch-runtime       the runtime deps the no-torch path shares
#   extras-no-deps         audio codecs, peft, trl, transformers, kernels
#   studio                 the Studio backend closure
#   data-designer-deps     data-designer's base closure
#   data-designer          the data-designer packages themselves
#
# Order still matters even under --no-deps: where two locks name the same
# distribution at the same version the later install simply rewrites identical
# bytes, but a lock's own view of a shared package should land in the order the
# installer would land it, so a future divergence shows up here rather than as a
# difference between the bundle and a user's machine.
LOCK_STEPS=(
    pip-bootstrap.lock.txt
    darwin-arm64-bundle.lock.txt
    no-torch-runtime.lock.txt
    extras-no-deps.lock.txt
    studio.lock.txt
    data-designer-deps.lock.txt
    data-designer.lock.txt
)
for lock in "${LOCK_STEPS[@]}"; do
    require_file "$LOCK_DIR/$lock"
done

# The two local data-designer seed plugins install_python_stack.py installs.
DD_PLUGINS=(
    "$STUDIO_DIR/backend/plugins/data-designer-unstructured-seed"
    "$STUDIO_DIR/backend/plugins/data-designer-github-repo-seed"
)
for plugin in "${DD_PLUGINS[@]}"; do
    require_dir "$plugin"
done

UV_PINNED_VERSION="$(
    sed -n 's/^UV_PINNED_VERSION="\([^"]*\)".*/\1/p' "$REPO_ROOT/install.sh" | head -n 1
)"
[ -n "$UV_PINNED_VERSION" ] || die "could not read UV_PINNED_VERSION out of install.sh"

UV_BIN="${UV:-uv}"
command -v "$UV_BIN" >/dev/null 2>&1 || die "'$UV_BIN' not found; install uv $UV_PINNED_VERSION or pass UV=..."
# `|| true`: a $UV_BIN that exists but is not uv exits non-zero, and under
# `set -e` a failing command substitution would abort here with no message at
# all. Let the empty result reach the comparison below, which explains itself.
UV_ACTUAL="$("$UV_BIN" --version 2>/dev/null | head -n 1 | awk '{print $2}' || true)"
if [ "$UV_ACTUAL" != "$UV_PINNED_VERSION" ]; then
    # Same refusal as the lock generators, for the same reason: a payload installed
    # by a different resolver is a payload for an install nobody performs.
    die "'$UV_BIN' is uv ${UV_ACTUAL:-unknown}, but install.sh pins uv $UV_PINNED_VERSION"
fi

command -v npm >/dev/null 2>&1 || die "npm not found; it is needed to prefetch the OXC node_modules"

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
PYTHON_FULL="$(read_pin components.cpython.python_version)"
UV_PYTHON_PLATFORM="$(read_pin target.uv_python_platform)"
RUST_TARGET="$(read_pin target.rust_target)"
MACOS_TARGET="$(read_pin target.macos_deployment_target)"

# The lock's own header records the platform it was resolved for. Compare rather
# than trust: installing a lock resolved for another platform would silently select
# the wrong wheels, and the failure would surface as an ImportError on a user's Mac.
BUNDLE_LOCK="$LOCK_DIR/darwin-arm64-bundle.lock.txt"
LOCK_PLATFORM="$(sed -n 's/^# unsloth-lock-python-platform: *//p' "$BUNDLE_LOCK" | head -n 1)"
LOCK_PYVER="$(sed -n 's/^# unsloth-lock-python-version: *//p' "$BUNDLE_LOCK" | head -n 1)"
LOCK_MACOS="$(sed -n 's/^# unsloth-lock-macos-deployment-target: *//p' "$BUNDLE_LOCK" | head -n 1)"
[ "$LOCK_PLATFORM" = "$UV_PYTHON_PLATFORM" ] || die \
    "$BUNDLE_LOCK was resolved for '$LOCK_PLATFORM' but the pins say '$UV_PYTHON_PLATFORM'; regenerate it"
[ "$LOCK_PYVER" = "$PYTHON_MINOR" ] || die \
    "$BUNDLE_LOCK was resolved for Python '$LOCK_PYVER' but the pins say '$PYTHON_MINOR'; regenerate it"
[ "$LOCK_MACOS" = "$MACOS_TARGET" ] || die \
    "$BUNDLE_LOCK was resolved for MACOSX_DEPLOYMENT_TARGET '$LOCK_MACOS' but the pins say '$MACOS_TARGET'; regenerate it"

# uv honours MACOSX_DEPLOYMENT_TARGET when it builds the macOS platform tags for
# --python-platform (measured: at 13.0 pytorch-tokenizers resolves 1.2.0, at 14.0
# it resolves 1.4.1). Export it once, here, for every uv call below.
export MACOSX_DEPLOYMENT_TARGET="$MACOS_TARGET"

# ─────────────────────────────────────────────────────────────── scratch space

if [ -n "$WORK_DIR" ]; then
    mkdir -p "$WORK_DIR"
    WORK_DIR="$(cd "$WORK_DIR" && pwd)"
else
    WORK_DIR="$(mktemp -d)"
fi
cleanup() { [ "$KEEP_WORK" = "1" ] || rm -rf "$WORK_DIR"; }
trap cleanup EXIT

# The payload is built in full at a staging path and swapped in at the end, so an
# interrupted run never leaves a half-assembled runtime that looks complete.
STAGE="$WORK_DIR/runtime"
rm -rf "$STAGE"
mkdir -p "$STAGE"
SITE="$STAGE/site-packages"
mkdir -p "$SITE"

echo "==> assembling the macOS arm64 runtime payload"
echo "    target:        $RUST_TARGET ($UV_PYTHON_PLATFORM)"
echo "    python:        $PYTHON_FULL (minor $PYTHON_MINOR)"
echo "    deployment:    macOS $MACOS_TARGET"
echo "    uv:            $UV_ACTUAL"
echo "    staging:       $STAGE"

# ───────────────────────────────────── 1. the prebuilts (CPython + native bits)

echo
echo "==> 1/7 prebuilt archives (verified against the in-tree digest chain)"
PREBUILT_REPORT="$WORK_DIR/prebuilts.json"
fetch_args=(
    "$REPO_ROOT/scripts/fetch_macos_prebuilts.py"
    --runtime-dir "$STAGE"
    --report "$PREBUILT_REPORT"
    --work-dir "$WORK_DIR/downloads"
)
SD_CPP_STATUS="installed"
if [ "$SKIP_SD_CPP" = "1" ]; then
    SD_CPP_STATUS="skipped (--skip-sd-cpp)"
    fetch_args+=(--skip-sd-cpp)
    echo "    -> stable-diffusion.cpp SKIPPED; this payload is incomplete"
fi
"$PYTHON_BIN" "${fetch_args[@]}" || die "prebuilt fetch failed"

# The interpreter the bundle ships must be the one the locks were resolved for.
BUNDLED_PY="$STAGE/python/bin/python$PYTHON_MINOR"
require_file "$BUNDLED_PY"

# ───────────────────────────────────────────── 2. the Python distributions

echo
echo "==> 2/7 Python distributions (--require-hashes, into site-packages/)"

# Shared by every uv call: one platform, one Python, no host interpreter consulted
# for markers, and nothing cached between runs.
#
# NOTE ON --no-deps, WHICH EVERY STEP BELOW USES
#
# A lock is a closure, so there is nothing left to resolve at install time: this
# phase places bytes, it does not decide anything. That is not just tidiness, it is
# forced. Resolution and --require-hashes are mutually exclusive here -- uv still
# validates the dependency graph when it is not given --no-deps, and satisfying that
# graph on macOS arm64 needs the UV_OVERRIDE file install.sh exports (without it
# mlx-audio's own `transformers>=5.14.0`, pulled in by mlx-vlm, collides with
# constraints.txt's transformers==5.5.0 and the install fails outright), while
# --require-hashes rejects that file's unpinned entries by design. The override
# therefore belongs where the resolution happens, which is
# scripts/gen_macos_bundle_lock.sh, and NOT here. Every version in the payload was
# decided once, in a lock, in a reviewed diff.
uv_common=(
    --target "$SITE"
    --python-platform "$UV_PYTHON_PLATFORM"
    --python-version "$PYTHON_MINOR"
    --no-deps
    --no-cache
)

# The pure-Python sdist-only requirements, read out of install_python_stack.py so
# the list cannot drift. See gen_macos_bundle_lock.sh for why building these, and
# only these, is sound on a foreign host.
#
# Read with a while loop rather than `mapfile`: this script's whole point is to run
# on macOS, where /bin/bash is 3.2 and mapfile does not exist. It is a bash 4
# builtin, so it works on a Linux runner and dies with "mapfile: command not found"
# on the platform we are building for -- which is exactly how it shipped broken.
SDIST_ONLY=()
while IFS= read -r _sdist_name; do
    [ -n "$_sdist_name" ] || continue
    SDIST_ONLY+=("$_sdist_name")
done < <(
    "$PYTHON_BIN" - "$STUDIO_DIR/install_python_stack.py" <<'PY'
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
[ "${#SDIST_ONLY[@]}" -gt 0 ] || die "no sdist-only packages resolved from install_python_stack.py"
binary_args=(--only-binary :all:)
for name in "${SDIST_ONLY[@]}"; do
    binary_args+=(--no-binary "$name")
done

for lock in "${LOCK_STEPS[@]}"; do
    echo "    -> $lock"
    "$UV_BIN" pip install \
        "${uv_common[@]}" \
        "${binary_args[@]}" \
        --require-hashes \
        -r "$LOCK_DIR/$lock" \
        >/dev/null || die "installing $lock failed"
done

# The two local seed plugins, exactly as install_python_stack.py installs them.
# Pure Python, so building them here yields the same py3-none-any wheel a Mac would.
for plugin in "${DD_PLUGINS[@]}"; do
    echo "    -> local plugin $(basename "$plugin")"
    "$UV_BIN" pip install \
        "${uv_common[@]}" \
        "$plugin" \
        >/dev/null || die "installing $(basename "$plugin") failed"
done

# The pinned Diffusers revision, last, for install_python_stack.py's reason: no
# earlier step may re-resolve diffusers back to a release. Its whole closure is
# already installed above and pinned by the locks, and the shared --no-deps keeps
# the archive's own metadata from re-resolving transformers and friends unhashed.
DIFFUSERS_URL="$(
    sed -n 's/^diffusers @ \([^ ;]*\).*/\1/p' "$REQ_DIR/diffusers-pin.txt" | head -n 1
)"
DIFFUSERS_SHA256=""
DIFFUSERS_STATUS="installed"
if [ "$SKIP_DIFFUSERS" = "1" ]; then
    DIFFUSERS_STATUS="skipped (--skip-diffusers-pin)"
    echo "    -> diffusers pin SKIPPED; this payload is incomplete"
else
    [ -n "$DIFFUSERS_URL" ] || die "could not read the diffusers archive URL out of diffusers-pin.txt"
    echo "    -> diffusers pin (unhashed URL archive; sha256 recorded in the manifest)"
    ARCHIVE="$WORK_DIR/diffusers-pin.zip"
    "$PYTHON_BIN" - "$DIFFUSERS_URL" "$ARCHIVE" <<'PY' || die "could not fetch the diffusers pin"
import shutil
import sys
import urllib.request

url, dest = sys.argv[1:3]
if not url.startswith("https://"):
    sys.exit(f"error: refusing a non-https diffusers pin: {url}")
request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-macos-runtime-build"})
with urllib.request.urlopen(request, timeout = 600) as response:  # noqa: S310
    with open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle, length = 1024 * 1024)
PY
    DIFFUSERS_SHA256="$("$PYTHON_BIN" -c \
        "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
        "$ARCHIVE")"
    "$UV_BIN" pip install \
        "${uv_common[@]}" \
        "diffusers @ file://$ARCHIVE" \
        >/dev/null || die "installing the diffusers pin failed"
fi

# The single-env metadata relaxation install_python_stack.py runs as its own step.
# Pointed at the payload rather than at this machine's interpreter: importlib.metadata
# reads sys.path, so PYTHONPATH is all it takes to make it operate on site-packages/.
echo "    -> patching single-env metadata"
PYTHONPATH="$SITE" "$PYTHON_BIN" "$REQ_DIR/single-env/patch_metadata.py" >/dev/null \
    || die "patch_metadata.py failed"

# ─────────────────────────── 3. this checkout's own unsloth, over the index copy

echo
echo "==> 3/7 overlaying this checkout's unsloth (the payload ships the code being built)"

# WHY THIS STEP EXISTS
#
# The lock installed `unsloth` from PyPI, because the closure had to be resolved
# against some version of it. Keeping that wheel would ship PUBLISHED backend code
# inside an app built from THIS commit -- and the two are released together, so the
# only correct answer is the checkout. The failure that made this concrete: the
# bundled-runtime seam (unsloth_cli/_bundled_runtime.py, and _serves_backend_in_process
# in unsloth_cli/commands/studio.py) exists only here, so the index copy left the CLI
# taking the managed-venv path and exiting with "Unsloth Studio not set up. Run
# install.sh first." after the Rust preflight had already reported Ready.
#
# WHAT IS AND IS NOT LOCAL
#
# Only `unsloth` itself. unsloth-zoo is a separate upstream project and stays exactly
# as darwin-arm64-bundle.lock.txt pins it, as does every other distribution: the
# install below passes --no-deps, so the local wheel cannot perturb the resolved
# closure even though its own metadata carries unpinned requirements.
#
# ORDER, AND WHY IT IS AN OVERLAY RATHER THAN A SUBSTITUTION
#
# The lock install has to come first -- it is what puts the closure in place -- so this
# is necessarily a replacement of files already on disk. `uv pip install --target` would
# happily write the new wheel's files OVER the old ones and leave anything the old wheel
# had and the new one does not, which is the difference between replacing a package and
# landing beside it. So the index copy is removed through its own RECORD first, and the
# result is checked afterwards: no file may survive under the trees the local wheel owns
# that the local wheel does not list.
LOCAL_WHEEL_DIR="$WORK_DIR/local-wheel"
LOCAL_BUILD_BASE="$WORK_DIR/local-wheel-build-base"
LOCAL_PROVENANCE="$WORK_DIR/local_provenance.json"
rm -rf "$LOCAL_WHEEL_DIR" "$LOCAL_BUILD_BASE"
mkdir -p "$LOCAL_WHEEL_DIR" "$LOCAL_BUILD_BASE"

# The version the lock installed, so the manifest can say what was replaced. Read from
# the lock rather than written here: this must follow the pin, not shadow it.
LOCK_UNSLOTH_VERSION="$(
    sed -n 's/^unsloth==\([^ \\]*\).*/\1/p' "$BUNDLE_LOCK" | head -n 1
)"
[ -n "$LOCK_UNSLOTH_VERSION" ] || die "could not read the unsloth pin out of $BUNDLE_LOCK"

# setuptools copies sources into <build_base>/lib and never prunes that directory, so a
# build/lib left in the checkout by an earlier build re-ships files the working tree no
# longer has (measured: a stray build/lib/unsloth_cli/_stale_probe.py landed in the
# wheel). DIST_EXTRA_CONFIG is setuptools' own hook for an extra distutils config file,
# which lets the build base live in this script's work dir instead -- so the wheel is a
# function of the working tree alone. egg_base goes with it for the milder version of
# the same courtesy: a payload build should not leave anything behind in somebody's
# checkout, not even a regenerated .egg-info. Verified: the two configurations produce
# the same 2606-entry wheel.
DIST_CFG="$WORK_DIR/dist-extra.cfg"
LOCAL_EGG_BASE="$WORK_DIR/local-wheel-egg-info"
mkdir -p "$LOCAL_EGG_BASE"
printf '[build]\nbuild_base = %s\n[egg_info]\negg_base = %s\n' \
    "$LOCAL_BUILD_BASE" "$LOCAL_EGG_BASE" > "$DIST_CFG"

# The build backend is the one pyproject.toml pins (setuptools==80.9.0 +
# setuptools-scm==9.2.0), fetched into uv's isolated build environment exactly as any
# `pip install .` would fetch it. Nothing from that environment reaches the payload --
# only the wheel it produces does. Output goes to a log rather than the console: a
# setuptools wheel build narrates every one of ~2600 files onto stderr, and it is only
# interesting when the build fails.
echo "    -> building the wheel from $REPO_ROOT"
LOCAL_WHEEL_LOG="$WORK_DIR/local-wheel-build.log"
if ! DIST_EXTRA_CONFIG="$DIST_CFG" "$UV_BIN" build \
        --wheel \
        --out-dir "$LOCAL_WHEEL_DIR" \
        "$REPO_ROOT" \
        >"$LOCAL_WHEEL_LOG" 2>&1; then
    tail -n 40 "$LOCAL_WHEEL_LOG" >&2 || true
    die "building the local unsloth wheel failed (full log: $LOCAL_WHEEL_LOG)"
fi

# Validate the artifact before anything is deleted on the strength of it, and record
# what it is. The cross-build claim is checked rather than assumed: a py3-none-any
# wheel with no compiled objects in it is the same wheel a Mac would have produced, and
# that is the ONLY reason building the payload's own backend on a Linux runner is sound.
LOCAL_WHEEL="$(
    "$PYTHON_BIN" - "$LOCAL_WHEEL_DIR" "$LOCAL_PROVENANCE" "$REPO_ROOT" \
        "$LOCK_UNSLOTH_VERSION" "${DD_PLUGINS[@]}" <<'PY'
"""Check the locally built wheel, and write the provenance record for the manifest.

Prints the wheel path on stdout; the caller installs exactly that file.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile

wheel_dir = pathlib.Path(sys.argv[1])
provenance_path = pathlib.Path(sys.argv[2])
repo_root = pathlib.Path(sys.argv[3])
lock_version = sys.argv[4]
plugin_dirs = [pathlib.Path(part) for part in sys.argv[5:]]

wheels = sorted(wheel_dir.glob("*.whl"))
if len(wheels) != 1:
    sys.exit(f"error: expected exactly one locally built wheel, found {[w.name for w in wheels]}")
wheel = wheels[0]

name, version, build_tag_and_tags = wheel.name[: -len(".whl")].split("-", 2)
if name != "unsloth":
    sys.exit(f"error: the local build produced {name!r}, not 'unsloth': {wheel.name}")
if not build_tag_and_tags.endswith("py3-none-any"):
    # A platform tag here would mean the wheel was compiled for THIS host, and the
    # payload is for macOS arm64. The package is pure Python; if that ever stops being
    # true the payload's backend has to be built on a Mac and this must stop pretending
    # otherwise.
    sys.exit(
        f"error: the local wheel is tagged {build_tag_and_tags!r}, not py3-none-any. "
        f"A platform-tagged build cannot be produced for macOS arm64 from this host."
    )

with zipfile.ZipFile(wheel) as archive:
    entries = archive.namelist()
    native = sorted(
        entry for entry in entries
        if entry.endswith((".so", ".dylib", ".pyd", ".dll", ".a", ".lib"))
    )
    if native:
        sys.exit(
            f"error: the local wheel carries compiled objects, so it is not portable: {native[:5]}"
        )
    wheel_metadata = f"{name}-{version}.dist-info/WHEEL"
    if wheel_metadata not in entries:
        sys.exit(f"error: {wheel.name} has no {wheel_metadata}")
    tags = [
        line.partition(":")[2].strip()
        for line in archive.read(wheel_metadata).decode("utf-8").splitlines()
        if line.startswith("Tag:")
    ]
    if tags != ["py3-none-any"]:
        sys.exit(f"error: {wheel_metadata} declares tags {tags}, expected ['py3-none-any']")
    # The seam, checked in the artifact itself. Everything after this point is about
    # getting these bytes into site-packages; if they are not in the wheel there is
    # nothing to get there, and the reason is upstream of this script.
    seam_module = "unsloth_cli/_bundled_runtime.py"
    if seam_module not in entries:
        sys.exit(f"error: {wheel.name} is missing {seam_module}; the bundled runtime has no seam")
    studio_cli = archive.read("unsloth_cli/commands/studio.py").decode("utf-8", "replace")
    if "_serves_backend_in_process" not in studio_cli:
        sys.exit(
            "error: unsloth_cli/commands/studio.py in the local wheel has no "
            "_serves_backend_in_process; the bundled CLI would look for a managed venv "
            "that a bundle never has"
        )


def git(*args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output = True,
            text = True,
            timeout = 30,
            check = False,
        )
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
commit = git("rev-parse", "HEAD") or None
dirty = bool(git("status", "--porcelain"))
if commit is None:
    # Two things break at once without git metadata, and only one of them is visible.
    # The obvious one: this distribution is identified by a commit and nothing else, so
    # a payload that cannot name one cannot be audited. The quiet one: pyproject builds
    # its package data through setuptools-scm's file finder, which asks `git ls-files`,
    # so a git-less build produces a wheel missing the tracked data files -- silently,
    # and only noticed as a missing asset at runtime.
    sys.exit(
        f"error: {repo_root} is not a usable git checkout, so the payload's `unsloth` "
        f"could be neither identified nor completely built. Assemble the payload from a "
        f"git checkout (both CI legs check one out)."
    )

record = {
    "comment": (
        "Not every distribution in site-packages came from an index. The entries below "
        "were built from the checkout this build ran in, so they are pinned by a commit "
        "rather than by a hash in a lock file -- for `unsloth` that is the point: the "
        "desktop app and the Python backend are released together, and a payload holding "
        "a published wheel would be shipping code that was never built with the app "
        "around it. Everything NOT listed here was installed from the locks under "
        "--require-hashes."
    ),
    "index_verified": False,
    "distributions": [
        {
            "name": "unsloth",
            "version": version,
            "provenance": "local-checkout",
            "source": "the working tree this build ran in (scripts/build_macos_runtime.sh builds a wheel from it)",
            "index_verified": False,
            "wheel": wheel.name,
            "wheel_sha256": digest,
            "wheel_tag": "py3-none-any",
            "repo_commit": commit,
            "worktree_dirty": dirty,
            "replaced_index_version": lock_version,
            "replaced_lock_entry": f"darwin-arm64-bundle.lock.txt: unsloth=={lock_version}",
            "note": (
                "The lock still pins unsloth, because the closure has to be resolved "
                "against a version of it; the payload then replaces those files with "
                "this wheel (installed --no-deps, so the resolved closure is untouched). "
                "unsloth-zoo is NOT local: it is a separate upstream project and stays "
                "at the version the lock pins."
            ),
        },
        *(
            {
                "name": plugin.name,
                "provenance": "local-checkout",
                "source": plugin.relative_to(repo_root).as_posix(),
                "index_verified": False,
                "repo_commit": commit,
                "worktree_dirty": dirty,
                "note": "A data-designer seed plugin, installed from the checkout exactly as studio/install_python_stack.py installs it.",
            }
            for plugin in plugin_dirs
        ),
    ],
}
provenance_path.write_text(json.dumps(record, indent = 2, sort_keys = True) + "\n", encoding = "utf-8")

print(str(wheel))
PY
)" || die "the locally built wheel was rejected"

echo "    -> built $(basename "$LOCAL_WHEEL")"

# Remove the lock-installed copy through its own RECORD, so the local wheel replaces it
# instead of being written over the top of it. The two versions can even be identical --
# the lock pins whatever was published last -- and identical version strings are exactly
# the case where "landed beside" is invisible.
echo "    -> removing the index-installed unsloth==$LOCK_UNSLOTH_VERSION"
"$PYTHON_BIN" - "$SITE" <<'PY' || die "removing the index-installed unsloth failed"
"""Delete the installed `unsloth` distribution, file by file, from its RECORD.

Only paths that resolve INSIDE site-packages are touched; anything else is reported and
left alone rather than followed. Nothing here globs: the distribution's own manifest is
the list, so a file the wheel put there is removed and a file it did not is not.
"""

import pathlib
import sys

site = pathlib.Path(sys.argv[1]).resolve()


def distribution_name(info: pathlib.Path) -> str:
    """`unsloth-2026.8.18.dist-info` -> `unsloth`, and `unsloth_zoo-...` -> `unsloth_zoo`."""
    return info.name[: -len(".dist-info")].rsplit("-", 1)[0]


infos = [
    info for info in sorted(site.glob("unsloth-*.dist-info"))
    if info.is_dir() and distribution_name(info) == "unsloth"
]
if len(infos) != 1:
    sys.exit(
        f"error: expected exactly one installed unsloth dist-info in {site}, "
        f"found {[info.name for info in infos]}"
    )
info = infos[0]

record = info / "RECORD"
if not record.is_file():
    sys.exit(f"error: {record} is missing, so there is no manifest to remove {info.name} by")

removed = outside = 0
directories: set[pathlib.Path] = set()
for line in record.read_text(encoding = "utf-8").splitlines():
    entry = line.split(",", 1)[0].strip()
    if not entry:
        continue
    target = (site / entry).resolve()
    if target != site and site not in target.parents:
        outside += 1
        print(f"    left alone (outside site-packages): {entry}")
        continue
    if target.is_file() or target.is_symlink():
        target.unlink()
        removed += 1
        directories.add(target.parent)

# The dist-info itself, whatever the RECORD did or did not list inside it (uv writes
# direct_url.json there, which a wheel's own RECORD cannot mention).
for stray in sorted(info.rglob("*"), reverse = True):
    if stray.is_file() or stray.is_symlink():
        stray.unlink()
        removed += 1
    elif stray.is_dir():
        stray.rmdir()
if info.is_dir():
    info.rmdir()

# Now-empty directories, deepest first, and only inside site-packages.
for directory in sorted(directories, key = lambda path: len(path.parts), reverse = True):
    current = directory
    while current != site and site in current.parents:
        try:
            next(current.iterdir())
        except StopIteration:
            current.rmdir()
            current = current.parent
            continue
        except OSError:
            break
        break

print(f"removed {removed} file(s) of {info.name}" + (f"; {outside} outside" if outside else ""))
PY

"$UV_BIN" pip install \
    "${uv_common[@]}" \
    "$LOCAL_WHEEL" \
    >/dev/null || die "installing the local unsloth wheel failed"

# The assertion that makes the original failure unshippable. It is here, in the
# assembly, and repeated after pruning in step 7 -- a payload whose unsloth_cli does not
# carry the bundled-runtime seam is a payload whose app cannot start, and it must cost a
# build rather than a user's evening.
echo "    -> asserting the payload's unsloth_cli carries the bundled-runtime seam"
"$PYTHON_BIN" - "$SITE" "$LOCAL_WHEEL" "$LOCAL_PROVENANCE" <<'PY' \
    || die "the overlaid unsloth is not the one this checkout builds"
"""Prove the overlay landed: the seam is present, and nothing of the index copy is left.

Two different claims, both of which the earlier failure would have flunked:

  1. the files whose absence broke the app are in site-packages;
  2. every file under the trees the local wheel owns is a file the local wheel lists,
     so the install replaced the index copy rather than landing beside it.
"""

import json
import pathlib
import sys
import zipfile

site = pathlib.Path(sys.argv[1]).resolve()
wheel = pathlib.Path(sys.argv[2])
provenance = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding = "utf-8"))
expected_version = next(
    entry["version"] for entry in provenance["distributions"] if entry["name"] == "unsloth"
)

with zipfile.ZipFile(wheel) as archive:
    listed = {
        entry for entry in archive.namelist()
        if not entry.endswith("/") and not entry.startswith(f"unsloth-{expected_version}.data/")
    }

# 1. the seam, in the payload rather than in the wheel.
seam = site / "unsloth_cli" / "_bundled_runtime.py"
if not seam.is_file():
    sys.exit(
        "error: site-packages/unsloth_cli/_bundled_runtime.py is missing. The bundled CLI "
        "cannot tell that it IS the bundled runtime, so `unsloth studio` exits with "
        "'Unsloth Studio not set up. Run install.sh first.' on a machine where install.sh "
        "never runs. This is what installing the published wheel into the payload did."
    )
studio_cli = site / "unsloth_cli" / "commands" / "studio.py"
if not studio_cli.is_file():
    sys.exit("error: site-packages/unsloth_cli/commands/studio.py is missing")
text = studio_cli.read_text(encoding = "utf-8", errors = "replace")
for symbol in ("_serves_backend_in_process", "_bundled_runtime"):
    if symbol not in text:
        sys.exit(
            f"error: site-packages/unsloth_cli/commands/studio.py does not reference "
            f"{symbol}; the payload's CLI is not the one this checkout builds"
        )
# The backend's own mirror of the seam: the studio backend runs from interpreters that
# have no unsloth_cli on sys.path, so it carries its own copy (see
# studio/backend/utils/bundled_runtime.py). A payload with one and not the other is half
# converted.
mirror = site / "studio" / "backend" / "utils" / "bundled_runtime.py"
if not mirror.is_file():
    sys.exit("error: site-packages/studio/backend/utils/bundled_runtime.py is missing")

# 2. exactly one unsloth, at the local version, and no orphans under its trees.
infos = sorted(
    info for info in site.glob("unsloth-*.dist-info")
    if info.is_dir() and info.name[: -len(".dist-info")].rsplit("-", 1)[0] == "unsloth"
)
if [info.name for info in infos] != [f"unsloth-{expected_version}.dist-info"]:
    sys.exit(
        f"error: expected exactly unsloth-{expected_version}.dist-info in site-packages, "
        f"found {[info.name for info in infos]}"
    )

# The dist-info is excluded: the installer writes INSTALLER / direct_url.json into it,
# which a wheel's own RECORD cannot list, and its identity is asserted above anyway.
roots = sorted({
    entry.split("/", 1)[0] for entry in listed
    if "/" in entry and not entry.split("/", 1)[0].endswith(".dist-info")
})
orphans = []
for root in roots:
    base = site / root
    if not base.is_dir():
        sys.exit(f"error: the local wheel ships {root}/ but site-packages has no such directory")
    for path in base.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(site).as_posix()
        if relative not in listed:
            orphans.append(relative)
if orphans:
    sys.exit(
        "error: site-packages holds files under the local wheel's own trees that the "
        "local wheel does not ship, so the index copy was written over rather than "
        f"replaced: {sorted(orphans)[:10]} ({len(orphans)} total)"
    )

print(f"overlay ok: unsloth {expected_version} from this checkout, {len(listed)} files, no orphans")
PY

# ───────────────────────────────── 4. relocatable console scripts

echo
echo "==> 4/7 rewriting console-script shebangs to be relocation-proof"
"$PYTHON_BIN" - "$SITE" "$PYTHON_MINOR" <<'PY' || die "shebang rewrite failed"
"""Make site-packages/bin/* survive being moved.

uv writes each console script with an absolute shebang naming the interpreter that
performed the install -- on a CI runner, a path that does not exist on any user's
Mac. python-build-standalone solved the same problem for its own bin/pip and
bin/idle3 with a two-line sh header that resolves the interpreter relative to the
script, so use exactly that mechanism rather than a new one. `realpath` is present
on every macOS the bundle targets.

The entry point is read out of the script uv already generated, so nothing is
re-derived from package metadata and a script whose shape is not recognised is left
alone and reported rather than silently mangled.
"""

import pathlib
import re
import sys

site = pathlib.Path(sys.argv[1])
minor = sys.argv[2]
bin_dir = site / "bin"
if not bin_dir.is_dir():
    print("no site-packages/bin; nothing to rewrite")
    raise SystemExit(0)

# site-packages/bin/<script> -> ../../python/bin/python<minor>
HEADER = (
    "#!/bin/sh\n"
    "'''exec' \"$(dirname -- \"$(realpath -- \"$0\")\")/../../python/bin/python{minor}\" \"$0\" \"$@\"\n"
    "' '''\n"
)

rewritten, skipped = [], []
for script in sorted(bin_dir.iterdir()):
    if not script.is_file() or script.is_symlink():
        continue
    try:
        text = script.read_text(encoding = "utf-8")
    except (UnicodeDecodeError, OSError):
        skipped.append(script.name)
        continue
    if not text.startswith("#!"):
        continue
    first, _, rest = text.partition("\n")
    if not re.match(r"^#!\s*\S*python", first):
        skipped.append(script.name)
        continue
    mode = script.stat().st_mode
    script.write_text(HEADER.format(minor = minor) + rest, encoding = "utf-8")
    script.chmod(mode)
    rewritten.append(script.name)

print(f"rewrote {len(rewritten)} console script(s)")
if skipped:
    print(f"left alone (unrecognised shebang): {sorted(skipped)}")
PY

# ────────────────────────────────────────── 5. the OXC validator node_modules

echo
echo "==> 5/7 prefetching the OXC validator node_modules for darwin/arm64"
OXC_STAGE="$WORK_DIR/oxc"
rm -rf "$OXC_STAGE"
mkdir -p "$OXC_STAGE"
cp "$OXC_DIR/package.json" "$OXC_DIR/package-lock.json" "$OXC_STAGE/"

# --os/--cpu, not the runner's own platform: oxc-parser's parser is a native N-API
# binding shipped as one optional dependency per platform, so a Linux runner would
# otherwise install @oxc-parser/binding-linux-*-gnu and the validator would fail on
# every Mac. --ignore-scripts because nothing in this closure needs a lifecycle
# script and a bundle build must not run one. `npm ci` (not install) so the
# committed lockfile, with its integrity digests, is the only thing that decides
# what is fetched.
(
    cd "$OXC_STAGE"
    npm ci --ignore-scripts --no-fund --no-audit --os=darwin --cpu=arm64 >/dev/null
) || die "npm ci for the OXC validator failed"

[ -d "$OXC_STAGE/node_modules" ] || die "npm ci produced no node_modules"
mv "$OXC_STAGE/node_modules" "$STAGE/oxc-node-modules"

# Fail closed on the thing --os/--cpu exists to get right. A missing darwin-arm64
# binding is the exact failure mode that would otherwise ship and only show up as a
# broken data-recipe validator on a user's Mac.
if ! ls -d "$STAGE/oxc-node-modules/@oxc-parser/binding-darwin-arm64" >/dev/null 2>&1; then
    die "oxc-node-modules has no @oxc-parser/binding-darwin-arm64; npm resolved for the wrong platform"
fi
for stray in "$STAGE/oxc-node-modules"/@oxc-parser/binding-linux-* \
             "$STAGE/oxc-node-modules"/@oxc-parser/binding-win32-*; do
    [ -e "$stray" ] && die "oxc-node-modules contains a foreign binding: $stray"
done

# ─────────────────────────────────────────────────────────────── 6. pruning

echo
echo "==> 6/7 pruning"
PRUNE_REPORT="$WORK_DIR/prune.json"
"$PYTHON_BIN" - "$PINS_JSON" "$STAGE" "$PRUNE_REPORT" <<'PY' || die "pruning failed"
"""Delete exactly what studio/macos_runtime_pins.json's prune list names.

The list is the contract, not this code: every deletion is one of its entries, a
non-optional entry that matches nothing is a hard error (the payload changed shape
and nobody noticed), and an entry that would remove a required path is refused
outright so the list cannot quietly grow into something the app needs.
"""

import json
import pathlib
import shutil
import sys

pins_path, stage_path, report_path = (pathlib.Path(part) for part in sys.argv[1:4])
pins = json.loads(pins_path.read_text(encoding = "utf-8"))
stage = stage_path.resolve()
layout = pins["layout"]
required = {
    *(str(part) for part in layout["required_dirs"]),
    *(str(part) for part in layout["required_files"]),
}


def size_of(path: pathlib.Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def guard(target: pathlib.Path) -> None:
    relative = target.relative_to(stage).as_posix()
    for keep in required:
        if relative == keep or keep.startswith(relative + "/"):
            sys.exit(
                f"error: prune entry would remove {relative}, which the layout contract "
                f"requires (or contains {keep}). Refusing."
            )


removed, freed = [], 0
for entry in pins["prune"]["entries"]:
    kind = entry["kind"]
    pattern = entry["path"]
    optional = bool(entry.get("optional"))
    if kind == "dir-name":
        matches = [path for path in stage.rglob(pattern) if path.is_dir() and not path.is_symlink()]
    elif kind == "glob":
        matches = sorted(stage.glob(pattern))
    else:
        candidate = stage / pattern
        matches = [candidate] if candidate.exists() else []
    if not matches:
        if optional:
            continue
        sys.exit(
            f"error: prune entry '{pattern}' matched nothing. The payload's shape "
            f"changed; update studio/macos_runtime_pins.json rather than ignoring this."
        )
    for match in matches:
        guard(match)
        freed += size_of(match)
        if match.is_dir() and not match.is_symlink():
            shutil.rmtree(match)
        else:
            match.unlink()
        removed.append(match.relative_to(stage).as_posix())

# Loose .pyc files outside a __pycache__ directory, for the same reason.
for stray in stage.rglob("*.pyc"):
    if stray.is_file():
        freed += stray.stat().st_size
        stray.unlink()

report = {"removed": sorted(removed), "bytes_freed": freed}
report_path.write_text(json.dumps(report, indent = 2) + "\n", encoding = "utf-8")
print(f"pruned {len(removed)} path(s), freed {freed / (1024 * 1024):.1f} MiB")
PY

# ──────────────────────────────────────── 7. the manifest, and the contract

echo
echo "==> 7/7 BUNDLE_MANIFEST.json and the layout contract"
"$PYTHON_BIN" - \
    "$PINS_JSON" "$STAGE" "$PREBUILT_REPORT" "$PRUNE_REPORT" "$LOCK_DIR" \
    "$GENERATOR_VERSION" "$UV_ACTUAL" "$DIFFUSERS_URL" "$DIFFUSERS_SHA256" \
    "$DIFFUSERS_STATUS" "$REPO_ROOT" "$SD_CPP_STATUS" "$LOCAL_PROVENANCE" \
    <<'PY' || die "manifest generation failed"
"""Write runtime/BUNDLE_MANIFEST.json, then assert the layout contract holds.

The manifest exists so that a .dmg in somebody's Downloads folder can be audited
without rebuilding it: every component's version, source URL and sha256, the exact
locks that produced site-packages, what was pruned, and what the build could not do.

It also has to be honest about what it CANNOT anchor to a digest in this tree.
`local_provenance` names every distribution that came from the checkout instead of from
an index -- `unsloth` itself and the two data-designer seed plugins -- with the commit
that identifies them, so nobody reading `locks` concludes the whole payload is
index-verified.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import time

(
    pins_path,
    stage_path,
    prebuilt_report,
    prune_report,
    lock_dir,
    generator_version,
    uv_version,
    diffusers_url,
    diffusers_sha256,
    diffusers_status,
    repo_root,
    sd_cpp_status,
    local_provenance_path,
) = sys.argv[1:14]

pins = json.loads(pathlib.Path(pins_path).read_text(encoding = "utf-8"))
stage = pathlib.Path(stage_path).resolve()
prebuilts = json.loads(pathlib.Path(prebuilt_report).read_text(encoding = "utf-8"))
prune = json.loads(pathlib.Path(prune_report).read_text(encoding = "utf-8"))
# What in this payload came from the checkout rather than from an index. Recorded rather
# than glossed over: the locks below are hash-verified and `unsloth` is not, because it
# cannot be -- it is the code being released.
local_provenance = json.loads(pathlib.Path(local_provenance_path).read_text(encoding = "utf-8"))
local_unsloth = next(
    entry for entry in local_provenance["distributions"] if entry["name"] == "unsloth"
)
layout = pins["layout"]


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_bytes(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output = True,
            text = True,
            timeout = 20,
            check = False,
        )
    except OSError:
        return None
    value = out.stdout.strip()
    return value or None


# Every distribution actually present, read from the .dist-info directories rather
# than from the locks: this records what IS in the payload, not what was asked for.
distributions = {}
for info in sorted(stage.glob("site-packages/*.dist-info")):
    metadata = info / "METADATA"
    name = version = None
    if metadata.is_file():
        for line in metadata.read_text(encoding = "utf-8", errors = "replace").splitlines():
            if line.startswith("Name: ") and name is None:
                name = line[6:].strip()
            elif line.startswith("Version: ") and version is None:
                version = line[9:].strip()
            if name and version:
                break
    if not name:
        name = info.name.rsplit("-", 2)[0]
    distributions[name] = version or ""

locks = {}
for lock in sorted(pathlib.Path(lock_dir).glob("*.lock.txt")):
    text = lock.read_text(encoding = "utf-8")
    locks[lock.name] = {
        "sha256": sha256_of(lock),
        "packages": sum(1 for line in text.splitlines() if line[:1].isalnum()),
        "hashes": text.count("--hash=sha256:"),
    }

components = dict(prebuilts)
components["diffusers_pin"] = {
    "url": diffusers_url,
    "archive_sha256": diffusers_sha256 or None,
    "status": diffusers_status,
    "note": (
        "A source archive, so it carries no index hash; scripts/gen_python_locks.sh "
        "excludes diffusers-pin.txt for the same reason. The digest above is of the "
        "bytes this build installed, recorded so the payload can be audited."
    ),
}
# The one distribution in site-packages that did not come from an index, recorded as a
# component in its own right so an auditor reading this file sees it beside the archives
# that DO carry a digest from a lock.
components["unsloth_local_wheel"] = dict(local_unsloth)
components["oxc_node_modules"] = {
    "source": "studio/backend/core/data_recipe/oxc-validator/package-lock.json",
    "package_lock_sha256": sha256_of(
        pathlib.Path(repo_root)
        / "studio/backend/core/data_recipe/oxc-validator/package-lock.json"
    ),
    "npm_flags": "ci --ignore-scripts --os=darwin --cpu=arm64",
}

sizes = {
    name: tree_bytes(stage / name)
    for name in ("python", "site-packages", "llama.cpp", "whisper.cpp",
                 "stable-diffusion.cpp", "node", "oxc-node-modules")
    if (stage / name).exists()
}

# What this build could not do, so a payload assembled with a skip flag can never be
# mistaken for a shippable one: the dev-build workflow refuses a non-empty list.
incomplete = []
if diffusers_status != "installed":
    incomplete.append(f"diffusers_pin: {diffusers_status}")
if sd_cpp_status != "installed":
    incomplete.append(f"sd_cpp: {sd_cpp_status}")

manifest = {
    "schema_version": 1,
    "generator": "scripts/build_macos_runtime.sh",
    "generator_version": generator_version,
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repo_commit": git_commit(),
    "target": {
        "rust_target": pins["target"]["rust_target"],
        "uv_python_platform": pins["target"]["uv_python_platform"],
        "macos_deployment_target": pins["target"]["macos_deployment_target"],
    },
    "python": {
        "version": pins["components"]["cpython"]["python_version"],
        "minor": pins["components"]["cpython"]["python_minor"],
        "interpreter": "python/bin/python3",
        "site_packages": "site-packages",
        "venv": False,
        "venv_note": (
            "No venv on purpose: a venv bakes absolute paths into pyvenv.cfg and into "
            "every console-script shebang and does not survive relocation. The app sets "
            "PYTHONHOME/PYTHONPATH into this directory instead."
        ),
    },
    "built_with": {"uv": uv_version, "generator_version": generator_version},
    "components": components,
    "locks": locks,
    "locks_note": (
        "Every distribution in site-packages was installed from these locks under "
        "--require-hashes, with ONE deliberate exception: `unsloth` itself, which is "
        "built from the checkout the payload was assembled in (see local_provenance). "
        "The lock's unsloth pin is what the closure was resolved against, not what the "
        "payload ships."
    ),
    "local_provenance": local_provenance,
    "distributions": distributions,
    "distribution_count": len(distributions),
    "pruned": prune,
    "sizes_bytes": sizes,
    "total_bytes": sum(sizes.values()),
    "layout": {
        "required_dirs": layout["required_dirs"],
        "required_files": layout["required_files"],
    },
    "incomplete": incomplete,
}

(stage / "BUNDLE_MANIFEST.json").write_text(
    json.dumps(manifest, indent = 2, sort_keys = True) + "\n", encoding = "utf-8"
)

# The contract, asserted here rather than trusted. A payload that does not satisfy
# it must not reach a .dmg, because the Rust side resolves these exact paths.
#
# A component the invocation EXPLICITLY skipped is excluded from the assertion, by
# name, and only that component: --skip-sd-cpp is for a fast local iteration, and a
# build that used it is already recorded as incomplete above. Everything else stays
# mandatory, so a component that was meant to be fetched and silently is not still
# stops the build here rather than on somebody's Mac.
skipped_prefixes = () if sd_cpp_status == "installed" else ("stable-diffusion.cpp",)


def expected(paths):
    return [
        path
        for path in paths
        if not any(path == prefix or path.startswith(prefix + "/") for prefix in skipped_prefixes)
    ]


missing = []
for name in expected(layout["required_dirs"]):
    if not (stage / name).is_dir():
        missing.append(f"dir {name}")
for name in expected(layout["required_files"]):
    if not (stage / name).exists():
        missing.append(f"file {name}")
for name in expected(layout["required_executables"]):
    path = stage / name
    resolved = path.resolve()
    if not resolved.is_file():
        missing.append(f"executable {name} (dangling)")
    elif not resolved.stat().st_mode & 0o111:
        missing.append(f"executable {name} (not executable)")
if missing:
    sys.exit("error: the assembled payload violates the layout contract: " + "; ".join(missing))

# Every import the app needs must be present as a top-level module or package in
# site-packages. This is a static check, not an import: the payload holds macOS arm64
# extension modules, so a Linux build host cannot import them -- the workflow's DMG
# verification step runs the real imports on macOS.
site = stage / "site-packages"
absent = []
for module in layout["required_imports"]:
    if (site / module).is_dir() or list(site.glob(f"{module}.*")):
        continue
    absent.append(module)
if absent:
    sys.exit(f"error: site-packages is missing required top-level modules: {sorted(absent)}")

# THE SEAM, ASSERTED ON THE FINISHED PAYLOAD -- after pruning, after everything.
#
# The app launches the bundled interpreter and asks the CLI to serve the backend
# in-process. The CLI can only agree if it carries unsloth_cli/_bundled_runtime and the
# _serves_backend_in_process branch that consults it; without them it looks for
# <STUDIO_HOME>/unsloth_studio/bin/python, does not find one (there never will be one on
# a machine that installed the app by copying it), prints "Unsloth Studio not set up.
# Run install.sh first." and exits 1 -- with the Rust preflight having already reported
# Ready. That shipped once, because the payload installed `unsloth` from the index. Step
# 3 fixes the cause; this is the check that a payload without the fix cannot leave the
# build, wherever the regression comes from next time.
seam_failures = []
if not (site / "unsloth_cli" / "_bundled_runtime.py").is_file():
    seam_failures.append("site-packages/unsloth_cli/_bundled_runtime.py is missing")
if not (site / "studio" / "backend" / "utils" / "bundled_runtime.py").is_file():
    seam_failures.append("site-packages/studio/backend/utils/bundled_runtime.py is missing")
studio_cli = site / "unsloth_cli" / "commands" / "studio.py"
if not studio_cli.is_file():
    seam_failures.append("site-packages/unsloth_cli/commands/studio.py is missing")
elif "_serves_backend_in_process" not in studio_cli.read_text(
    encoding = "utf-8", errors = "replace"
):
    seam_failures.append(
        "site-packages/unsloth_cli/commands/studio.py has no _serves_backend_in_process"
    )
installed_unsloth = distributions.get("unsloth")
if installed_unsloth != local_unsloth["version"]:
    seam_failures.append(
        f"site-packages holds unsloth {installed_unsloth!r}, but the wheel built from "
        f"this checkout is {local_unsloth['version']!r}"
    )
if seam_failures:
    sys.exit(
        "error: the payload's unsloth_cli does not carry the bundled-runtime seam, so the "
        "app would start the backend and be told to run install.sh: "
        + "; ".join(seam_failures)
    )

print(f"manifest: {len(distributions)} distributions, {manifest['total_bytes'] / (1024 ** 3):.2f} GiB")
for name, value in sorted(sizes.items()):
    print(f"  {name:<18} {value / (1024 * 1024):9.1f} MiB")
commit = local_unsloth["repo_commit"] or "unknown commit"
print(
    f"  unsloth {local_unsloth['version']} is THIS checkout ({commit[:12]}"
    + (", dirty worktree" if local_unsloth["worktree_dirty"] else "")
    + f"), not the index's {local_unsloth['replaced_index_version']}"
)
PY

# ───────────────────────────────────────────────────────── swap into place

mkdir -p "$(dirname "$OUT_DIR")"
rm -rf "$OUT_DIR"
mv "$STAGE" "$OUT_DIR"

echo
echo "==> payload assembled at $OUT_DIR"
du -sh "$OUT_DIR" 2>/dev/null || true
