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
#                          (see scripts/fetch_macos_prebuilts.py).
#   OXC node_modules       npm ci against the committed package-lock.json, whose
#                          every entry carries an integrity digest.
#   The two local          Built from this checkout. Their identity is the commit.
#   data-designer plugins
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
#   bash scripts/build_macos_runtime.sh --out <dir> --with-sd-cpp
#   UV=/path/to/uv bash scripts/build_macos_runtime.sh --out <dir>
#
# --skip-diffusers-pin exists for build hosts that cannot reach
# github.com/*/archive/*.zip (some egress policies block it while allowing
# releases). It is recorded in the manifest as an incomplete payload so a build that
# used it cannot be mistaken for a shippable one.

set -euo pipefail

# Bumped whenever the payload's shape or contents change in a way an auditor
# reading an old BUNDLE_MANIFEST.json would need to know about.
GENERATOR_VERSION="1"

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
WITH_SD_CPP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT_DIR="${2:-}"; shift 2 ;;
        --work-dir) WORK_DIR="${2:-}"; KEEP_WORK=1; shift 2 ;;
        --skip-diffusers-pin) SKIP_DIFFUSERS=1; shift ;;
        --with-sd-cpp) WITH_SD_CPP=1; shift ;;
        -h|--help)
            sed -n '2,78p' "${BASH_SOURCE[0]}"
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
echo "==> 1/6 prebuilt archives (verified against the in-tree digest chain)"
PREBUILT_REPORT="$WORK_DIR/prebuilts.json"
fetch_args=(
    "$REPO_ROOT/scripts/fetch_macos_prebuilts.py"
    --runtime-dir "$STAGE"
    --report "$PREBUILT_REPORT"
    --work-dir "$WORK_DIR/downloads"
)
[ "$WITH_SD_CPP" = "1" ] && fetch_args+=(--with-sd-cpp)
"$PYTHON_BIN" "${fetch_args[@]}" || die "prebuilt fetch failed"

# The interpreter the bundle ships must be the one the locks were resolved for.
BUNDLED_PY="$STAGE/python/bin/python$PYTHON_MINOR"
require_file "$BUNDLED_PY"

# ───────────────────────────────────────────── 2. the Python distributions

echo
echo "==> 2/6 Python distributions (--require-hashes, into site-packages/)"

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
mapfile -t SDIST_ONLY < <(
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

# ───────────────────────────────── 3. relocatable console scripts

echo
echo "==> 3/6 rewriting console-script shebangs to be relocation-proof"
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

# ────────────────────────────────────────── 4. the OXC validator node_modules

echo
echo "==> 4/6 prefetching the OXC validator node_modules for darwin/arm64"
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

# ─────────────────────────────────────────────────────────────── 5. pruning

echo
echo "==> 5/6 pruning"
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

# ──────────────────────────────────────── 6. the manifest, and the contract

echo
echo "==> 6/6 BUNDLE_MANIFEST.json and the layout contract"
"$PYTHON_BIN" - \
    "$PINS_JSON" "$STAGE" "$PREBUILT_REPORT" "$PRUNE_REPORT" "$LOCK_DIR" \
    "$GENERATOR_VERSION" "$UV_ACTUAL" "$DIFFUSERS_URL" "$DIFFUSERS_SHA256" \
    "$DIFFUSERS_STATUS" "$REPO_ROOT" \
    <<'PY' || die "manifest generation failed"
"""Write runtime/BUNDLE_MANIFEST.json, then assert the layout contract holds.

The manifest exists so that a .dmg in somebody's Downloads folder can be audited
without rebuilding it: every component's version, source URL and sha256, the exact
locks that produced site-packages, what was pruned, and what the build could not do.
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
) = sys.argv[1:12]

pins = json.loads(pathlib.Path(pins_path).read_text(encoding = "utf-8"))
stage = pathlib.Path(stage_path).resolve()
prebuilts = json.loads(pathlib.Path(prebuilt_report).read_text(encoding = "utf-8"))
prune = json.loads(pathlib.Path(prune_report).read_text(encoding = "utf-8"))
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
    for name in ("python", "site-packages", "llama.cpp", "whisper.cpp", "node",
                 "oxc-node-modules", "sd.cpp")
    if (stage / name).exists()
}

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
    "distributions": distributions,
    "distribution_count": len(distributions),
    "pruned": prune,
    "sizes_bytes": sizes,
    "total_bytes": sum(sizes.values()),
    "layout": {
        "required_dirs": layout["required_dirs"],
        "required_files": layout["required_files"],
    },
    "incomplete": [] if diffusers_status == "installed" else [f"diffusers_pin: {diffusers_status}"],
}

(stage / "BUNDLE_MANIFEST.json").write_text(
    json.dumps(manifest, indent = 2, sort_keys = True) + "\n", encoding = "utf-8"
)

# The contract, asserted here rather than trusted. A payload that does not satisfy
# it must not reach a .dmg, because the Rust side resolves these exact paths.
missing = []
for name in layout["required_dirs"]:
    if not (stage / name).is_dir():
        missing.append(f"dir {name}")
for name in layout["required_files"]:
    if not (stage / name).exists():
        missing.append(f"file {name}")
for name in layout["required_executables"]:
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

print(f"manifest: {len(distributions)} distributions, {manifest['total_bytes'] / (1024 ** 3):.2f} GiB")
for name, value in sorted(sizes.items()):
    print(f"  {name:<18} {value / (1024 * 1024):9.1f} MiB")
PY

# ───────────────────────────────────────────────────────── swap into place

mkdir -p "$(dirname "$OUT_DIR")"
rm -rf "$OUT_DIR"
mv "$STAGE" "$OUT_DIR"

echo
echo "==> payload assembled at $OUT_DIR"
du -sh "$OUT_DIR" 2>/dev/null || true
