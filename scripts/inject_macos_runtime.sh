#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Put an assembled runtime payload inside a built Unsloth.app, and inside the .dmg
# tauri already produced from it.
#
# WHY THIS IS A SEPARATE STEP AND NOT A TAURI `resources` ENTRY
#
# Two reasons, both about not breaking things that already work.
#
# 1. Tauri's resource copier walks the source tree and copies file by file. The
#    payload is full of symlinks that matter -- python/bin/python3 ->
#    python3.13, and the whole libggml-*.0.dylib -> libggml-*.0.20.1.dylib
#    versioning chain in llama.cpp/build/bin. A copier that dereferences them
#    turns 27 MB of llama.cpp into three copies of every dylib, and one that drops
#    them leaves python/bin/python3 -- the exact path the app invokes -- missing.
#    `ditto` preserves symlinks, ownership and permissions, so it is used instead.
# 2. Declaring the payload in tauri.macos.conf.json would make every macOS build
#    require it, including release-desktop.yml, which is a separate stage and a
#    separate agent's file. This keeps the dev build's proving ground independent.
#
# WHY THE .DMG IS REBUILT BY CONVERSION AND NOT FROM SCRATCH
#
# tauri's dmg bundler produces a specific layout: the background image, the window
# size, the icon positions, the /Applications symlink, and the .DS_Store that
# encodes all of it. Recreating that with `hdiutil create -srcfolder` would lose it,
# and a dev build that produced a visibly different .dmg from the release would stop
# being evidence about the release. So the existing image is converted to a writable
# one, grown, mounted, written into, and converted back compressed -- the layout is
# carried through untouched and only files are added.
#
# The app inside carries no REAL signature when this finishes. A Developer ID
# signature must be applied after this step, never before, because injecting into a
# signed bundle invalidates it -- so an app that already carries one is refused.
#
# An ad-hoc seal is the exception, and is not optional: tauri applies one on Apple
# Silicon whether or not any Apple credential is present, because arm64 will not
# execute unsigned code. Injection invalidates it, so it is re-applied here and the
# bundle stays launchable. That is why the check below reads Signature= rather than
# just asking whether codesign succeeds.
#
# USAGE
#
#   bash scripts/inject_macos_runtime.sh \
#       --payload <runtime dir> --app <path/to/Unsloth.app> [--dmg <path.dmg>]
#
# --app is required; --dmg is optional and only meaningful on macOS.

set -euo pipefail

PAYLOAD=""
APP=""
DMG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --payload) PAYLOAD="${2:-}"; shift 2 ;;
        --app)     APP="${2:-}"; shift 2 ;;
        --dmg)     DMG="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }

[ -n "$PAYLOAD" ] || die "--payload <dir> is required"
[ -n "$APP" ] || die "--app <path/to/Unsloth.app> is required"
[ -d "$PAYLOAD" ] || die "payload directory does not exist: $PAYLOAD"
[ -d "$APP" ] || die "app bundle does not exist: $APP"
[ -f "$PAYLOAD/BUNDLE_MANIFEST.json" ] || die \
    "$PAYLOAD has no BUNDLE_MANIFEST.json, so it is not a payload this script will ship"

PAYLOAD="$(cd "$PAYLOAD" && pwd)"
APP="$(cd "$APP" && pwd)"

# `ditto` is macOS-only and is the only copier here that is trusted with the
# payload's symlinks and exec bits. `cp -R` on macOS would do, but ditto is the
# tool Apple documents for exactly this and it fails loudly.
copy_tree() {
    local source="$1" destination="$2"
    rm -rf "$destination"
    mkdir -p "$(dirname "$destination")"
    if command -v ditto >/dev/null 2>&1; then
        ditto "$source" "$destination"
    else
        # Non-macOS hosts: -a keeps symlinks and modes. Reached only when this
        # script is exercised off a Mac (the layout assertions below still run).
        cp -a "$source" "$destination"
    fi
}

RESOURCES="$APP/Contents/Resources"
[ -d "$RESOURCES" ] || die "$APP has no Contents/Resources; is this a macOS app bundle?"

# Refuse to invalidate a REAL signature rather than doing it quietly -- but an ad-hoc
# one is not a real signature and must not be treated as one.
#
# tauri ad-hoc signs the bundle on Apple Silicon whether or not any Apple credential
# is present, because arm64 refuses to execute unsigned code at all. So every build
# arrives here signed: the dev build with nothing but an ad-hoc seal, and the release
# build too, since its Apple credentials moved to a later signing step. Refusing on
# `codesign -dv` succeeding therefore refused everything, which is how it failed.
#
# An ad-hoc seal carries no identity and asserts nothing about origin; it exists so
# the code can run. Injecting invalidates it, so it is re-applied at the end of this
# script and the app stays launchable. A Developer ID signature is a different thing
# and is still refused: it must be applied after injection, never before.
ADHOC_SIGNED=0
if command -v codesign >/dev/null 2>&1; then
    if codesign -dv "$APP" >/dev/null 2>&1; then
        SIGNATURE="$(codesign -dv "$APP" 2>&1 | sed -n 's/^Signature=//p' | head -n 1)"
        if [ "$SIGNATURE" = "adhoc" ]; then
            ADHOC_SIGNED=1
            echo "    note: $APP is ad-hoc signed; re-sealing ad-hoc after injection"
        else
            die "$APP carries a real code signature (Signature=${SIGNATURE:-unknown}). \
Injecting the runtime would invalidate it; inject before signing."
        fi
    fi
fi

# Re-apply the ad-hoc seal so the bundle's resource manifest covers the payload and
# the app still launches on Apple Silicon. Not --deep: it does not sign Mach-Os under
# Contents/Resources anyway, and the payload's own wheels arrive ad-hoc signed from
# the build that produced them. A real release signature replaces this wholesale in
# the signing step that follows.
reseal_adhoc() {
    [ "$ADHOC_SIGNED" = "1" ] || return 0
    command -v codesign >/dev/null 2>&1 || return 0
    codesign --force --sign - "$1" >/dev/null 2>&1 \
        || die "could not re-apply the ad-hoc signature to $1"
}

echo "==> injecting the runtime payload into $APP"
copy_tree "$PAYLOAD" "$RESOURCES/runtime"

# The same contract the build script asserts, re-asserted here against the app
# bundle itself: this is the last point before the .dmg, and it is the layout the
# Rust side resolves.
for required in \
    runtime/BUNDLE_MANIFEST.json \
    runtime/python/bin/python3 \
    runtime/site-packages \
    runtime/llama.cpp/build/bin/llama-server \
    runtime/whisper.cpp/build/bin/whisper-server \
    runtime/stable-diffusion.cpp/build/bin/sd-cli \
    runtime/stable-diffusion.cpp/build/bin/sd-server \
    runtime/node/bin/node \
    runtime/oxc-node-modules
do
    [ -e "$RESOURCES/$required" ] || die "injection incomplete: $RESOURCES/$required is missing"
done
# Every binary the app executes, not just the interpreter: `ditto` and `cp -a` both
# preserve modes, so a missing exec bit here means the payload was assembled wrong,
# and finding out on a user's Mac costs a failed launch or a failed generation.
for executable in \
    runtime/python/bin/python3 \
    runtime/llama.cpp/build/bin/llama-server \
    runtime/whisper.cpp/build/bin/whisper-server \
    runtime/stable-diffusion.cpp/build/bin/sd-cli \
    runtime/stable-diffusion.cpp/build/bin/sd-server \
    runtime/node/bin/node
do
    [ -x "$RESOURCES/$executable" ] || die \
        "$executable is not executable inside the app bundle"
done

# And that they are Mach-O arm64. A binary for the wrong architecture copies, signs and
# ships perfectly and then cannot start, which is precisely the failure a cross-built
# payload risks. `file -L` so python3 (a symlink to python3.13) is judged by its target
# rather than reported as a link. Skipped where `file` is absent, which off a Mac it can
# be; the dev-build workflow's proof step runs this same assertion on macOS.
if command -v file >/dev/null 2>&1; then
    for executable in \
        runtime/python/bin/python3 \
        runtime/llama.cpp/build/bin/llama-server \
        runtime/whisper.cpp/build/bin/whisper-server \
        runtime/stable-diffusion.cpp/build/bin/sd-cli \
        runtime/stable-diffusion.cpp/build/bin/sd-server \
        runtime/node/bin/node
    do
        file -L "$RESOURCES/$executable" | grep -q 'arm64' || die \
            "$executable inside the app bundle is not an arm64 binary: \
$(file -L "$RESOURCES/$executable")"
    done
fi
reseal_adhoc "$APP"
echo "    ok: $(du -sh "$RESOURCES/runtime" 2>/dev/null | cut -f1) in Contents/Resources/runtime"

if [ -z "$DMG" ]; then
    exit 0
fi

[ -f "$DMG" ] || die "dmg does not exist: $DMG"
command -v hdiutil >/dev/null 2>&1 || die "hdiutil not found; --dmg only works on macOS"

echo "==> injecting the runtime payload into $DMG"
WORK="$(mktemp -d)"
MOUNT="$WORK/mnt"
mkdir -p "$MOUNT"
RW="$WORK/rw.dmg"

detach() { hdiutil detach "$MOUNT" -quiet >/dev/null 2>&1 || true; }
cleanup() { detach; rm -rf "$WORK"; }
trap cleanup EXIT

# UDRW: a read/write image the payload can be written into. The conversion carries
# the background, the window geometry and the /Applications symlink across.
hdiutil convert "$DMG" -format UDRW -o "$RW" -quiet

# Grow it to fit the payload plus slack. The tauri image is sized to the app alone,
# so writing ~2.5 GB into it without this fails with "No space left on device".
PAYLOAD_MB="$(du -sm "$PAYLOAD" | cut -f1)"
CURRENT_MB="$(du -sm "$RW" | cut -f1)"
TARGET_MB=$(( PAYLOAD_MB + CURRENT_MB + 512 ))
echo "    growing the image to ${TARGET_MB}m (payload ${PAYLOAD_MB}m + image ${CURRENT_MB}m + slack)"
hdiutil resize -size "${TARGET_MB}m" "$RW" >/dev/null

hdiutil attach "$RW" -nobrowse -mountpoint "$MOUNT" -quiet
MOUNTED_APP="$(find "$MOUNT" -maxdepth 1 -name '*.app' -print -quit)"
[ -n "$MOUNTED_APP" ] || die "no .app found inside $DMG"
copy_tree "$PAYLOAD" "$MOUNTED_APP/Contents/Resources/runtime"
[ -x "$MOUNTED_APP/Contents/Resources/runtime/python/bin/python3" ] || die \
    "the injected interpreter is not executable inside the mounted image"
reseal_adhoc "$MOUNTED_APP"
detach

# ULFO (lzfse) rather than UDZO (zlib): measurably smaller for a payload that is
# mostly already-compressed wheels and dylibs, and readable on every macOS the
# bundle targets.
OUT="$WORK/out.dmg"
hdiutil convert "$RW" -format ULFO -o "$OUT" -quiet
mv "$OUT" "$DMG"

echo "    ok: $(du -sh "$DMG" | cut -f1) $DMG"
