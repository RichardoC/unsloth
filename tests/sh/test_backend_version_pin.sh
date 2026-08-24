#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests for the exact-backend-version pin in install.sh.
#
# install.sh locates setup.sh and every requirements/constraints file inside the
# INSTALLED unsloth wheel, so the wheel a run resolves decides the entire pin set.
# A desktop release therefore has to install one exact wheel or the same .dmg
# produces a different Python stack every month. UNSLOTH_BACKEND_VERSION (or
# --backend-version) turns every backend spec into ==; unset, nothing may change,
# because `curl | sh` CLI users and CI still want the floors.
#
# The specs are not asserted by grepping the source: each case runs install.sh's own
# flag-parsing and pin blocks and then replays a real call site through a
# run_install_cmd_retry stub, so the argv asserted here is the argv uv would see.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_RS="$SCRIPT_DIR/../../studio/src-tauri/src/install.rs"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

assert_true() {
    _label="$1"; _ok="$2"
    if [ "$_ok" = "0" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label"; FAIL=$((FAIL + 1))
    fi
}

assert_eq() {
    _label="$1"; _got="$2"; _want="$3"
    if [ "$_got" = "$_want" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label"
        echo "        want: $_want"
        echo "        got:  $_got"
        FAIL=$((FAIL + 1))
    fi
}

has() { if grep -q "$@"; then return 0; else return 1; fi; }

echo "=== test_backend_version_pin ==="

# ── Harness ──
# The three blocks that decide the specs, lifted verbatim out of install.sh: flag
# parsing, the --backend-version argument check, and the pin/spec derivation.
PARSE_BLOCK="$_TMP_ROOT/parse.sh"
sed -n '/^# ── Parse flags ──$/,/^done$/p' "$INSTALL_SH" > "$PARSE_BLOCK"
if [ -s "$PARSE_BLOCK" ]; then _rc=0; else _rc=1; fi
assert_true "flag-parsing block extracted" "$_rc"

ARGCHECK_BLOCK="$_TMP_ROOT/argcheck.sh"
sed -n '/^if \[ "\$_next_is_backend_version" = true \]; then$/,/^fi$/p' "$INSTALL_SH" > "$ARGCHECK_BLOCK"
if [ -s "$ARGCHECK_BLOCK" ]; then _rc=0; else _rc=1; fi
assert_true "--backend-version argument check extracted" "$_rc"

PIN_BLOCK="$_TMP_ROOT/pin.sh"
sed -n '/^# ── Backend version pin ──$/,/^# ── Tauri structured output ──$/p' "$INSTALL_SH" \
    | sed '$d' | sed '$d' > "$PIN_BLOCK"
if has '^_UNSLOTH_SPEC=' "$PIN_BLOCK"; then _rc=0; else _rc=1; fi
assert_true "pin/spec derivation block extracted" "$_rc"

# render <version|-> <sed-range> [args...] : run the real blocks, then the real
# call-site lines, with run_install_cmd_retry stubbed to print the argv it was handed.
# The version goes through `env` rather than an assignment prefix, because an
# assignment prefixed onto a shell FUNCTION leaks into the rest of the script.
render() {
    _ver="$1"; _range="$2"; shift 2
    _script="$_TMP_ROOT/render.sh"
    {
        echo 'set -e'
        cat "$PARSE_BLOCK"
        cat "$ARGCHECK_BLOCK"
        # Stub the reporting helpers and the venv/overrides state the call sites read.
        echo 'substep() { :; }'
        echo 'run_install_cmd_retry() { shift; printf "%s\n" "$*"; }'
        echo 'run_install_cmd() { shift; printf "%s\n" "$*"; }'
        echo '_VENV_PY=/venv/bin/python'
        echo '_UNSLOTH_TORCH_OVERRIDES=""'
        cat "$PIN_BLOCK"
        sed -n "$_range" "$INSTALL_SH"
    } > "$_script"
    _rc=0
    if [ "$_ver" = "-" ]; then
        env -u UNSLOTH_BACKEND_VERSION sh "$_script" "$@" || _rc=$?
    else
        env UNSLOTH_BACKEND_VERSION="$_ver" sh "$_script" "$@" || _rc=$?
    fi
    return $_rc
}

# The call sites, by their run_install_cmd_retry label.
R_BOOTSTRAP='/"prepare Apple Silicon dependencies"/,/_UNSLOTH_SPEC"$/p'
R_DESKTOP='/"install unsloth" uv pip install/,/_UNSLOTH_SPEC"$/p'
R_AUTO='/"install unsloth (auto torch backend)" uv pip install --python "\$_VENV_PY" --torch-backend=auto/p'
R_NOTORCH='/"install unsloth (no-torch)"/,/_UNSLOTH_ZOO_SPEC"$/p'
R_MIGRATED='/"install unsloth (migrated)"/,/_UNSLOTH_ZOO_SPEC"$/p'
R_MIGRATED_NT='/"install unsloth (migrated no-torch)"/,/_UNSLOTH_ZOO_SPEC"$/p'
R_LOCAL='/"install unsloth (local)"/,/unsloth-zoo>=2026.8.12"$/p'

# ── 1. Unset: every spec is byte-for-byte what it is today ──
# Hardcoded, not diffed against git HEAD: once this lands HEAD is the new file and a
# self-comparison would assert nothing. These strings are the CLI-user contract.
_got=$(render - "$R_BOOTSTRAP")
assert_eq "unpinned: Apple Silicon bootstrap unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --no-deps --upgrade-package unsloth -- unsloth'

_got=$(render - "$R_DESKTOP")
assert_eq "unpinned: default desktop path unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --upgrade-package unsloth -- unsloth'

_got=$(render - "$R_AUTO")
assert_eq "unpinned: auto-torch-backend fallback unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --torch-backend=auto -- unsloth'

_got=$(render - "$R_NOTORCH")
assert_eq "unpinned: fresh no-torch unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --no-deps --upgrade-package unsloth --upgrade-package unsloth-zoo unsloth>=2026.8.18 unsloth-zoo>=2026.8.12'

_got=$(render - "$R_MIGRATED")
assert_eq "unpinned: migrated env unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --reinstall-package unsloth --reinstall-package unsloth-zoo unsloth>=2026.8.18 unsloth-zoo>=2026.8.12'

_got=$(render - "$R_MIGRATED_NT")
assert_eq "unpinned: migrated no-torch unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo unsloth>=2026.8.18 unsloth-zoo>=2026.8.12'

_got=$(render - "$R_LOCAL")
assert_eq "unpinned: --local unchanged" "$_got" \
    'uv pip install --python /venv/bin/python --upgrade-package unsloth unsloth>=2026.8.18 unsloth-zoo>=2026.8.12'

# --package still reaches the sites that honor it.
_got=$(render - "$R_DESKTOP" --package unsloth-nightly)
assert_eq "unpinned: --package still reaches the desktop path" "$_got" \
    'uv pip install --python /venv/bin/python --upgrade-package unsloth -- unsloth-nightly'

# ── 2. Pinned: the default desktop path installs an exact version ──
# Both input shapes, because the desktop exports the env var while a developer
# reproducing a release passes the flag.
for _mode in env flag; do
    if [ "$_mode" = env ]; then
        _got=$(render 2026.8.18 "$R_DESKTOP")
    else
        _got=$(render - "$R_DESKTOP" --backend-version 2026.8.18)
    fi
    assert_eq "pinned ($_mode): default desktop path installs unsloth==2026.8.18" "$_got" \
        'uv pip install --python /venv/bin/python -- unsloth==2026.8.18'
    # The point of the whole change: no >= floor for unsloth survives on that path.
    if printf '%s' "$_got" | grep -q 'unsloth>='; then _rc=1; else _rc=0; fi
    assert_true "pinned ($_mode): no unsloth >= floor on the desktop path" "$_rc"
    # --upgrade-package is dropped, not left contradicting the ==pin.
    if printf '%s' "$_got" | grep -q -- '--upgrade-package'; then _rc=1; else _rc=0; fi
    assert_true "pinned ($_mode): --upgrade-package dropped alongside the ==pin" "$_rc"
done

# An explicit flag beats an inherited env var, like every other option here.
_got=$(render 2026.8.4 "$R_DESKTOP" --backend-version 2026.8.18)
assert_eq "pinned: --backend-version overrides UNSLOTH_BACKEND_VERSION" "$_got" \
    'uv pip install --python /venv/bin/python -- unsloth==2026.8.18'

# ── 2b. Pinned: the other non-local sites ──
_got=$(render 2026.8.18 "$R_BOOTSTRAP")
assert_eq "pinned: Apple Silicon bootstrap pins the same wheel" "$_got" \
    'uv pip install --python /venv/bin/python --no-deps -- unsloth==2026.8.18'

_got=$(render 2026.8.18 "$R_AUTO")
assert_eq "pinned: auto-torch-backend fallback pins" "$_got" \
    'uv pip install --python /venv/bin/python --torch-backend=auto -- unsloth==2026.8.18'

# unsloth-zoo keeps its floor: there is no stamped zoo version to pin to, and the
# exact unsloth constrains it through its own metadata. Guards against a future edit
# inventing a zoo pin here.
_got=$(render 2026.8.18 "$R_NOTORCH")
assert_eq "pinned: fresh no-torch pins unsloth, keeps the zoo floor" "$_got" \
    'uv pip install --python /venv/bin/python --no-deps --upgrade-package unsloth-zoo unsloth==2026.8.18 unsloth-zoo>=2026.8.12'

_got=$(render 2026.8.18 "$R_MIGRATED")
assert_eq "pinned: migrated env pins and keeps --reinstall-package" "$_got" \
    'uv pip install --python /venv/bin/python --reinstall-package unsloth --reinstall-package unsloth-zoo unsloth==2026.8.18 unsloth-zoo>=2026.8.12'

_got=$(render 2026.8.18 "$R_MIGRATED_NT")
assert_eq "pinned: migrated no-torch pins and keeps --reinstall-package" "$_got" \
    'uv pip install --python /venv/bin/python --no-deps --reinstall-package unsloth --reinstall-package unsloth-zoo unsloth==2026.8.18 unsloth-zoo>=2026.8.12'

# ── 2c. --local ignores the pin: the editable overlay is the developer's version ──
_got=$(render 2026.8.18 "$R_LOCAL")
assert_eq "pinned: --local branch still resolves from the floors" "$_got" \
    'uv pip install --python /venv/bin/python --upgrade-package unsloth unsloth>=2026.8.18 unsloth-zoo>=2026.8.12'

# ── 3. Malformed versions are refused, loudly, before anything is installed.
# The value is concatenated into a uv argv, so this doubles as the injection gate. ──
for _bad in \
    "latest" \
    "v2026.8.18" \
    ">=2026.8.18" \
    "2026.8.18 --index-url http://evil.invalid" \
    "2026.8.18;rm -rf /" \
    '2026.8.18$(id)' \
    "2026.8.18 unsloth-zoo" \
    "2026.8.18-" \
    "2026.8.18.beta" \
    "2026.8.18RC1" \
    ".8.18" \
    "-2026.8.18"
do
    _rc=0
    _out=$(render "$_bad" "$R_DESKTOP" 2>&1) || _rc=$?
    if [ "$_rc" = 0 ]; then
        echo "  FAIL: malformed version accepted: [$_bad] -> $_out"; FAIL=$((FAIL + 1))
    else
        case "$_out" in
            *"PEP 440 release version"*)
                echo "  PASS: malformed version refused with a useful message: [$_bad]"
                PASS=$((PASS + 1)) ;;
            *)
                echo "  FAIL: malformed version refused without saying why: [$_bad] -> $_out"
                FAIL=$((FAIL + 1)) ;;
        esac
    fi
done

# A newline-smuggled second requirement must not sneak past the anchored regex.
_multiline=$(printf '2026.8.18\nunsloth-zoo')
_rc=0
_out=$(render "$_multiline" "$R_DESKTOP" 2>&1) || _rc=$?
if [ "$_rc" = 0 ]; then
    echo "  FAIL: multi-line version accepted -> $_out"; FAIL=$((FAIL + 1))
else
    echo "  PASS: multi-line version refused"; PASS=$((PASS + 1))
fi

# ── 3b. Well-formed pre/post/dev releases are accepted: the release workflow can
# publish one, and rejecting it would strand that build on a floor. ──
for _good in 2026.8.18 2026.8.18rc1 2026.8.18a1 2026.8.18b2 2026.8.18.post1 2026.8.18.dev0 1.2; do
    _got=$(render "$_good" "$R_DESKTOP" 2>&1)
    assert_eq "accepted: $_good" "$_got" \
        "uv pip install --python /venv/bin/python -- unsloth==$_good"
done

# ── 4. --backend-version with no argument is an error, not a silent no-op ──
_rc=0
_out=$(render - "$R_DESKTOP" --backend-version 2>&1) || _rc=$?
if [ "$_rc" = 0 ]; then
    echo "  FAIL: --backend-version with no argument was accepted"; FAIL=$((FAIL + 1))
else
    case "$_out" in
        *"--backend-version requires a version argument"*)
            echo "  PASS: --backend-version with no argument is refused"; PASS=$((PASS + 1)) ;;
        *)
            echo "  FAIL: --backend-version with no argument gave: $_out"; FAIL=$((FAIL + 1)) ;;
    esac
fi

# ── 5. PACKAGE_NAME must stay a bare name: it is a `case` match, a --package re-run
# hint and a `uv pip show` argument, all of which a specifier would break. ──
if has '^PACKAGE_NAME="unsloth"$' "$INSTALL_SH"; then _rc=0; else _rc=1; fi
assert_true "PACKAGE_NAME is still a bare package name" "$_rc"
if has -E '^PACKAGE_NAME=.*(==|>=)' "$INSTALL_SH"; then _rc=1; else _rc=0; fi
assert_true "no version specifier was folded into PACKAGE_NAME" "$_rc"

# ── 5b. A pinned install is a materially different install, so it says so once,
# where tauri.log will pick it up. ──
if has 'step "backend" "pinned to \$_UNSLOTH_SPEC"' "$INSTALL_SH"; then _rc=0; else _rc=1; fi
assert_true "the pinned version is reported once" "$_rc"
_reports=$(grep -c 'pinned to \$_UNSLOTH_SPEC' "$INSTALL_SH" || true)
assert_eq "the pin is reported exactly once" "$_reports" "1"

# ── 6. The WSL reroute re-runs a freshly fetched install.sh, so the pin has to
# cross with it or the rerouted install silently floats. ──
if has 'export UNSLOTH_BACKEND_VERSION=\$(_rr_q "\$_BACKEND_VERSION")' "$INSTALL_SH"; then
    _rc=0
else
    _rc=1
fi
assert_true "WSL reroute forwards the backend pin" "$_rc"

# ── 7. The desktop app passes the stamp, and only when the build was stamped ──
if [ ! -f "$INSTALL_RS" ]; then
    echo "  FAIL: install.rs not found"; FAIL=$((FAIL + 1))
else
    if has 'option_env!("UNSLOTH_DESKTOP_BACKEND_VERSION")' "$INSTALL_RS"; then _rc=0; else _rc=1; fi
    assert_true "install.rs reads the stamp with option_env!" "$_rc"
    if has 'cmd.env("UNSLOTH_BACKEND_VERSION", backend_version)' "$INSTALL_RS"; then _rc=0; else _rc=1; fi
    assert_true "install.rs exports UNSLOTH_BACKEND_VERSION to the installer" "$_rc"
    # expected_backend_version() floors an unstamped build at MIN_DESKTOP_BACKEND_VERSION.
    # Right for staleness checks, wrong here: it would pin dev builds to a stale version.
    # Comment lines are stripped first -- explaining the floor is fine, calling it is not.
    if grep -vE '^[[:space:]]*//' "$INSTALL_RS" \
        | grep -qE 'expected_backend_version|unwrap_or\(MIN_DESKTOP_BACKEND_VERSION\)'; then
        _rc=1
    else
        _rc=0
    fi
    assert_true "install.rs does not pin through the MIN_DESKTOP_BACKEND_VERSION floor" "$_rc"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
