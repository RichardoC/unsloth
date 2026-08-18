<!--
SPDX-License-Identifier: AGPL-3.0-only
Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
-->

# Homebrew Cask submission runbook

`unsloth-studio.rb` in this directory is the in-tree source of truth for the Homebrew
cask of the Unsloth desktop app. Homebrew/homebrew-cask keeps its own copy at
`Casks/u/unsloth-studio.rb`; that copy is mirrored **from** this file, so land changes
here first and then port them. Strip the file's SPDX/maintenance header block when
mirroring — homebrew-cask files carry no license header.

The cask is kept honest by `tests/security/test_homebrew_cask_contract.py`, which runs in
the default pytest invocation (`pyproject.toml` sets `testpaths = ["tests/security"]`).

## Token: `unsloth-studio`

Not `unsloth-studio@beta`, and not `unsloth`.

* Homebrew's `@`-suffix convention marks an *alternate* channel that exists alongside a
  default cask (`firefox` / `firefox@nightly`). Unsloth Studio has exactly one desktop
  channel. Its versions carry a `-beta` suffix, but `.github/workflows/release-desktop.yml`
  (lines 312–314) deliberately publishes them as **non-prerelease** GitHub releases and
  promotes them to `/releases/latest` — the beta line *is* the shipping line. There is no
  channel setting anywhere in the codebase, so there is no second cask for an `@beta`
  variant to sit beside.
* `unsloth` is the name of the PyPI library and of the repository. Taking it for a macOS
  desktop cask would squat the more general name; `unsloth-studio` names the product this
  cask actually installs.

The app bundle itself is `Unsloth.app` with display name `Unsloth` (`productName` in
`studio/src-tauri/tauri.conf.json`), which is what the `name` and `app` stanzas say. The
token and the bundle name differing is normal and expected in homebrew-cask.

## Tag and asset naming contract

`.github/workflows/release-desktop.yml` is the only authority on asset names:

```python
# release-desktop.yml:129
asset_version = re.sub(r'[^0-9A-Za-z]+', '_', app_version).strip('_')
```

where `app_version = studio_version.removeprefix('v')`, and the macOS DMG is renamed to
`f'Unsloth-Desktop-{ASSET_VERSION}-MacOS.dmg'` (`release-desktop.yml:1576` and `:1584`;
the same prefix is asserted at `:316`, `:1747`, `:1806`). The tag itself must be SemVer
with a leading `v` (`release-desktop.yml:118–125`).

| Piece | Value for the current release |
| --- | --- |
| Git tag | `v0.1.800-beta` |
| `ASSET_VERSION` | `0_1_800_beta` |
| DMG asset | `Unsloth-Desktop-0_1_800_beta-MacOS.dmg` |
| DMG volume name | `Unsloth_0.1.800-beta_aarch64` |
| Bundle | `Unsloth.app`, executable `Contents/MacOS/unsloth-studio` |
| Bundle id | `ai.unsloth.studio` |

So the general shape is `v0.1.NN-beta` → `Unsloth-Desktop-0_1_NN_beta-MacOS.dmg`.

The cask reproduces that transform in Ruby with
`version.tr(".-", "__")`. The `-` is the **last** character of the from-set, so Ruby reads
it as a literal rather than a range, and the call maps `.` → `_` and `-` → `_`. A SemVer
tag accepted by the workflow contains no other non-alphanumeric characters (`+build`
metadata is rejected by the workflow's regex), so `tr` and the Python `re.sub` agree on
every version that can ship. The only theoretical divergence is a run of consecutive
separators (`re.sub` collapses `--` to one `_`, `tr` produces two) — a version string like
`0.1.800-beta--x` is not something this project publishes, and the contract test pins the
realistic shapes.

## Why `livecheck` follows `latest.json` and not `:github_latest`

A unified `v...` release is created before its desktop bundles finish uploading, so
`https://github.com/unslothai/unsloth/releases/latest` transiently resolves to a release
that has no `.dmg` attached. That is exactly the failure
`.github/workflows/publish-desktop-updater.yml` carries a `repair_pointer` mode for, and
`tests/security/test_desktop_updater_pointer.py` documents.

`:github_latest` would read that pointer directly and offer a version whose asset URL
404s. The Tauri updater manifest `latest.json` instead has per-platform URLs pinned to the
immutable release that actually built those bundles, so its `version` field is only ever a
version whose DMG exists. The cask therefore livechecks the manifest:

```ruby
livecheck do
  url "https://github.com/unslothai/unsloth/releases/latest/download/latest.json"
  strategy :json do |json|
    json["version"]
  end
end
```

The URL is the same single endpoint configured at
`studio/src-tauri/tauri.conf.json` → `plugins.updater.endpoints`, and the contract test
asserts the two stay equal.

## Platform support: arm64 only

The published Mach-O is arm64-only — the Intel matrix leg is commented out at
`release-desktop.yml:506–508`, so no `x86_64` or universal DMG is produced. Hence
`depends_on arch: :arm64`.

`Info.plist` still carries Tauri's default `LSMinimumSystemVersion` of `10.13`, which is
not the real floor: an arm64-only binary cannot run on anything before Big Sur. The cask
therefore declares `depends_on macos: :big_sur` rather than trusting the plist. The bare
symbol is Homebrew's minimum-version form, so that reads "Big Sur or newer"; the
equivalent `">= :big_sur"` is rejected by `brew style`'s `Homebrew/OSDependsOn` cop, and
only the `"== :big_sur"` string form would pin one exact release.

If an Intel or universal build is ever enabled, drop `depends_on arch: :arm64` and split
the `sha256`/`url` per arch with an `on_arm` / `on_intel` block (or a `sha256` +
`arch` stanza pair), and relax the macOS floor only if the build actually supports older
systems.

## Refreshing `version` and `sha256`

For a new release tagged `vX.Y.Z-beta`:

```sh
VERSION=0.1.801-beta
ASSET_VERSION=$(printf '%s' "$VERSION" | tr '.-' '__')
curl -fL "https://github.com/unslothai/unsloth/releases/download/v${VERSION}/Unsloth-Desktop-${ASSET_VERSION}-MacOS.dmg" \
  | shasum -a 256
```

Paste the digest into `sha256` and the version into `version`. Nothing else in the cask
changes — the `url` interpolates `version`.

Current values, verified against the live release:

* `version "0.1.800-beta"`
* `sha256 "0cd2f2001b08df8bd4e47ea5784ccae9144ab80168f4964a2b89c9cd8e0b15ab"` (45041447 bytes)

## `zap` stays in lockstep with `scripts/uninstall.sh`

Every path in `zap trash:` is a path `scripts/uninstall.sh` also removes (the shell script
writes them with `$HOME` and a `$_bid` bundle-id variable; the contract test normalizes
before comparing). Add a path to one and you must add it to the other.

Two categories are deliberately **excluded**:

* `~/.cache/huggingface` — the shared Hugging Face model cache, routinely tens of
  gigabytes and used by every other Hugging Face tool on the machine.
  `scripts/uninstall.sh` preserves it on purpose and says so (lines 33–34 and 888–889), so
  `zap` does too. The cask's `caveats` tells the user where it is and how to delete it.
* `~/Applications/Unsloth Studio.app`, `~/.local/bin/unsloth`, `~/Desktop/Unsloth Studio`
  — artifacts of the `curl | sh` installer (`scripts/install.sh`) only. The packaged app
  and the shell install share the `ai.unsloth.studio` bundle id, which is why
  `scripts/uninstall.sh` is ownership-aware via its `_bundle_id_owner` helper. A cask must
  not delete another installation method's files, so `scripts/uninstall.sh` remains the
  way to remove a CLI install.

`~/.unsloth/stable-diffusion.cpp` is in the zap list even though `scripts/uninstall.sh`
only removes it when it carries Unsloth's `.unsloth-studio-owned` marker (the default path
is also what a plain `git clone` of `leejet/stable-diffusion.cpp` produces). `zap` is
documented as the aggressive, opt-in teardown, so it takes the directory unconditionally.

## Validation in CI

`.github/workflows/homebrew-cask-ci.yml` runs all of the below on an Apple Silicon
`macos-15` runner whenever this directory, `tauri.conf.json`, `scripts/uninstall.sh` or the
contract test changes. It stages the cask into a throwaway tap (so `brew audit --new` and
`brew livecheck` resolve it by token, the way homebrew-cask will), then styles, audits,
installs the published DMG for real, verifies the installed bundle, and zaps it.

That run is the evidence a cask PR needs. From the first green run on
`v0.1.800-beta`:

```
brew style                 no offenses
brew audit --strict --online  passed
brew audit --new --online     passed
codesign --verify --deep --strict  valid
xcrun stapler validate     The validate action worked!
spctl --assess             /Applications/Unsloth.app: accepted
                           source=Notarized Developer ID
brew livecheck             unsloth-studio: 0.1.800-beta ==> 0.1.800-beta
brew uninstall --zap       trashed all 16 paths; ~/.cache/huggingface preserved
```

The install step also proves the `url` the cask builds from `version` resolves and matches
the pinned `sha256` — `brew audit --online` fetches it, and `brew install` would reject a
mismatch.

## Local validation

The same checks, on macOS with Homebrew installed, from this directory:

```sh
brew style ./unsloth-studio.rb
brew audit --cask --new --online ./unsloth-studio.rb
brew livecheck --cask ./unsloth-studio.rb
HOMEBREW_NO_INSTALL_FROM_API=1 brew install --cask ./unsloth-studio.rb
brew uninstall --cask unsloth-studio
```

`brew audit --new` is the gate homebrew-cask CI applies to a brand-new cask; `--online`
adds the URL/checksum fetch. `HOMEBREW_NO_INSTALL_FROM_API=1` is required so `brew
install` uses this local file instead of the JSON API index.

The repo-side check is pure Python and needs no Ruby or network:

```sh
python3 -m pytest tests/security/test_homebrew_cask_contract.py -v
```

## Out of scope / cannot be done from this repo

* **The homebrew-cask pull request itself.** It has to be opened against
  `Homebrew/homebrew-cask` by a person with a GitHub account; nothing here can do that.
  The `brew style` / `brew audit --cask --new --online` evidence homebrew-cask asks for is
  no longer the blocker it was — `homebrew-cask-ci.yml` produces it on every change, so
  link that run in the PR.
* **Notability.** homebrew-cask judges a new cask on the notability of the upstream
  project (stars, forks, activity on `unslothai/unsloth`), not on anything in this
  directory. Nothing here can change that verdict.
* **The `Casks/u/unsloth-studio.rb` path, tap layout, and `brew bump-cask-pr` automation**
  all live in homebrew-cask. This directory intentionally holds a single template file
  plus this runbook — no `Casks/` tree, because a tap in this repo would be a second,
  competing distribution channel to keep in sync.
* **Code signing / notarization credentials.** Already handled in
  `release-desktop.yml` (sign → notarize → staple); the cask relies on that and does not
  set `quarantine` overrides.

## Mapping to issue unslothai/unsloth#5156

The issue predates the current release pipeline; most of it is already satisfied.

| Issue item | Status | Detail |
| --- | --- | --- |
| 1. Publish real, downloadable macOS desktop assets | **Already resolved before this change** | `release-desktop.yml` builds, signs, notarizes and staples the arm64 DMG and uploads it. Verified live: `https://github.com/unslothai/unsloth/releases/download/v0.1.800-beta/Unsloth-Desktop-0_1_800_beta-MacOS.dmg` returns HTTP 200. |
| 2. Use stable, predictable tag and asset names | **Satisfied, with different names than the issue proposed** | The shipped scheme is `v0.1.NN-beta` → `Unsloth-Desktop-0_1_NN_beta-MacOS.dmg`, derived deterministically from the tag by `ASSET_VERSION` (`release-desktop.yml:129`). The cask reproduces the transform, so it is stable and predictable — see the naming section above. |
| 3. Point the auto-updater at the official repo | **Already resolved before this change** | `studio/src-tauri/tauri.conf.json` has the single endpoint `https://github.com/unslothai/unsloth/releases/latest/download/latest.json`. The `danielhanchen/unsloth-staging-2` endpoint the issue complains about is gone, and `tests/studio/test_tauri_branding_contract.py` pins the official one. |
| 4. Provide a cask draft ready for submission | **Done by this change** | `unsloth-studio.rb` in this directory, plus the contract test and this runbook. |
| 5. Document the submission process | **Done by this change** | This file. |
| Issue's literal `desktop-v2026.4.7` tags | **Superseded** | Desktop releases share the repo's unified SemVer `v...` tags. `tests/security/test_desktop_release_resolver.py:63` asserts `desktop-v` is *not* used, so implementing the issue literally would break CI. |
| Issue's literal `Unsloth-Studio-Desktop-aarch64.dmg` asset | **Superseded** | The asset name is version-stamped (the cask needs that, so a new version is a new URL and a new checksum) and the string "Unsloth Studio" is forbidden in display sources by `tests/studio/test_tauri_branding_contract.py:195–216`. |
| Issue's `desktop-latest` moving tag | **Superseded** | Discovery goes through `latest.json` on `/releases/latest`, maintained by `release-desktop.yml` and repairable by `publish-desktop-updater.yml`. See the livecheck section. |
| Issue's concern that the app version is CalVer (`2026.4.7`) | **Not applicable** | The CalVer `2026.4.8` in `studio/src-tauri/Cargo.toml:9` is a documented placeholder that never ships; CI rewrites it to the SemVer release version before every release build. No CalVer string reaches a released artifact, so it cannot confuse a cask. |
| Channel and token strategy (`unsloth-studio` vs `unsloth-studio@beta`) | **Decided: `unsloth-studio`** | See the token section above. There is one desktop channel; its versions carry `-beta`, but they ship as non-prerelease releases promoted to `/releases/latest`. |
| 6. Confirm the final `.app` bundle name on disk | **Confirmed empirically** | `Unsloth.app`, executable `Contents/MacOS/unsloth-studio`, `CFBundleIdentifier` `ai.unsloth.studio`, `CFBundleShortVersionString` `0.1.800-beta` — read out of the published `v0.1.800-beta` DMG, not inferred. The bundled Mach-O is arm64-only and carries a `_CodeSignature`. The cask's `app` stanza is pinned to `productName` in `tauri.conf.json` by the contract test. |
| 7. Confirm the real macOS cleanup paths for `zap` | **Done by this change** | The 16 paths in the `zap trash:` array, each verified to be a path `scripts/uninstall.sh` also removes, and held that way by the contract test. `~/.cache/huggingface` is deliberately excluded — see the lockstep section. |
| 8. Confirm whether the desktop app is the primary macOS install path | **Documented; remains an upstream product decision** | `README.md` presents the desktop app first and now offers Homebrew as a macOS route, while keeping the install script for Intel Macs and CLI users. Nothing in this directory needs that question settled: the cask is valid either way. |
