# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# In-tree source of truth for the Homebrew cask of the Unsloth Studio desktop app.
# Homebrew/homebrew-cask keeps its own copy at Casks/u/unsloth-studio.rb; this file is
# what that copy is mirrored FROM, so land changes here first and then port them.
# Strip this header block when mirroring -- homebrew-cask files carry no license header.
#
# Kept honest by tests/security/test_homebrew_cask_contract.py, which is part of the
# default pytest run (pyproject.toml testpaths = ["tests/security"]). It checks that
# the url template still reproduces the ASSET_VERSION transform in
# .github/workflows/release-desktop.yml, that `app`/`uninstall`/`zap`/`livecheck` still
# agree with studio/src-tauri/tauri.conf.json, and that every `zap trash:` path below
# is one that scripts/uninstall.sh also removes. Edit that script and this stanza
# together or the test fails.
#
# To refresh for a new release (tag vX.Y.Z-beta):
#   1. version "X.Y.Z-beta"                 # no leading v; the url adds it back
#   2. curl -fL https://github.com/unslothai/unsloth/releases/download/vX.Y.Z-beta/\
#        Unsloth-Desktop-$(printf %s X.Y.Z-beta | tr '.-' '__')-MacOS.dmg | shasum -a 256
#      and paste the digest into sha256
#   3. brew style ./unsloth-studio.rb && brew audit --cask --online ./unsloth-studio.rb
# See ./README.md for the full submission and validation runbook.

cask "unsloth-studio" do
  version "0.1.800-beta"
  sha256 "0cd2f2001b08df8bd4e47ea5784ccae9144ab80168f4964a2b89c9cd8e0b15ab"

  # release-desktop.yml derives the asset name from the tag with
  # ASSET_VERSION = re.sub(r'[^0-9A-Za-z]+', '_', app_version), so 0.1.800-beta
  # becomes 0_1_800_beta. Ruby's tr with a two-character from-set (the "-" is last,
  # so it is a literal and not a range) reproduces that for every version the
  # workflow accepts, because a SemVer tag's only non-alphanumerics are "." and "-".
  url "https://github.com/unslothai/unsloth/releases/download/v#{version}/Unsloth-Desktop-#{version.tr(".-", "__")}-MacOS.dmg",
      verified: "github.com/unslothai/unsloth/"
  name "Unsloth"
  desc "Local workbench for fine-tuning and running open-weight language models"
  homepage "https://unsloth.ai/"

  # Follows the updater manifest, not :github_latest. A unified v... release is
  # published before its desktop bundles finish uploading, so /releases/latest
  # transiently points at a release with no .dmg on it -- the very failure
  # .github/workflows/publish-desktop-updater.yml has a repair_pointer mode for.
  # latest.json's platform urls are pinned to the immutable release that built
  # them, so following the manifest never offers a version whose .dmg 404s.
  livecheck do
    url "https://github.com/unslothai/unsloth/releases/latest/download/latest.json"
    strategy :json do |json|
      json["version"]
    end
  end

  # tauri-plugin-updater checks latest.json and updates the app in place.
  auto_updates true
  # The published Mach-O is arm64 only; the Intel matrix leg in release-desktop.yml
  # is commented out. Info.plist still carries Tauri's default LSMinimumSystemVersion
  # of 10.13, but an arm64-only binary cannot run before Big Sur.
  depends_on arch: :arm64
  depends_on macos: ">= :big_sur"

  app "Unsloth.app"

  uninstall quit: "ai.unsloth.studio"

  # Kept in lockstep with scripts/uninstall.sh. Deliberately absent:
  # ~/.cache/huggingface (a multi-GB model cache shared with every other Hugging
  # Face tool, which uninstall.sh also preserves), and the artifacts that only the
  # curl | sh installer creates (~/Applications/Unsloth Studio.app, ~/.local/bin/unsloth,
  # ~/Desktop/Unsloth Studio) -- those belong to scripts/install.sh and are removed by
  # scripts/uninstall.sh, which is ownership-aware because both installs share the
  # ai.unsloth.studio bundle id.
  zap trash: [
    "~/.local/share/unsloth",
    "~/.unsloth/.cache",
    "~/.unsloth/.staging",
    "~/.unsloth/llama.cpp",
    "~/.unsloth/node",
    # uninstall.sh only removes this one when it carries Unsloth's
    # .unsloth-studio-owned marker, because the default path is also what a plain
    # `git clone` of leejet/stable-diffusion.cpp produces. zap is documented as the
    # aggressive teardown, so it takes the directory unconditionally.
    "~/.unsloth/stable-diffusion.cpp",
    "~/.unsloth/studio",
    "~/.unsloth/whisper.cpp",
    "~/Library/Application Support/ai.unsloth.studio",
    "~/Library/Caches/ai.unsloth.studio",
    "~/Library/Cookies/ai.unsloth.studio.binarycookies",
    "~/Library/HTTPStorages/ai.unsloth.studio",
    "~/Library/HTTPStorages/ai.unsloth.studio.binarycookies",
    "~/Library/Preferences/ai.unsloth.studio.plist",
    "~/Library/Saved Application State/ai.unsloth.studio.savedState",
    "~/Library/WebKit/ai.unsloth.studio",
  ]

  caveats <<~EOS
    The Hugging Face model cache at ~/.cache/huggingface is intentionally left in
    place, by both `brew uninstall --cask` and `brew uninstall --zap --cask`: it is
    shared with every other Hugging Face tool on this machine and is usually many
    gigabytes. Remove it yourself if you want the disk space back:
      rm -rf ~/.cache/huggingface/hub

    This cask only manages the packaged desktop app. If you also installed Unsloth
    with the shell installer (scripts/install.sh), remove that install with its own
    uninstaller, which knows about the CLI shim, the ~/Applications bundle and the
    Desktop shortcut:
      curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh
  EOS
end
