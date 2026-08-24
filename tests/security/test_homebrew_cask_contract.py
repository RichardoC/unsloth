# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The Homebrew cask draft must keep telling the truth about what we ship.

`studio/packaging/homebrew/unsloth-studio.rb` is mirrored into Homebrew/homebrew-cask,
where nothing in this repo can fix it. So the facts it hardcodes are pinned here instead:
the download url must reproduce the ASSET_VERSION transform in release-desktop.yml, the
app/bundle-id/updater-endpoint stanzas must agree with tauri.conf.json, and every path
`zap` deletes must be a path scripts/uninstall.sh also deletes.

Pure text and JSON parsing: no network, and no Ruby interpreter required.
"""

import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CASK = REPO / "studio/packaging/homebrew/unsloth-studio.rb"
TAURI_CONFIG = REPO / "studio/src-tauri/tauri.conf.json"
RELEASE_WORKFLOW = REPO / ".github/workflows/release-desktop.yml"
UNINSTALLER = REPO / "scripts/uninstall.sh"

# Shapes the release workflow accepts, spanning the ones that have actually shipped.
SAMPLE_VERSIONS = ("0.1.800-beta", "0.1.52-beta", "0.1.528-beta", "1.0.0")


def read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def tauri_config() -> dict:
    return json.loads(read(TAURI_CONFIG))


def cask_source() -> str:
    assert CASK.is_file(), f"missing cask draft: {CASK.relative_to(REPO)}"
    return read(CASK)


def strip_ruby_comments(source: str) -> str:
    """Drop whole-line `#` comments, which is the only comment style the cask uses."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def stanza(name: str) -> str:
    """The single double-quoted argument of a one-line stanza, e.g. version "1.2.3"."""
    body = strip_ruby_comments(cask_source())
    found = re.findall(rf'^\s*{re.escape(name)}\s+"([^"]*)"\s*,?\s*$', body, re.MULTILINE)
    assert len(found) == 1, (
        f'expected exactly one `{name} "..."` stanza in {CASK.name}, found {found}'
    )
    return found[0]


def cask_url_template() -> str:
    body = strip_ruby_comments(cask_source())
    found = re.findall(r'^\s*url\s+"(.+?)",\s*$', body, re.MULTILINE)
    assert len(found) == 1, (
        f"expected exactly one `url \"...\",` stanza in {CASK.name}, found {found}"
    )
    return found[0]


def cask_zap_paths() -> list[str]:
    body = strip_ruby_comments(cask_source())
    match = re.search(r"^\s*zap\s+trash:\s*\[(.*?)^\s*\]", body, re.MULTILINE | re.DOTALL)
    assert match, f"{CASK.name} must carry a multi-line `zap trash: [...]` stanza"
    paths = re.findall(r'"([^"]+)"', match.group(1))
    assert paths, f"the `zap trash:` array in {CASK.name} is empty"
    return paths


def workflow_asset_version_transform():
    """The workflow's own ASSET_VERSION rule, parsed out rather than duplicated."""
    workflow = read(RELEASE_WORKFLOW)
    match = re.search(
        r"asset_version = re\.sub\(\s*r'([^']+)'\s*,\s*'([^']*)'\s*,\s*app_version\s*\)"
        r"(?P<strip>\.strip\('_'\))?",
        workflow,
    )
    assert match, (
        "could not find `asset_version = re.sub(...)` in "
        f"{RELEASE_WORKFLOW.relative_to(REPO)}; the asset naming rule moved, so the cask's "
        "url template needs re-deriving"
    )
    pattern, replacement, strip = match.group(1), match.group(2), match.group("strip")

    def transform(version: str) -> str:
        result = re.sub(pattern, replacement, version)
        return result.strip("_") if strip else result

    return transform


def workflow_semver_pattern() -> str:
    """The `semver_tag` regex from the workflow, reassembled from its adjacent literals."""
    workflow = read(RELEASE_WORKFLOW)
    match = re.search(r"semver_tag = re\.compile\(\s*((?:r'[^']*'\s*)+)\)", workflow)
    assert match, (
        f"could not find `semver_tag = re.compile(...)` in {RELEASE_WORKFLOW.relative_to(REPO)}"
    )
    return "".join(re.findall(r"r'([^']*)'", match.group(1)))


def expand_tr_set(spec: str) -> list[str]:
    """Expand a Ruby String#tr set: `a-c` is a range, a leading or trailing `-` is literal."""
    characters: list[str] = []
    index = 0
    while index < len(spec):
        if spec[index] == "\\" and index + 1 < len(spec):
            characters.append(spec[index + 1])
            index += 2
            continue
        if (
            spec[index] == "-"
            and characters
            and index + 1 < len(spec)
            and spec[index + 1] != "-"
        ):
            start, stop = characters[-1], spec[index + 1]
            characters.extend(chr(code) for code in range(ord(start) + 1, ord(stop) + 1))
            index += 2
            continue
        characters.append(spec[index])
        index += 1
    return characters


def ruby_tr(value: str, from_spec: str, to_spec: str) -> str:
    """Ruby's String#tr for the non-negated case: `to` is padded with its last character."""
    assert not from_spec.startswith("^"), "negated tr sets are not modelled here"
    source = expand_tr_set(from_spec)
    target = expand_tr_set(to_spec)
    assert target, "an empty tr replacement set deletes characters; the cask must not do that"
    mapping = {
        character: target[min(position, len(target) - 1)]
        for position, character in enumerate(source)
    }
    return "".join(mapping.get(character, character) for character in value)


def cask_tr_arguments() -> tuple[str, str]:
    match = re.search(r'version\.tr\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)', cask_url_template())
    assert match, (
        f"the `url` in {CASK.name} must derive the asset name with version.tr(\"...\", \"...\") "
        "so a version bump alone produces the right filename"
    )
    return match.group(1), match.group(2)


def uninstaller_contains(path: str, identifier: str) -> bool:
    """Does scripts/uninstall.sh remove this cask path? Its own spelling uses shell variables."""
    script = read(UNINSTALLER)
    shell_path = "$HOME" + path.removeprefix("~")
    candidates = {shell_path, shell_path.replace(identifier, "$_bid")}
    return any(candidate in script for candidate in candidates)


def test_download_url_targets_the_official_release_asset() -> None:
    template = cask_url_template()
    prefix = "https://github.com/unslothai/unsloth/releases/download/v#{version}/"
    assert template.startswith(prefix), (
        f"the cask url must download from the official repo's tag directory ({prefix}); got "
        f"{template!r}"
    )

    basename = template[len(prefix):]
    assert re.fullmatch(r"Unsloth-Desktop-#\{[^}]+\}-MacOS\.dmg", basename), (
        "the DMG basename template must stay Unsloth-Desktop-<asset version>-MacOS.dmg, the "
        f"name release-desktop.yml stages; got {basename!r}"
    )

    workflow = read(RELEASE_WORKFLOW)
    assert "base_name = f'Unsloth-Desktop-{os.environ[\"ASSET_VERSION\"]}'" in workflow
    assert "f'{base_name}-MacOS.dmg'" in workflow, (
        "release-desktop.yml no longer stages the DMG as <base_name>-MacOS.dmg; update the "
        "cask url to match"
    )


def test_ruby_tr_reproduces_the_workflow_asset_version_rule() -> None:
    from_spec, to_spec = cask_tr_arguments()
    assert expand_tr_set(from_spec) == [".", "-"], (
        f"expected the cask to translate '.' and '-'; {from_spec!r} expands to "
        f"{expand_tr_set(from_spec)}"
    )
    assert set(expand_tr_set(to_spec)) == {"_"}, (
        f"the tr replacement set must be underscores only; got {to_spec!r}"
    )

    transform = workflow_asset_version_transform()
    for version in SAMPLE_VERSIONS:
        assert ruby_tr(version, from_spec, to_spec) == transform(version), (
            f"version.tr({from_spec!r}, {to_spec!r}) disagrees with release-desktop.yml's "
            f"ASSET_VERSION rule for {version!r}: the cask would build a url that 404s"
        )

    # The one that is actually published today, spelled out.
    assert ruby_tr("0.1.800-beta", from_spec, to_spec) == "0_1_800_beta"


def test_app_stanza_matches_the_tauri_product_name() -> None:
    product_name = tauri_config()["productName"]
    assert stanza("app") == f"{product_name}.app", (
        f"the cask installs {stanza('app')!r} but tauri.conf.json builds "
        f"{product_name}.app; Homebrew would fail to find the bundle"
    )
    assert stanza("name") == product_name, (
        f"the cask's `name` should be the app's display name {product_name!r}"
    )


def test_bundle_identifier_is_the_one_tauri_ships() -> None:
    identifier = tauri_config()["identifier"]

    quit_match = re.search(r'uninstall\s+quit:\s*"([^"]+)"', strip_ruby_comments(cask_source()))
    assert quit_match, f"{CASK.name} must quit the running app before removing it"
    assert quit_match.group(1) == identifier, (
        f"`uninstall quit:` is {quit_match.group(1)!r} but tauri.conf.json's identifier is "
        f"{identifier!r}; the running app would not be asked to quit"
    )

    library_paths = [path for path in cask_zap_paths() if path.startswith("~/Library/")]
    assert library_paths, "expected the zap list to cover the app's ~/Library state"
    for path in library_paths:
        assert identifier in path, (
            f"zap path {path!r} is under ~/Library but is not keyed by the bundle identifier "
            f"{identifier!r}"
        )


def test_livecheck_follows_the_configured_updater_endpoint() -> None:
    endpoints = tauri_config()["plugins"]["updater"]["endpoints"]
    assert len(endpoints) == 1, (
        f"expected a single updater endpoint to livecheck against, got {endpoints}"
    )

    body = strip_ruby_comments(cask_source())
    block = re.search(r"livecheck do\n(.*?)\n\s*end\n", body, re.DOTALL)
    assert block, f"{CASK.name} must carry a `livecheck do ... end` block"
    urls = re.findall(r'^\s*url\s+"([^"]+)"', block.group(1), re.MULTILINE)
    assert urls == [endpoints[0]], (
        f"livecheck must follow the updater manifest {endpoints[0]!r} (its platform urls are "
        f"pinned to the release that built them, unlike /releases/latest); got {urls}"
    )
    assert ":github_latest" not in body, (
        ":github_latest would offer a version whose .dmg is still missing, because a unified "
        "release becomes /releases/latest before its desktop bundles finish uploading"
    )


def test_every_zap_path_is_also_removed_by_the_shell_uninstaller() -> None:
    identifier = tauri_config()["identifier"]
    missing = [
        path for path in cask_zap_paths() if not uninstaller_contains(path, identifier)
    ]
    assert missing == [], (
        f"these zap paths are not removed by scripts/uninstall.sh: {missing}. The two "
        "uninstall routes must stay in lockstep, so add them there or drop them from the cask"
    )


def test_zap_never_deletes_the_shared_hugging_face_cache() -> None:
    offenders = [path for path in cask_zap_paths() if "huggingface" in path]
    assert offenders == [], (
        f"{offenders} must not be zapped: ~/.cache/huggingface is a multi-gigabyte cache "
        "shared with every other Hugging Face tool, and scripts/uninstall.sh preserves it "
        "on purpose"
    )


def test_cask_never_names_a_retired_release_channel() -> None:
    source = cask_source()
    for forbidden in ("unsloth-staging-2", "desktop-latest", "desktop-v"):
        assert forbidden not in source, (
            f"{CASK.name} refers to {forbidden!r}: desktop releases ship on the repo's "
            "unified v... tags from unslothai/unsloth, with discovery through latest.json"
        )


def test_version_and_checksum_have_publishable_shapes() -> None:
    checksum = stanza("sha256")
    assert re.fullmatch(r"[0-9a-f]{64}", checksum), (
        f"sha256 must be a 64-character lowercase hex digest, got {checksum!r}; regenerate "
        "with `curl -fL <dmg url> | shasum -a 256`"
    )

    version = stanza("version")
    pattern = workflow_semver_pattern()
    assert re.fullmatch(pattern, f"v{version}"), (
        f"version {version!r} is not a tag release-desktop.yml would accept (its rule is "
        f"{pattern!r}), so no such release can exist"
    )
    assert not version.startswith("v"), (
        f"version {version!r} must not carry the leading v; the url template adds it"
    )
    assert version.endswith("-beta"), (
        f"version {version!r} should be on the shipping -beta line the release workflow "
        "promotes to /releases/latest"
    )
