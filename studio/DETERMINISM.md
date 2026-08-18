# What a desktop build pins, and what it does not

A released `.dmg` is a thin launcher: it ships `install.sh`, and the first launch
builds the real environment under `~/.unsloth`. So "does this build reproduce" is two
questions, and only the second one is about the build:

1. **Same artifact → same environment.** Does one `.dmg` install the same thing today and
   in three months? This file is about that question.
2. **Same source → same artifact.** Whether two builds of one commit produce identical
   bytes. Not addressed here; see the note at the end.

Before the pins described below, the answer to (1) was no, and not marginally: the app
installed the `unsloth` wheel with no version constraint, and the requirements and
constraints files that pin everything else *ship inside that wheel*. The dependency
definition an install obeyed was therefore whatever PyPI served that day, not what the
build was tested against. llama.cpp and whisper.cpp separately resolved "newest published
release" at install time.

## What is pinned

| Component | Pinned to | Where |
| --- | --- | --- |
| `unsloth` backend | the exact version the desktop build was released with | stamped by `release-desktop.yml` into `UNSLOTH_DESKTOP_BACKEND_VERSION`, read via `option_env!` in `src-tauri/src/install.rs` and `update.rs`, applied by `install.sh` and `install_python_stack.py` |
| llama.cpp | `b10472-mix-4b653db` | `prebuilt_release_pins.json` |
| whisper.cpp | `v1.9.2-unsloth.11` | `prebuilt_release_pins.json` |
| stable-diffusion.cpp | `master-813-bfbef5b-u13b9d92` | `install_sd_cpp_prebuilt.py` (`DEFAULT_TAG`) |
| Node | `24.18.0` | `node_prebuilt_pins.json` |
| CPython | `3.13` via a uv-managed build, from pinned uv `0.12.1` | `install.sh` |

llama.cpp and whisper.cpp must stay paired: whisper's slim bundles need the ggml runtime
from the llama release its manifest names in `paired_llama_tag`. A test enforces this.

## What still floats

Pinning `unsloth==X` fixes *which* pin set applies. It does not make that set a lockfile,
and there is no lockfile for the Python stack — no `uv.lock`, no hashes, and
`install_python_stack.py` deliberately clears `PIP_REQUIRE_HASHES`/`UV_REQUIRE_HASHES`
because the shipped requirements files carry no hashes to satisfy them.

So these still resolve fresh at install time:

- **Every transitive dependency** of any step not installed with `--no-deps`.
- **`unsloth-zoo`** — a floor, not a pin. There is no stamped zoo version; the exact
  `unsloth` constrains it only through its own metadata.
- **Ranges in the requirements files** — `torch>=2.4,<2.12` against a GPU-chosen index,
  `huggingface-hub>=1.23,<2.0`, and all of `single-env/data-designer-deps.txt`.
- **`pip` itself**, upgraded to latest during bootstrap.
- **`mlx`, `mlx-lm`, `mlx-vlm`, `mlx-metal`** on Apple Silicon — installed `--upgrade`,
  unbounded.
- **`triton_kernels`**, from the moving branch `release/3.6.x`, and **ROCm
  `bitsandbytes`**, from the mutable tag `continuous-release_main`.

Honest summary: this moves the app from "a different stack most months" to "the same
direct dependencies, with drifting transitives". It is a large improvement and not a
guarantee. Closing the rest means a real lockfile with hashes.

## Integrity is a separate axis from determinism

Pinning decides *which* artifact is fetched. It is not by itself a trust anchor.

- `node_prebuilt_pins.json` freezes per-asset sha256 **in-tree**, so the digest is
  reviewed code. It refuses unpinned versions outright. This is the strongest anchor and
  the model the others should follow.
- llama.cpp and whisper.cpp archives are checked against their release's own checksum
  index, fetched from the same release over the same TLS channel. That is
  tamper-*consistency* between index, manifest and archive — not independent attestation.
  `checksum_index_sha256` in the pins file is recorded for human verification and is **not
  enforced**; do not read it as digest pinning.
- stable-diffusion.cpp is the weakest: it verifies against the `digest` field of the same
  GitHub API response that supplied the URL, warns and proceeds when absent, and silently
  falls back to `latest` if its pinned tag 404s.

No component verifies a signature.

## Escape hatches

All of these restore pre-pin behaviour, and each turns off detection and installation
together so the update banner never offers what the installer would refuse:

| Variable | Effect |
| --- | --- |
| `UNSLOTH_PREBUILT_ALLOW_LATEST=1` | llama.cpp and whisper.cpp track newest again |
| `UNSLOTH_LLAMA_RELEASE_TAG=<tag>` | install that llama release instead |
| `UNSLOTH_WHISPER_RELEASE_TAG=<tag>` | install that whisper release instead |
| `UNSLOTH_SD_CPP_TAG=<tag>` | override sd.cpp (empty tracks latest) |
| `UNSLOTH_NODE_ALLOW_UNVERIFIED=1` | allow a Node version with no in-tree digest |
| `UNSLOTH_BACKEND_VERSION=<ver>` / `--backend-version` | pin the backend by hand |

A `curl | sh` CLI install sets none of these and none of the pins, so it tracks latest
exactly as it always has. Only release-stamped desktop builds pin the backend; an
unstamped dev build sets nothing, which is why the plumbing reads `option_env!` directly
rather than the `MIN_DESKTOP_BACKEND_VERSION` floor.

## Consequences worth knowing

- **The older-release walk-back is off under a pin.** A host that previously relied on
  walking back to skip a too-new prebuilt now fails to a source build instead. Silently
  installing a different release is the drift the pin exists to remove, so the walk stays
  off; the error names the pin, this file, and the escape hatches.
- **`setup.sh`'s fast path stops firing.** It skips the dependency pass when the installed
  version equals PyPI latest. Under a pin those usually differ, so updates do more work.
- **A broken upstream release is no longer auto-healed.** Tracking latest meant the next
  launch could repair itself past a bad release. Pinning trades that for determinism: a
  bad pinned release needs a new desktop build.

## Bumping the pins

`prebuilt_release_pins.json` carries the procedure in its own `comment`. In short: pick the
new tag per component, re-record `checksum_index_sha256`, update `pinned_at_utc`, and
confirm whisper's `paired_llama_tag` still equals the llama tag. The backend version needs
no manual step — it follows `pypi_version` at release time.

Changes to the pins file trigger `clean-machine-install-ci.yml`, which installs on a
toolchain-stripped machine, so a bump is exercised rather than assumed.

## On the build itself

Question (2) — same source, same bytes — is partly addressed.

The build inputs are pinned: SHA-pinned actions, an exact Tauri CLI, digest-pinned
packaging tools, committed lockfiles, and now an exact rustc (`rust-toolchain.toml`, with
the release workflow passing the same version so targets install against it), an exact
Node, and `npm ci` so neither npm lockfile can be rewritten mid-build. `fix-path-env` was
a git dependency with no `rev`, held only by the lockfile; it is pinned to a commit. A
guard snapshots all three lockfile digests and re-checks them after every platform build,
so a build that rewrote one fails rather than shipping.

Published assets carry provenance. `actions/attest-build-provenance` runs over the staged
set after validation and before upload, so a third party can verify an artifact came from
this source and this workflow:

```sh
gh attestation verify Unsloth-Desktop-<version>-MacOS.dmg -R unslothai/unsloth
```

A `build-inputs.json` asset records the commit, toolchain versions, lockfile digests,
runner images and pinned tool digests — the things that otherwise exist only in run logs
that expire. It asserts the running rustc matches `rust-toolchain.toml`, so a pin that
silently failed to apply fails the release.

**What is still not reproducible.** Signed artifacts can never be byte-identical:
`codesign` embeds a secure timestamp, Apple's stapled notarization ticket is
per-submission, and the minisign `.sig` files carry timestamps of their own. The Windows
installer embeds a WebView2 bootstrapper fetched at build time and not digest-pinned.
`SOURCE_DATE_EPOCH` is exported from the tag commit as groundwork, but nothing verifies
byte reproducibility and tauri-bundler's handling of it is unconfirmed. The release
workflow also rewrites `Cargo.toml`/`Cargo.lock` to the dispatched version before
building, so a checkout of the tag does not reproduce the built tree without replaying
that mutation — deterministic and recoverable (`tag.removeprefix('v')`, and
`latest.json` carries both it and `pypi_version`), but not automatic.

The realistic remaining targets are the unsigned macOS payload compared via CodeDirectory
page hashes, and the Linux `.deb` and AppImage under `SOURCE_DATE_EPOCH`. Neither is
implemented, and neither has been measured.
