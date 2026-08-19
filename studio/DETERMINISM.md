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
| `triton_kernels` | commit `7c56a5e40f7fd928dfd5c72902d5def0097db73a` (tip of `release/3.6.x` on 2026-08-19) | `backend/requirements/triton-kernels.txt` |
| ROCm `bitsandbytes` | `==0.50.1` on PyPI, replacing the `continuous-release_main` tag | `install_python_stack.py` (`_BNB_ROCM_PINNED_SPEC`) |
| bootstrap `pip` | `==26.2.1` | `install_python_stack.py` (`_PIP_BOOTSTRAP_VERSION`) |
| `mlx`, `mlx-metal` | `<0.33` — a **ceiling, not a pin** | `install_python_stack.py` (`_MLX_STACK_SPECS`) |
| `mlx-lm` | `<0.32` — a ceiling | same |
| `mlx-vlm` | `<0.7` — a ceiling | same |

The four MLX bounds are ceilings, not exact versions: mlx is pre-1.0, so its breaking
changes land in minor releases, and `<next-minor` is the window that keeps patch fixes
reaching users. They were set one minor above the releases the stack resolved to on
2026-08-19 (mlx/mlx-metal 0.32.1, mlx-lm 0.31.3, mlx-vlm 0.6.15). A patch release inside
a window still arrives unreviewed — that is the deliberate trade, because an exact `==`
strands Apple Silicon on a broken combination as readily as no ceiling does.

There are deliberately **no floors**. `--upgrade` already takes the newest admissible
version, so a floor cannot change what a healthy install resolves to; it can only change
the failing case, and there it makes things worse. Without uv there is no `UV_OVERRIDE`,
so mlx-vlm's own `transformers>=5.14.0` collides with the `transformers==5.5.0` in
`constraints.txt`, and the resolver backtracking to an older mlx-vlm is the only thing
keeping that install from failing outright. A floor would turn a degraded-but-installed
stack — which `_report_mlx_stack_health()` already reports and the startup self-heal
already retries — into a fatal macOS install error. A test asserts the floors stay absent,
so adding one has to be a deliberate act.

`mlx` declares `mlx-metal==<its own version>` on Darwin, so those two ceilings must stay
identical or the pair is unsatisfiable; a test enforces that, and that each ceiling is the
minor after the release it was resolved against.

llama.cpp and whisper.cpp must stay paired: whisper's slim bundles need the ggml runtime
from the llama release its manifest names in `paired_llama_tag`. A test enforces this.

## The torch-independent stack installs from hash-verified locks

`studio/backend/requirements/locks/` holds six machine-generated locks — **289 pinned
requirements carrying 6,157 sha256 digests** — and `install_python_stack.py` installs those
steps with `--require-hashes`. Every wheel they cover is verified against a digest that was
reviewed in this repository, so transitive drift stops for that part of the closure.

| lock | source | packages | digests |
| --- | --- | --- | --- |
| `studio.lock.txt` | `studio.txt` | 144 | 3,343 |
| `data-designer-deps.lock.txt` | `single-env/data-designer-deps.txt` | 81 | 1,868 |
| `no-torch-runtime.lock.txt` | `no-torch-runtime.txt` (`--no-deps`) | 47 | 831 |
| `extras-no-deps.lock.txt` | `extras-no-deps.txt` (`--no-deps`) | 12 | 105 |
| `data-designer.lock.txt` | `single-env/data-designer.txt` (`--no-deps`) | 4 | 8 |
| `pip-bootstrap.lock.txt` | the pinned bootstrap `pip` | 1 | 2 |

Regenerate with `bash scripts/gen_python_locks.sh`. It reads the uv version `install.sh`
pins and **refuses any other uv**, because a lock built by a different resolver is not the
lock users get. `--exclude-newer` freezes the index, so regeneration reproduces the same
resolution rather than drifting; bumping that cutoff is the deliberate act of taking
upstream releases. Each compile also passes the other shipped requirements files as
constraints — without that, independent locks silently overrode pins from other files in
the same environment.

`UNSLOTH_PYTHON_NO_LOCK=1` reverts to unlocked installs. A missing lock warns loudly and
falls back, so a wheel built before the locks existed still installs. A lock is skipped
below its recorded Python floor, so a 3.9 host stays on today's path rather than being
handed a 3.10 resolution.

Two guards, because neither alone is enough: `tests/security/test_python_locks.py` catches
a missing, malformed or ranged requirement, and `.github/workflows/python-lock-freshness.yml`
regenerates and diffs, which is the only thing that catches a digest that is well-formed
but *wrong*.

## What still floats

The locks above cover the torch-independent steps. They are not a whole-stack lockfile,
and these still resolve fresh at install time:

- **Everything torch-bound.** `extras.txt` is deliberately unlocked: a with-deps universal
  resolution pins torch, torchvision, torchaudio, triton and fifteen `nvidia-*` packages
  from PyPI, which would override the index chosen from detected hardware. Locking it means
  a per-family lock matrix — roughly 23 files — and that carries its own costs: each family
  collapses to one exact torch version, which ends the keep-previous-torch self-healing, and
  the AMD per-arch indexes give no immutability guarantee, so those locks hard-fail whenever
  a wheel is republished in place.
- **`diffusers-pin.txt`** — unlocked. Note before locking it that a digest over a
  GitHub-generated source archive is a digest over bytes GitHub does not promise to keep
  stable.
- **`triton-kernels.txt`** — a git requirement cannot carry a hash, and `--require-hashes`
  rejects it. The commit pin is the anchor instead, and being content-addressed it is a
  stronger one than a version.
- **`unsloth-zoo`** — a floor, not a pin. There is no stamped zoo version; the exact
  `unsloth` constrains it only through its own metadata.
- **`install.sh`'s own `no-torch-runtime.txt` installs** on the fresh `--no-torch` legs run
  unlocked; the update path through `install_python_stack.py` uses the lock.
- **`pytorch_tokenizers`**, carved out into `locks/extras-no-deps.unlocked.txt` and
  installed unlocked on purpose: it has no musllinux wheel at its cap and its arm64 wheel is
  `macosx_14_0`, and neither axis has a PEP 508 marker, so pinning one version pushes musl
  and macOS 13 hosts onto the sole sdist and a cmake build.
- **Ranges in the still-unlocked requirements files** — `torch>=2.4,<2.12` against a
  GPU-chosen index, and the torch followers keyed off whatever torch resolves to.
- **Everything below the four MLX ceilings** — the newest admissible patch, and on the
  non-uv fallback path anything the resolver backtracks to. Per the table above, this is
  deliberate.
- **The MLX startup self-heal.** `backend/utils/mlx_repair.py` reinstalls
  `mlx`/`mlx-lm`/`mlx-vlm` by *floor* with `--upgrade`, so a host whose MLX stack is
  actually blocked can still be walked past the installer's ceilings at launch. That
  is the auto-heal working as designed — it runs only when the stack is already broken
  — but it means the ceilings hold for healthy installs, not for repaired ones.
- **The bitsandbytes fallback on a ROCm host** is still the shared `>=0.50.0` floor.
  It fires only when the pinned release cannot install; `install.sh` and the `amd`
  extra in `pyproject.toml` share that constant.
- **`install.sh`'s own ROCm bitsandbytes step**, which still installs the
  `continuous-release_main` wheel. The Python stack pass runs after it and
  force-reinstalls the pinned release over the top, so the *installed* version is
  pinned either way — but the shell installer still fetches a mutable wheel first.

### One behaviour change the locks introduced

Both with-deps locks pin `cryptography==48.0.1`, where Linux, Windows and macOS arm64
resolve `50.0.0` today. The `<49` cap in `constraints.txt` exists only because 49.0.0
dropped the `macosx_10_9_universal2` wheel that Intel Macs need — but `cryptography` arrives
transitively (via `authlib`, `joserfc`, `pyjwt`, `secretstorage`), so it cannot be carved out
of a hashed closure, and uv collapses the darwin-scoped cap onto every marker fork. The
effect is that three platforms are held two majors back on a security-relevant library in
exchange for a hashed closure. The fix is to lift the cap and regenerate once upstream ships
an x86_64 macOS wheel again, or to move to per-platform locks.

Honest summary: the app has gone from "a different stack most months" to a hash-verified
closure for the torch-independent majority, with torch and its followers still resolving
fresh. That is a large improvement and still not a whole-stack guarantee.

## Integrity is a separate axis from determinism

Pinning decides *which* artifact is fetched. It is not by itself a trust anchor.

- `node_prebuilt_pins.json` freezes per-asset sha256 **in-tree**, so the digest is
  reviewed code. It refuses unpinned versions outright. This is the strongest anchor.
- llama.cpp and whisper.cpp archives are checked against their release's own checksum
  index — and that index is now checked against `checksum_index_sha256` in
  `prebuilt_release_pins.json`, which **is enforced** at install time. The chain is
  reviewed in-tree digest → checksum index → per-archive sha256 → archive on disk, so
  every installed byte traces back to a digest that went through code review. Without
  that first link the index and the archives it vouches for arrive over the same TLS
  channel from the same release, which is tamper-*consistency*, not attestation.
  Enforcement covers both routes to the index (the GitHub API path and the download-host
  fast path), and applies **only to the release the pins file names**: an env-overridden
  tag, a different `--published-repo`, or either escape hatch means no in-tree digest
  exists for that release, so verification is skipped rather than failed. A pin bump that
  forgets to re-record the digest fails closed.
- stable-diffusion.cpp verifies against the `digest` field of the same GitHub API
  response that supplied the URL, and now **fails closed**: a missing digest and an
  unrecognised algorithm are errors, not warnings, and a pinned tag that no longer
  resolves is an error rather than a silent install of `latest`. Its remaining weakness
  is the one this cannot fix from here: the digest is still same-origin, so it is
  tamper-consistency only. It has no in-tree digest of its own, because it publishes no
  checksum-index asset to pin. Because sd.cpp installs lazily at the first image
  generation, these failures surface there rather than during bootstrap.

- `triton_kernels` is the one Python dependency whose pin *is* a digest. A full git
  commit sha is content-addressed over the whole tree, and it lives in-tree in
  `triton-kernels.txt`, so git rejects a fetch whose contents do not hash to it. That
  makes it a stronger anchor than any version pin here — which is why the file also
  insists on the full 40 characters, an abbreviation being a prefix match rather than
  an identity.
- The pinned PyPI versions (`bitsandbytes`, `pip`, and the MLX windows) are **not**
  integrity anchors. pip checks each wheel against the hash the index served alongside
  it, which is same-origin: it detects corruption in transit, not a compromised index.
  The in-tree part is only the version number.

No component verifies a signature, and apart from the `triton_kernels` commit, only
Node and (one hop removed) llama.cpp / whisper.cpp verify anything against a digest that
lives in this repository.

## Escape hatches

All of these restore pre-pin behaviour, and each turns off detection and installation
together so the update banner never offers what the installer would refuse:

| Variable | Effect |
| --- | --- |
| `UNSLOTH_PREBUILT_ALLOW_LATEST=1` | llama.cpp, whisper.cpp and sd.cpp track newest again — and with the pin off, the in-tree checksum-index digest no longer applies. It also reverts the three first-run Python pins that live in `install_python_stack.py`: bootstrap `pip` goes back to `--upgrade pip`, the MLX stack to four bare names, and ROCm `bitsandbytes` to the `continuous-release_main` wheel first with the `>=0.50.0` floor behind it. (It does not reach `triton_kernels`: a requirements file cannot read the environment, so that commit is pinned unconditionally — override it by editing the file or pointing the step at your own requirement.) |
| `UNSLOTH_PREBUILT_ALLOW_UNVERIFIED=1` | keep the pinned versions but install without checking the llama/whisper checksum index against its in-tree digest, and let sd.cpp install an asset that publishes no usable digest |
| `UNSLOTH_LLAMA_RELEASE_TAG=<tag>` | install that llama release instead |
| `UNSLOTH_WHISPER_RELEASE_TAG=<tag>` | install that whisper release instead |
| `UNSLOTH_SD_CPP_TAG=<tag>` | override sd.cpp (empty tracks latest) |
| `UNSLOTH_NODE_ALLOW_UNVERIFIED=1` | allow a Node version with no in-tree digest |
| `UNSLOTH_BACKEND_VERSION=<ver>` / `--backend-version` | pin the backend by hand |

The two `ALLOW_` variables say different things and are deliberately separate:
`ALLOW_LATEST` is "I accept a different version", `ALLOW_UNVERIFIED` is "I accept bytes
nothing checked". Neither excuses a digest that *was* checked and disagreed — a real
mismatch always stops the install.

**A `curl | sh` CLI install is no longer entirely unpinned.** It still sets none of
these variables, so the *backend version* and the ggml-family prebuilts track latest
exactly as they always have — only release-stamped desktop builds pin the backend, and
an unstamped dev build sets nothing, which is why the plumbing reads `option_env!`
directly rather than the `MIN_DESKTOP_BACKEND_VERSION` floor. But the first-run Python
pins — `triton_kernels`, ROCm `bitsandbytes`, bootstrap `pip`, the MLX ceilings — live in
`install_python_stack.py` and the requirements tree, so they are properties of the
*source* rather than of a release stamp and apply to every install: CLI, CI, dev build
and `.dmg` alike. That is deliberate — a moving git branch and a republished tag
are supply-chain surface for CLI users too — and it is a change in what a CLI install
gets: pinned `pip`, bounded MLX, `triton_kernels` at one commit, and ROCm
`bitsandbytes` at one release instead of whatever the rolling tag held that day.

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
- **A release that is retagged or republished now breaks the install rather than
  changing it.** Re-uploading assets under a pinned tag changes the checksum index, which
  no longer matches the in-tree digest; for sd.cpp, deleting a pinned tag no longer
  resolves to `latest`. Both are deliberate: a pin that quietly moved is the failure mode
  worth catching, and the errors name the pin, the file and the hatch.
- **ROCm hosts stop tracking bitsandbytes' main branch.** They used to install the
  `continuous-release_main` wheel, which is republished on every merge, so they picked up
  every ROCm change (and every regression) within a day. They now get one release. The
  substitution is not a downgrade today: all five `libbitsandbytes_rocm*.so` and both
  `libbitsandbytes_xpu*.so` are byte-identical between that wheel and PyPI `0.50.1`
  (verified per-member sha256, 2026-08-19); the rolling wheel's only functional
  difference is in `backends/triton/kernels_4bit.py`, which bnb registers for XPU alone
  and never for the HIP path. That equality is a fact about today's wheel, not a
  guarantee — a future ROCm fix will need this pin bumped, exactly as `.dmg` prebuilts do.
- **`triton_kernels` is now built from one commit on `release/3.6.x`, not its tip.** It
  is a training speedup installed `--no-deps` and skipped entirely without git, so the
  blast radius of a stale pin is performance, not function. The comment in the file
  keeps the branch name, because a bare sha is otherwise unbumpable.
- **The MLX ceilings will eventually hold users back.** mlx and mlx-vlm ship minor
  releases frequently; when they do, Apple Silicon stops receiving them until the window
  moves. That is the point — `_report_mlx_stack_health()` exists because a newer
  mlx-lm/mlx-vlm against the Studio transformers pin has repeatedly blacked out Train —
  but it does mean the windows need looking at as part of a release, not once a year.
- **The pinned `pip` ages.** Nothing needs bootstrap pip to be newest (uv venvs need
  *a* pip; the behaviours relied on are years old), and `ensurepip` is untouched because
  the pip CPython bundles is already fixed by the pinned CPython. But a future Python
  release can need a newer pip than this line names, and the symptom would be a
  bootstrap failure rather than a wrong install.

## Bumping the pins

`prebuilt_release_pins.json` carries the procedure in its own `comment`. In short: pick the
new tag per component, re-record `checksum_index_sha256`, update `pinned_at_utc`, and
confirm whisper's `paired_llama_tag` still equals the llama tag. Re-recording the digest is
no longer optional bookkeeping: it is enforced, so a bump that skips it fails every install
closed until it is corrected. The backend version needs no manual step — it follows
`pypi_version` at release time.

Changes to the pins file trigger `clean-machine-install-ci.yml`, which installs on a
toolchain-stripped machine, so a bump is exercised rather than assumed.

The four first-run Python pins live next to what they pin, and each carries its own
bump recipe in a comment:

| Pin | How to re-resolve it |
| --- | --- |
| `triton_kernels` | `git ls-remote https://github.com/triton-lang/triton refs/heads/release/3.6.x` — paste the full sha into `triton-kernels.txt` |
| ROCm `bitsandbytes` | pick the newest PyPI release, confirm it still ships `libbitsandbytes_rocm*.so` for the arch families in `_GFX_TO_AMD_INDEX_ARCH`, move `_BNB_ROCM_PINNED_SPEC` |
| bootstrap `pip` | `curl -sL https://pypi.org/pypi/pip/json` → `.info.version` |
| MLX windows | move each ceiling to the minor after the current release; keep `mlx` and `mlx-metal` identical |

`tests/studio/install/test_first_run_install_pins.py` guards all four: it fails on any
requirements file that references a git ref which is not a full 40-character commit sha,
on a malformed or floor-violating version pin, on an MLX window that excludes the release
it was resolved against or admits the next minor, and — driving `install_python_stack()`
and `_ensure_rocm_torch()` — on argv that reaches pip without the pinned values.
The bump table above is only bookkeeping; those tests are what actually fails.

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

The Windows WebView2 bootstrapper is pinned too. `tauri-bundler` fetched it with an
unverified download — the only such fetch left in the Windows bundler, while its siblings
NSIS and `nsis_tauri_utils.dll` both went through `download_and_verify` — and embedded it in
the NSIS installer, which with `createUpdaterArtifacts` is both the fresh installer and the
auto-update payload. The build now fetches it from the immutable per-version URL, verifies a
pinned sha256, fails closed, and seeds the bundler's tools cache; a second step then reads
`WEBVIEW2BOOTSTRAPPERPATH` out of the rendered `installer.nsi` and hashes what `makensis`
actually embedded, so a CLI bump that moves that cache fails the release rather than quietly
reverting to an unverified download. The rotating `go.microsoft.com` fwlink is deliberately
*not* the pinned URL, since it is repointed on Microsoft's schedule.

This pins the 1.8 MB installer stub, **not** the runtime: it is Edge Update Setup, whose job
is to fetch the current Evergreen runtime at install time, so users keep receiving WebView2
security patches. `webviewInstallMode` stays `embedBootstrapper` and is guarded, because
`fixedRuntime` would pin the runtime itself and cut users off from those updates — a product
decision, not a cleanup.

**What is still not reproducible.** Signed artifacts can never be byte-identical:
`codesign` embeds a secure timestamp, Apple's stapled notarization ticket is
per-submission, and the minisign `.sig` files carry timestamps of their own.
`SOURCE_DATE_EPOCH` is exported from the tag commit as groundwork, but nothing verifies
byte reproducibility and tauri-bundler's handling of it is unconfirmed. The release
workflow also rewrites `Cargo.toml`/`Cargo.lock` to the dispatched version before
building, so a checkout of the tag does not reproduce the built tree without replaying
that mutation — deterministic and recoverable (`tag.removeprefix('v')`, and
`latest.json` carries both it and `pypi_version`), but not automatic.

The realistic remaining targets are the unsigned macOS payload compared via CodeDirectory
page hashes, and the Linux `.deb` and AppImage under `SOURCE_DATE_EPOCH`. Neither is
implemented, and neither has been measured.
