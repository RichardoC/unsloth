# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Am I running inside the runtime baked into Unsloth.app?

The macOS desktop app ships a complete runtime under
``Unsloth.app/Contents/Resources/runtime/`` -- its own CPython (``python/``), its
own ``site-packages/``, its own ``llama.cpp/``, ``whisper.cpp/``, ``node/`` and
``oxc-node-modules/``, plus a ``BUNDLE_MANIFEST.json`` receipt -- and launches
this backend as that interpreter (``studio/src-tauri/src/bundled_runtime.rs``).
Two things in the backend have to know:

  * **nothing may be written there.** The bundle is code-signed and, once dragged
    to ``/Applications``, root-owned. The in-app llama.cpp / whisper.cpp updaters
    resolve their install root from ``UNSLOTH_LLAMA_CPP_PATH`` /
    ``UNSLOTH_WHISPER_CPP_PATH``, which the app points at the bundled copies, so
    without this seam they would treat a signed directory as a managed tree and
    offer an update that can only fail -- or worse, succeed and invalidate the
    signature. See ``utils/prebuilt/update_flow.immutable_runtime_root``.
  * **the OXC validator's ``node_modules`` moved.** They sit at
    ``runtime/oxc-node-modules`` rather than inside the validator's own
    directory, so Node's upward walk cannot find them and the path must be handed
    over (``core/data_recipe/local_callable_validators``).

Stdlib only -- safe to import without the rest of the backend.

Mirrors ``unsloth_cli/_bundled_runtime.py``, which is where the reasoning lives
in full. The logic is duplicated rather than shared for the same reason
``utils/host_policy.py`` duplicates ``unsloth_cli/_tool_policy.py``: the backend
is self-contained (see run.py's "can be moved to any directory") and runs from
environments with no ``unsloth_cli`` on ``sys.path``. Keep the two in sync;
``tests/studio/test_bundled_runtime_seam.py`` calls both through the same
scenarios and asserts they agree.

The short version of the predicate, because it is a security boundary: a bare
read of ``UNSLOTH_BUNDLED_SITE_PACKAGES`` is NOT enough. A variable left behind
in a shell profile would otherwise persuade an ordinary install that it is a
signed app bundle. All three must hold:

  1. the variable names an existing directory;
  2. its parent carries ``BUNDLE_MANIFEST.json`` (written by
     ``scripts/build_macos_runtime.sh``; the Rust health check refuses to launch
     a payload without it);
  3. ``sys.prefix`` resolves inside that same parent -- the interpreter running
     this code is the bundle's own, which no environment variable can arrange.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

#: ``bundled_runtime.rs``'s ``SITE_PACKAGES_ENV``.
SITE_PACKAGES_ENV = "UNSLOTH_BUNDLED_SITE_PACKAGES"

#: ``bundled_runtime.rs``'s ``MANIFEST_NAME``.
BUNDLE_MANIFEST_NAME = "BUNDLE_MANIFEST.json"


def _resolved(path: Path) -> Optional[Path]:
    try:
        return path.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def bundled_runtime_root() -> Optional[Path]:
    """``Contents/Resources/runtime`` when this process IS the bundled runtime,
    else ``None``. Never raises: a wrong answer must degrade to "no bundle",
    which is the behaviour every non-bundled install already has."""
    raw = (os.environ.get(SITE_PACKAGES_ENV) or "").strip()
    if not raw:
        return None
    site_packages = _resolved(Path(raw).expanduser())
    if site_packages is None or not site_packages.is_dir():
        return None
    root = site_packages.parent
    if not (root / BUNDLE_MANIFEST_NAME).is_file():
        return None
    prefix = _resolved(Path(sys.prefix))
    if prefix is None:
        return None
    if prefix != root and root not in prefix.parents:
        return None
    return root


def bundled_oxc_node_modules() -> Optional[Path]:
    """``runtime/oxc-node-modules`` when this process is the bundled runtime and
    the directory is there, else ``None``."""
    root = bundled_runtime_root()
    if root is None:
        return None
    modules = root / "oxc-node-modules"
    return modules if modules.is_dir() else None


def path_is_inside_bundled_runtime(path) -> bool:
    """Whether *path* lives inside this process's bundled runtime -- i.e. whether
    writing there would touch the signed app bundle."""
    root = bundled_runtime_root()
    if root is None or path is None:
        return False
    candidate = _resolved(Path(path))
    if candidate is None:
        return False
    return candidate == root or root in candidate.parents
