# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Am I the interpreter baked into Unsloth.app?

The macOS desktop app ships a complete runtime inside the bundle::

    Unsloth.app/Contents/Resources/runtime/
      python/                relocatable CPython; python/bin/python3 is the interpreter
      site-packages/         every installed distribution (unsloth_cli/, studio/, deps)
      llama.cpp/  whisper.cpp/  node/  oxc-node-modules/
      BUNDLE_MANIFEST.json   what the build assembled, with digests

and launches the backend as that interpreter running ``-X utf8 -I -B -c
<bootstrap>``, with ``UNSLOTH_BUNDLED_SITE_PACKAGES`` naming the ``site-packages``
directory the bootstrap appends to ``sys.path`` (``studio/src-tauri/src/
bundled_runtime.rs``). ``-I`` implies ``-E``, so ``PYTHONHOME`` / ``PYTHONPATH``
are discarded and that variable is the only channel there is.

Everything in the CLI that used to ask "am I inside the managed studio venv?"
really wants to know "does the interpreter I am already running carry the
backend?", and for a bundled runtime the answer is yes even though ``sys.prefix``
is ``runtime/python`` rather than ``<STUDIO_HOME>/unsloth_studio``. That is the
one question this module answers.

Deliberately NOT a bare environment-variable read. A variable left in somebody's
shell profile -- or exported by a script that once ran the app -- would otherwise
convince an ordinary ``curl | sh`` install that it is a signed app bundle, and
send it looking for a backend that is not there. So the answer requires three
facts, of which the caller's environment supplies only the first:

  1. ``UNSLOTH_BUNDLED_SITE_PACKAGES`` is set and names a directory that exists;
  2. its parent holds ``BUNDLE_MANIFEST.json`` -- the receipt
     ``scripts/build_macos_runtime.sh`` writes and the Rust health check
     (``BundledRuntime::health``) refuses to launch without, so requiring it here
     cannot invent a failure the app would otherwise have survived;
  3. ``sys.prefix`` resolves inside that same parent, i.e. the interpreter
     executing this code is the bundle's own.

(3) is the one that cannot be faked from a shell: an ordinary install's
``sys.prefix`` is ``<STUDIO_HOME>/unsloth_studio`` or ``/usr``, never a sibling
of the named ``site-packages``. (2) hardens (1) against pointing at some
unrelated tree that happens to contain the running interpreter. Only their
conjunction says "bundle".

Mirrored, not shared, at ``studio/backend/utils/bundled_runtime.py``: the backend
is self-contained and runs from venvs that have no ``unsloth_cli`` on
``sys.path`` (same reason ``utils/host_policy.py`` mirrors ``_tool_policy.py``).
Keep the two in sync; ``tests/studio/test_bundled_runtime_seam.py`` asserts they
agree.

Stdlib only, and never raises: a wrong answer here must degrade to "no bundle",
which is exactly today's behaviour.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

#: How ``studio/src-tauri`` tells the interpreter where the bundled
#: distributions are (``bundled_runtime.rs``'s ``SITE_PACKAGES_ENV``).
SITE_PACKAGES_ENV = "UNSLOTH_BUNDLED_SITE_PACKAGES"

#: Written by ``scripts/build_macos_runtime.sh`` at the runtime root; the Rust
#: health check will not launch a payload without it (``MANIFEST_NAME`` there).
BUNDLE_MANIFEST_NAME = "BUNDLE_MANIFEST.json"


def _resolved(path: Path) -> Optional[Path]:
    try:
        return path.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def bundled_runtime_root() -> Optional[Path]:
    """``Contents/Resources/runtime`` when this process IS the bundled runtime.

    ``None`` for every other way the CLI can be running -- ``curl | sh``
    installs, ``pip install unsloth``, dev checkouts, CI -- including when
    ``UNSLOTH_BUNDLED_SITE_PACKAGES`` is set but unaccompanied by the two facts
    the module docstring lists.
    """
    raw = (os.environ.get(SITE_PACKAGES_ENV) or "").strip()
    if not raw:
        return None
    site_packages = _resolved(Path(raw).expanduser())
    if site_packages is None or not site_packages.is_dir():
        return None
    root = site_packages.parent
    # The payload receipt. A stray variable can name any directory; it cannot
    # also put the build's manifest next to it.
    if not (root / BUNDLE_MANIFEST_NAME).is_file():
        return None
    # And the decisive one: the running interpreter has to live in that payload.
    prefix = _resolved(Path(sys.prefix))
    if prefix is None:
        return None
    if prefix != root and root not in prefix.parents:
        return None
    return root


def bundled_site_packages() -> Optional[Path]:
    """``runtime/site-packages`` when this process is the bundled runtime."""
    root = bundled_runtime_root()
    return None if root is None else root / "site-packages"


def bundled_oxc_node_modules() -> Optional[Path]:
    """``runtime/oxc-node-modules`` -- the prefetched ``node_modules`` for the OXC
    validator -- when this process is the bundled runtime and it is present.

    The bundle keeps them beside the interpreter rather than inside
    ``site-packages/studio/backend/core/data_recipe/oxc-validator/``, so Node's
    upward directory walk from the validator's own directory never reaches them
    and the path has to be handed over explicitly.
    """
    root = bundled_runtime_root()
    if root is None:
        return None
    modules = root / "oxc-node-modules"
    return modules if modules.is_dir() else None


def path_is_inside_bundled_runtime(path) -> bool:
    """Whether *path* lives inside this process's bundled runtime.

    The question every writer has to ask before it writes: the runtime is inside
    a signed, possibly root-owned ``.app``, so a write there either fails or
    succeeds and invalidates the code signature.
    """
    root = bundled_runtime_root()
    if root is None or path is None:
        return False
    candidate = _resolved(Path(path))
    if candidate is None:
        return False
    return candidate == root or root in candidate.parents
