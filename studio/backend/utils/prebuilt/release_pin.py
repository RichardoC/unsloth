# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The in-tree prebuilt release pin, read from the backend side.

``studio/prebuilt_release_pins.json`` freezes which published release each
installer installs by default, so one desktop build lands the same runtime on
any day. Everything that asks "is a newer build available?" has to compare the
installed tag against THAT tag. Before this module they asked GitHub for the
newest published release instead, which after a correct pinned install is
deliberately newer -- so the UI showed a permanent "update available" banner for
an update the installer would never actually perform.

The one fact this module answers: *the release tag the installer would install
here, right now, if the caller named none.* ``None`` means "no pin is in force",
and every caller then keeps its pre-pin behaviour (resolve GitHub's newest).

Why a second reader rather than reusing ``studio/prebuilt_core.py``'s loader:

- Import direction. ``install_llama_prebuilt.py`` already imports
  ``backend.utils.prebuilt.*``; nothing under ``studio/backend`` imports
  ``studio/prebuilt_core.py``, and prebuilt_core deliberately imports no backend
  module (it runs in a bare spawned interpreter, which is why it pastes the TLS
  gate rather than importing ``utils.native_tls``). Reaching the other way would
  invert that direction and pull an 8k-line installer module -- plus its
  import-time ``truststore.inject_into_ssl()`` -- into the long-lived server.
- Opposite failure policy, which is the substantive reason. The installer must
  fail CLOSED on a missing or malformed pins file: silently falling back to
  "newest published release" is exactly the drift the pin exists to remove, and
  it would be invisible in production. A status banner must fail OPEN: an
  unreadable pins file degrades to the pre-pin behaviour rather than breaking
  ``/api/inference/status``. ``prebuilt_core.load_release_pins`` raises by
  contract, so it cannot be the reader here.

Stdlib only, like its siblings in this package, so it imports under both roots:
``utils.prebuilt.release_pin`` in the server (sys.path root ``studio/backend``)
and ``backend.utils.prebuilt.release_pin`` from ``studio/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

# Kept in sync with prebuilt_core.RELEASE_PINS_FILENAME / RELEASE_PINS_SCHEMA_VERSION /
# ALLOW_LATEST_ENV; test_prebuilt_pin_freshness.py asserts the pair does not drift.
PINS_FILENAME = "prebuilt_release_pins.json"
PINS_SCHEMA_VERSION = 1
ALLOW_LATEST_ENV = "UNSLOTH_PREBUILT_ALLOW_LATEST"

_TRUTHY = {"1", "true", "yes", "on"}

# Parsed pins keyed by (path, mtime_ns, size): the file is in-tree and never
# changes under a running server, but an update that replaces the app directory
# can, and a stat is cheaper than a re-parse on every status poll.
_pins_memo: dict = {}


def pins_path() -> Optional[Path]:
    """Locate ``prebuilt_release_pins.json``.

    Same walk-up shape as ``update_flow.find_installer_script``: try both
    ``<root>/<file>`` and ``<root>/studio/<file>`` at every ancestor, so it
    resolves in the dev tree and in an installed Unsloth layout alike. None when
    absent (a partial install) -- the caller then tracks latest as before.
    """
    here = Path(__file__).resolve()
    for up in here.parents:
        for candidate in (up / PINS_FILENAME, up / "studio" / PINS_FILENAME):
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
    return None


def allow_latest() -> bool:
    """True when the user opted out of the pin (``UNSLOTH_PREBUILT_ALLOW_LATEST``)."""
    return os.environ.get(ALLOW_LATEST_ENV, "").strip().lower() in _TRUTHY


def load_pins() -> Optional[dict]:
    """The parsed pins manifest, or None when it is missing or unusable.

    Fails open on purpose (see the module docstring): a corrupt manifest must not
    take out the status route, it must only cost the banner its pin awareness.
    """
    path = pins_path()
    if path is None:
        return None
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    cached = _pins_memo.get("key")
    if cached == key:
        return _pins_memo.get("value")
    value: Optional[dict] = None
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        data = None
    if (
        isinstance(data, dict)
        and data.get("schema_version") == PINS_SCHEMA_VERSION
        and isinstance(data.get("components"), dict)
    ):
        value = data
    _pins_memo.update(key = key, value = value)
    return value


def pinned_entry(component: str) -> Optional[dict[str, Any]]:
    """The ``{repo, release_tag, ...}`` pinned for ``component``, or None."""
    pins = load_pins()
    if pins is None:
        return None
    entry = pins["components"].get(component)
    return entry if isinstance(entry, dict) else None


def pinned_tag(component: str, *, published_repo: Optional[str] = None) -> Optional[str]:
    """The pinned release tag for ``component``, or None when no pin is in force.

    None when the user opted out via ``ALLOW_LATEST_ENV``, when the manifest has
    no usable entry, or when ``published_repo`` names a publisher other than the
    one the pin does -- the pinned tag exists only in the pinned repo, so
    applying it to a custom ``--published-repo`` would compare against a tag that
    repo has never published. Mirrors ``prebuilt_core.pinned_release_tag``, which
    returns "" for the same three cases.
    """
    if allow_latest():
        return None
    entry = pinned_entry(component)
    if entry is None:
        return None
    repo = entry.get("repo")
    tag = entry.get("release_tag")
    if not isinstance(repo, str) or not repo.strip():
        return None
    if not isinstance(tag, str) or not tag.strip():
        return None
    wanted = (published_repo or "").strip()
    if wanted and wanted.lower() != repo.strip().lower():
        return None
    return tag.strip()


def install_target_tag(
    component: str, *, env_var: str, published_repo: Optional[str] = None
) -> Optional[str]:
    """The release tag the installer would install here when given none.

    Precedence is exactly ``prebuilt_core.default_published_release_tag``'s --
    the explicit env override (``UNSLOTH_LLAMA_RELEASE_TAG`` /
    ``UNSLOTH_WHISPER_RELEASE_TAG``) > the in-tree pin > None -- because that IS
    what the apply half will do. A freshness check comparing against anything
    else offers an update the installer would not perform, which is the bug this
    exists to close; an override is honoured for the same reason the pin is.

    None means "no pin in force": resolve GitHub's newest, as before the pin.
    """
    override = (os.environ.get(env_var) or "").strip()
    if override:
        return override
    return pinned_tag(component, published_repo = published_repo)


def reset_cache() -> None:
    """Drop the parsed-manifest memo (test seam)."""
    _pins_memo.clear()
