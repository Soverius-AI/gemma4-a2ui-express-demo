"""Download every pinned input, verifying each file against its digest.

Standard library only. Existing files are verified and kept, never replaced, so
an interrupted download can simply be rerun.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import identities as pins
from .checkpoint_io import IdentityError, file_digest
from .workdir import WorkDir

log = logging.getLogger(__name__)

Verifier = Callable[[Path], None]
USER_AGENT = "a2ui-model-tools/1"
COPY_BLOCK_BYTES = 1024 * 1024


def digest_verifier(expected: str, *, algorithm: str = "sha256", size: int | None = None) -> Verifier:
    def verify(path: Path) -> None:
        if size is not None and path.stat().st_size != size:
            raise IdentityError(f"Size mismatch for {path}: expected {size} bytes")
        actual = file_digest(path, algorithm)
        if actual != expected:
            raise IdentityError(f"Identity mismatch for {path}: expected {expected}, got {actual}")

    return verify


def fetch(
    url: str,
    target: Path,
    verify: Verifier,
    *,
    attempts: int = 5,
    timeout: float = 180,
    backoff: float = 2.0,
) -> bool:
    """Download ``url`` to ``target`` unless it already exists. Return whether it downloaded.

    An existing file must pass ``verify``; it is never overwritten. New bytes land
    in a temporary file and are hard-linked into place only after verification.
    """
    if target.exists():
        try:
            verify(target)
        except IdentityError as error:
            raise IdentityError(f"Existing file does not match its pinned identity: {error}") from None
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, COPY_BLOCK_BYTES)
            verify(partial)
            os.link(partial, target)  # fails rather than replace a file written meanwhile
            partial.unlink()
            log.info("Downloaded %s", target)
            return True
        except FileExistsError:
            partial.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError, IdentityError) as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            log.warning("Attempt %d/%d for %s failed: %s", attempt, attempts, url, error)
            time.sleep(backoff**attempt)
    raise AssertionError("unreachable")


def download_plan(work: WorkDir) -> list[tuple[str, Path, Verifier]]:
    plan = [
        (pins.BASE.url(name), work.base / name, digest_verifier(sha)) for name, sha in pins.BASE_FILES.items()
    ]
    plan += [
        (pins.ADAPTER.url(f"{pins.ADAPTER_SUBDIR}/{name}"), work.adapter / name, digest_verifier(sha))
        for name, sha in pins.ADAPTER_FILES.items()
    ]
    plan += [
        (pins.MLC_BASE.url(name), work.producer / name, digest_verifier(sha))
        for name, sha in pins.MLC_METADATA_FILES.items()
    ]
    plan += [
        (pins.MLC_BASE.url(name), work.assets / name, digest_verifier(sha))
        for name, sha in pins.MLC_ASSET_FILES.items()
    ]
    plan += [
        (source.url, work.assets / name, digest_verifier(sha))
        for name, (source, sha) in pins.TVM_FILES.items()
    ]
    plan.append((pins.WASM.url, work.wasm, digest_verifier(pins.WASM_SHA256, size=pins.WASM_BYTES)))
    return plan


def wasm_provenance() -> dict[str, object]:
    return {
        "repository": f"https://github.com/{pins.WASM.repository}",
        "commit": pins.WASM.commit,
        "path": pins.WASM.path,
        "url": pins.WASM.url,
        "sha256": pins.WASM_SHA256,
        "bytes": pins.WASM_BYTES,
    }


def download_all(work: WorkDir) -> None:
    """Fetch the base, adapter, MLC metadata, licenses and WASM runtime into ``work``."""
    for url, target, verify in download_plan(work):
        if not fetch(url, target, verify):
            log.info("Verified existing %s", target)
    expected = json.dumps(wasm_provenance(), indent=2) + "\n"
    if work.wasm_provenance.exists():
        if work.wasm_provenance.read_text() != expected:
            raise IdentityError(f"Unexpected {work.wasm_provenance}")
    else:
        work.wasm_provenance.write_text(expected)
    log.info("All pinned inputs are in %s", work.root)
