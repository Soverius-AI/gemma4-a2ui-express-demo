"""Directory layout shared by the download, merge, convert and package steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkDir:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "base-hf"

    @property
    def adapter(self) -> Path:
        return self.root / "adapter"

    @property
    def producer(self) -> Path:
        """Official MLC metadata; ``base-mlc/`` caches its weight shards."""
        return self.root / "producer"

    @property
    def assets(self) -> Path:
        return self.producer / "assets"

    @property
    def wasm(self) -> Path:
        return self.producer / "model.wasm"

    @property
    def wasm_provenance(self) -> Path:
        return self.producer / "wasm-provenance.json"

    @property
    def merged(self) -> Path:
        return self.root / "merged-hf"

    @property
    def model(self) -> Path:
        return self.root / "mlc-model"
