"""Command line: ``a2ui-model-tools <download|merge|convert|package>``."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from . import identities as pins
from .workdir import WorkDir

log = logging.getLogger("a2ui_model_tools")
DEFAULT_WORK = Path(os.environ.get("A2UI_MODEL_WORK", "models/webllm-e2b"))


def _download(args: argparse.Namespace) -> None:
    from .download import download_all

    download_all(WorkDir(args.work))


def _merge(args: argparse.Namespace) -> None:
    from .merge import merge

    work = WorkDir(args.work)
    merge(
        args.base_dir or work.base,
        args.adapter_dir or work.adapter,
        args.output_dir or work.merged,
        expected_adapter_sha256=args.expected_adapter_sha256,
        expected_modules=args.expected_modules,
        expected_rank=args.expected_rank,
        expected_scale=args.expected_scale,
    )


def _convert(args: argparse.Namespace) -> None:
    from .convert import convert

    work = WorkDir(args.work)
    convert(
        args.base_dir or work.base,
        args.merged_dir or work.merged,
        args.producer_dir or work.producer,
        args.output_dir or work.model,
        expected_merged_sha256=args.expected_merged_sha256,
    )


def _package(args: argparse.Namespace) -> None:
    from .package import package

    work = WorkDir(args.work)
    package(
        args.model_dir or work.model,
        args.assets_dir or work.assets,
        args.wasm or work.wasm,
        args.wasm_provenance or work.wasm_provenance,
        adapter_sha256=args.expected_adapter_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a2ui-model-tools", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging and tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help, description=help)
        sub.set_defaults(handler=handler)
        return sub

    def work_option(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--work",
            type=Path,
            default=DEFAULT_WORK,
            help="work directory (default: $A2UI_MODEL_WORK or %(default)s); needs ~30 GB",
        )

    sub = command("download", _download, "download and verify every pinned input")
    work_option(sub)

    sub = command("merge", _merge, "merge the LoRA adapter into the BF16 base")
    work_option(sub)
    sub.add_argument("--base-dir", type=Path)
    sub.add_argument("--adapter-dir", type=Path, help="mlx_lm or PEFT (Unsloth) adapter directory")
    sub.add_argument("--output-dir", type=Path)
    sub.add_argument("--expected-adapter-sha256", default=pins.ADAPTER_SHA256)
    sub.add_argument("--expected-modules", type=int, default=pins.ADAPTER_MODULES)
    sub.add_argument("--expected-rank", type=int, default=pins.ADAPTER_RANK)
    sub.add_argument("--expected-scale", type=float, default=pins.ADAPTER_SCALE)

    sub = command("convert", _convert, "patch the official MLC q4f16_1 weights with the merge")
    work_option(sub)
    sub.add_argument("--base-dir", type=Path)
    sub.add_argument("--merged-dir", type=Path)
    sub.add_argument("--producer-dir", type=Path)
    sub.add_argument("--output-dir", type=Path)
    sub.add_argument("--expected-merged-sha256", default=pins.MERGED_WEIGHTS_SHA256)

    sub = command("package", _package, "add runtime, tokenizer, licenses and the app manifest")
    work_option(sub)
    sub.add_argument("--model-dir", type=Path)
    sub.add_argument("--assets-dir", type=Path)
    sub.add_argument("--wasm", type=Path)
    sub.add_argument("--wasm-provenance", type=Path)
    sub.add_argument("--expected-adapter-sha256", default=pins.ADAPTER_SHA256)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    try:
        args.handler(args)
    except (ImportError, OSError, ValueError) as error:
        if args.verbose:
            raise
        log.error("%s", error)
        return 1
    return 0
