#!/usr/bin/env python3
"""Run vLLM's serving benchmark without initializing the engine CLI.

The top-level ``vllm`` command constructs the serve-engine parser even for a
CPU-only benchmark client.  In a container with no exposed GPU that makes
device detection fail before the HTTP benchmark starts.  Calling the official
benchmark parser and entry point directly keeps the client CPU-only.
"""

from vllm.benchmarks.serve import add_cli_args, main
from vllm.utils.argparse_utils import FlexibleArgumentParser


def run() -> None:
    parser = FlexibleArgumentParser(description="Benchmark online serving throughput")
    add_cli_args(parser)
    main(parser.parse_args())


if __name__ == "__main__":
    run()
