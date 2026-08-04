"""Run a command with a portable wall-clock limit.

This keeps paid provider smoke tests bounded on both macOS and Linux without
requiring GNU ``timeout``. The child runs in its own process group so its
workers receive the same shutdown signal.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from typing import Any


def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_with_timeout(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    shutdown_grace_seconds: int,
) -> int:
    process = subprocess.Popen(list(command), start_new_session=True)
    previous_handlers: dict[signal.Signals, Any] = {}

    def _forward_signal(signum: int, _frame: object) -> None:
        _signal_process_group(process, signal.Signals(signum))

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _forward_signal)

    try:
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"Wall-clock limit of {timeout_seconds} seconds reached; " "requesting a clean shutdown.",
                file=sys.stderr,
            )
            _signal_process_group(process, signal.SIGINT)
            try:
                process.wait(timeout=shutdown_grace_seconds)
            except subprocess.TimeoutExpired:
                print(
                    f"Shutdown exceeded {shutdown_grace_seconds} seconds; " "terminating the process group.",
                    file=sys.stderr,
                )
                _signal_process_group(process, signal.SIGKILL)
                process.wait()
            return 124
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--shutdown-grace-seconds", type=int, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.timeout_seconds <= 0 or args.shutdown_grace_seconds <= 0:
        parser.error("timeout values must be positive")

    return run_with_timeout(
        command,
        timeout_seconds=args.timeout_seconds,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
