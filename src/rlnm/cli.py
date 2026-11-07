"""Command-line entrypoint."""

import argparse
import json
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible robust lesion network mapping")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run a JSON-configured analysis")
    run_parser.add_argument("config")
    run_parser.add_argument("--output", required=True)
    demo = commands.add_parser("demo", help="Write synthetic inputs and two example configs")
    demo.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare-upstream", help="Prepare the pinned upstream C2 sample")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .pipeline import run

            print(json.dumps(run(args.config, args.output), indent=2))
        elif args.command == "demo":
            from .examples import create_demo

            print(create_demo(args.output))
        else:
            from .examples import prepare_upstream

            print(prepare_upstream(args.source, args.output))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"rlnm: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
