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
    for name, help_text in (
        ("annotate", "Compare a NIfTI map to reference annotations using spatial nulls"),
        ("benchmark-eigenbrain", "Experimental nested MRI/PET prediction benchmark"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("config")
        sub.add_argument("--output", required=True)
    research_demo = commands.add_parser(
        "demo-research", help="Write synthetic annotation/MRI-PET data"
    )
    research_demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .pipeline import run

            print(json.dumps(run(args.config, args.output), indent=2))
        elif args.command == "demo":
            from .examples import create_demo

            print(create_demo(args.output))
        elif args.command == "prepare-upstream":
            from .examples import prepare_upstream

            print(prepare_upstream(args.source, args.output))
        elif args.command == "annotate":
            from .annotate import run_annotation

            print(json.dumps(run_annotation(args.config, args.output), indent=2))
        elif args.command == "benchmark-eigenbrain":
            from .benchmark import run_benchmark

            print(json.dumps(run_benchmark(args.config, args.output), indent=2))
        else:
            from .research_examples import create_research_demo

            print(create_research_demo(args.output))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"rlnm: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
