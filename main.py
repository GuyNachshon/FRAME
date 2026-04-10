"""FRAME — Fast Recurrent Action-Masked Egocentric World Model.

CLI entry point dispatching to subcommands.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRAME world model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  uv run python main.py demo --stub --profile
  uv run python main.py collect --frames 50000 --output data/vizdoom/raw/
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Demo / inference
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--stub", action="store_true")
    demo_parser.add_argument("--checkpoint", type=str, default=None)
    demo_parser.add_argument("--profile", action="store_true")
    demo_parser.add_argument("--headless", action="store_true")

    # Collect data
    collect_parser = subparsers.add_parser(
        "collect", help="Collect ViZDoom data"
    )
    collect_parser.add_argument("--frames", type=int, default=50000)
    collect_parser.add_argument("--output", type=str,
                                default="data/vizdoom/raw/")
    collect_parser.add_argument("--resolution", type=int, default=128)
    collect_parser.add_argument("--random_action_prob", type=float,
                                default=0.15)
    collect_parser.add_argument("--scenario", type=str, default="basic")

    args = parser.parse_args()

    if args.command == "demo":
        # Rebuild argv for inference/loop.py's own argparse
        loop_args = []
        if args.stub:
            loop_args.append("--stub")
        if args.checkpoint:
            loop_args.extend(["--checkpoint", args.checkpoint])
        if args.profile:
            loop_args.append("--profile")
        if args.headless:
            loop_args.append("--headless")
        sys.argv = ["inference/loop.py"] + loop_args

        from inference.loop import main as run_demo
        run_demo()

    elif args.command == "collect":
        from data.vizdoom.collect import collect_frames
        collect_frames(
            n_frames=args.frames,
            output_path=args.output,
            resolution=args.resolution,
            random_action_prob=args.random_action_prob,
            scenario=args.scenario,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
