import argparse
from .resources import preflight


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--view", choices=["fixed", "fusion"], required=True)
    t.add_argument("--augmented", action="store_true")
    t.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stop-at", type=int)
    t.add_argument("--scope", choices=["pilot", "main"], default="pilot")
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--condition", required=True)
    e.add_argument("--first", type=int, default=210000)
    e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--split", choices=["validation", "test"], default="validation")
    e.add_argument("--videos", type=int, default=0)
    sub.add_parser("checks")
    sub.add_parser("freeze")
    sub.add_parser("preflight")
    args = vars(p.parse_args())
    command = args.pop("command")
    preflight(command)
    if command == "train":
        from .training import train

        train(**args)
    elif command == "eval":
        from .evaluation import evaluate

        evaluate(**args)
    elif command == "checks":
        from .checks import checks

        checks()
    elif command == "freeze":
        from .protocol import freeze

        freeze()


if __name__ == "__main__":
    main()
