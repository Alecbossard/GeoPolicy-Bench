import argparse


def main():
    parser = argparse.ArgumentParser(description="Isolated GeoPolicy-Bench V3")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preserve")
    p.add_argument("--create", action="store_true")
    sub.add_parser("checks")
    sub.add_parser("baseline")
    sub.add_parser("audit")
    p = sub.add_parser("collect")
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--model", choices=["direct_bc", "v1_diffusion"], default="direct_bc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--continued", action="store_true")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--binary", action="store_true")
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--name", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--first", type=int, default=110100)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--reference", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--videos", type=int, default=0)
    args = parser.parse_args()
    if args.command == "preserve":
        from .common import preservation
        preservation(args.create)
    elif args.command == "checks":
        from .checks import checks
        checks()
    elif args.command == "baseline":
        from .pipeline import baseline
        baseline()
    elif args.command == "audit":
        from .audit import audit
        audit()
    elif args.command == "collect":
        from .collection import collect
        collect(args.first, args.episodes)
    elif args.command == "train":
        from .training import train
        train(args.name, args.model, args.seed, args.updates, args.limit, args.continued,
              args.history, args.binary, args.overfit, args.resume)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.name, args.checkpoint, args.first, args.episodes,
                 args.reference, args.raw, args.videos)


if __name__ == "__main__":
    main()
