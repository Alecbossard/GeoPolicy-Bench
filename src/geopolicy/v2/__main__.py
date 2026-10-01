import argparse
from .config import read_recipe


def main():
    parser = argparse.ArgumentParser(
        description="GeoPolicy V2: diagnostics, training, evaluation and local demo"
    )
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diagnose")
    sub.add_parser("preserve")
    sub.add_parser("augment")
    sub.add_parser("pilots")
    sub.add_parser("inspect-act")
    sub.add_parser("study")
    sub.add_parser("report")
    sub.add_parser("verify")
    sub.add_parser("status")
    d = sub.add_parser("demo")
    d.add_argument("--bundle", default="artifacts/v2/demo")
    d.add_argument("--out", default="artifacts/v2/demo_reproduced")
    t = sub.add_parser("train")
    choice = t.add_mutually_exclusive_group(required=True)
    choice.add_argument("--preset")
    choice.add_argument("--run-config")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--final", action="store_true")
    t.add_argument("--out", required=True)
    t.add_argument("--resume")
    g = sub.add_parser("gate")
    g.add_argument("--preset", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--first-seed", type=int, default=100200)
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--conditions", nargs="+", default=["nominal"])
    e.add_argument("--videos", type=int, default=0)
    e.add_argument("--reserved-test", action="store_true")
    e.add_argument("--object-id", type=int, choices=[0, 1])
    e.add_argument("--goal-id", type=int, choices=[0, 1])
    args = parser.parse_args()
    recipe = read_recipe(
        args.config
        or (
            "configs/v2/diagnostic_recipe.json"
            if args.command == "diagnose"
            else "configs/v2/recipe.json"
        )
    )
    if args.command == "diagnose":
        from .diagnostics import diagnose

        diagnose(recipe)
    elif args.command == "preserve":
        from .preservation import preserve_v1

        print(preserve_v1())
    elif args.command == "augment":
        from .augmentation import augment

        augment(recipe)
    elif args.command == "pilots":
        from .pipeline import run_pilots

        run_pilots(recipe)
    elif args.command == "inspect-act":
        from .diagnostics import inspect_act

        inspect_act(recipe)
    elif args.command == "study":
        from .study import run_study

        run_study(recipe)
    elif args.command == "status":
        import json
        from .study import status

        print(json.dumps(status(), indent=2))
    elif args.command == "report":
        from .reporting import report

        report(recipe)
    elif args.command == "verify":
        from .verification import verify

        verify(recipe)
    elif args.command == "demo":
        from .demo import replay_demo

        replay_demo(args.bundle, args.out)
    elif args.command in ["train", "gate"]:
        import json
        from pathlib import Path
        from .presets import run_config

        cfg = (
            json.loads(Path(args.run_config).read_text())
            if getattr(args, "run_config", None)
            else run_config(
                recipe,
                args.preset,
                getattr(args, "seed", 0),
                pilot=not getattr(args, "final", False),
            )
        )
        if args.command == "train":
            from .training import train

            print(train(recipe, cfg, args.out, args.resume))
        else:
            from .gates import gate

            print(gate(recipe, cfg))
    elif args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(
            recipe,
            args.checkpoint,
            args.out,
            args.first_seed,
            args.episodes,
            args.conditions,
            args.videos,
            object_id=args.object_id,
            goal_id=args.goal_id,
            frozen=args.reserved_test,
        )


if __name__ == "__main__":
    main()
