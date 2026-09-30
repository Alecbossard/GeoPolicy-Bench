import argparse
import json


def main():
    p = argparse.ArgumentParser(description="GeoPolicy Bench reproducible local commands")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("teacher-train")
    t.add_argument("--out", default="artifacts/teacher_pilot")
    t.add_argument("--bootstrap", default="artifacts/reference_pilot/bootstrap.npz")
    t.add_argument("--steps", type=int, default=2048)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume")
    c = sub.add_parser("collect")
    c.add_argument("--out", default="artifacts/data_smoke")
    c.add_argument("--episodes", type=int, default=1)
    c.add_argument("--first-seed", type=int, default=0)
    c.add_argument("--teacher")
    s = sub.add_parser("student-train")
    s.add_argument("--dataset", default="artifacts/dataset")
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["act", "mono", "fusion"], default="fusion")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--updates", type=int, default=3000)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--resume")
    s.add_argument("--limit", type=int)
    s.add_argument("--device", choices=["cpu", "cuda"])
    s.add_argument("--view-dropout", type=float, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--first-seed", type=int, default=100100)
    e.add_argument("--conditions", nargs="+", default=["nominal"])
    e.add_argument("--videos", type=int, default=0)
    e.add_argument("--device", choices=["cpu", "cuda"])
    e.add_argument("--execute-steps", type=int)
    e.add_argument("--onnx-path")
    e.add_argument("--voxel", type=float, default=0.004)
    e.add_argument("--object-id", type=int, choices=[0, 1])
    e.add_argument("--goal-id", type=int, choices=[0, 1])
    a = p.parse_args()
    if a.command == "teacher-train":
        from .teacher import train

        print(json.dumps(train(a.out, a.bootstrap, a.steps, a.seed, a.resume), indent=2))
    elif a.command == "collect":
        from .data import collect

        collect(a.out, a.episodes, a.first_seed, a.teacher)
    elif a.command == "student-train":
        from .training import train_student

        print(
            json.dumps(
                train_student(
                    a.dataset,
                    a.out,
                    a.mode,
                    a.seed,
                    a.updates,
                    a.batch_size,
                    a.resume,
                    a.limit,
                    a.device,
                    a.view_dropout,
                ),
                indent=2,
            )
        )
    elif a.command == "evaluate":
        from .evaluation import evaluate_student

        evaluate_student(
            a.checkpoint,
            a.out,
            a.episodes,
            a.first_seed,
            tuple(a.conditions),
            a.videos,
            object_id=a.object_id,
            goal_id=a.goal_id,
            device=a.device,
            execute_steps=a.execute_steps,
            onnx_path=a.onnx_path,
            voxel=a.voxel,
        )


if __name__ == "__main__":
    main()
