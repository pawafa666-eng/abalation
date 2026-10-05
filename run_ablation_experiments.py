import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "id": "A",
        "name": "A_original_3dgs_mask",
        "description": "Original 3DGS-style random initialization + mask",
        "init": "random",
        "use_mask": True,
        "freeze_positions": False,
        "disable_densification": False,
    },
    {
        "id": "B",
        "name": "B_lidar_init_mask",
        "description": "LiDAR point cloud initialization + mask",
        "init": "lidar",
        "use_mask": True,
        "freeze_positions": False,
        "disable_densification": False,
    },
    {
        "id": "C",
        "name": "C_lidar_init_fixed_mask",
        "description": "LiDAR initialization + fixed Gaussian positions + mask",
        "init": "lidar",
        "use_mask": True,
        "freeze_positions": True,
        "disable_densification": False,
    },
    {
        "id": "D",
        "name": "D_lidar_init_no_densify_mask",
        "description": "LiDAR initialization + disabled adaptive density control + mask",
        "init": "lidar",
        "use_mask": True,
        "freeze_positions": False,
        "disable_densification": True,
    },
    {
        "id": "E",
        "name": "E_full_method_mask",
        "description": "LiDAR initialization + fixed positions + disabled adaptive density control + mask",
        "init": "lidar",
        "use_mask": True,
        "freeze_positions": True,
        "disable_densification": True,
    },
    {
        "id": "F",
        "name": "F_full_method_no_mask",
        "description": "Full method without mask",
        "init": "lidar",
        "use_mask": False,
        "freeze_positions": True,
        "disable_densification": True,
    },
]


SUMMARY_FIELDS = [
    "experiment",
    "description",
    "model_path",
    "status",
    "final_loss",
    "final_l1",
    "final_opacity_loss",
    "final_psnr",
    "final_ssim",
    "final_lpips",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the six 3DGS ablation experiments sequentially.")
    parser.add_argument("--source", default="data_file/Dida_tunnel_260114", help="Dataset root containing images, sparse/0, and masks.")
    parser.add_argument("--output-root", default="", help="Output directory. Defaults to output/ablation_runs_<timestamp>.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["A", "B", "C", "D", "E", "F"],
        help="Experiments to run by id or name, e.g. --experiments B C D E F.",
    )
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--save-iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--test-iterations", nargs="+", type=int, default=[150000])
    parser.add_argument("--random-points", type=int, default=100000)
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch train.py. Defaults to the current venv Python.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later experiments if one experiment fails.")
    return parser.parse_args()


def read_final_metrics(path, columns):
    values = {column: "" for column in columns}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        for idx, column in enumerate(columns, start=1):
            if idx >= len(parts):
                continue
            try:
                values[column] = float(parts[idx])
            except ValueError:
                pass
    return values


def summarize_metrics(exp, model_path, status):
    summary = {field: "" for field in SUMMARY_FIELDS}
    summary["experiment"] = exp["name"]
    summary["description"] = exp["description"]
    summary["model_path"] = str(model_path)
    summary["status"] = status

    metric_specs = [
        ("loss_results.txt", ["loss", "l1", "opacity_loss"]),
        ("psnr_results.txt", ["psnr"]),
        ("ssim_results.txt", ["ssim"]),
        ("lpips_results.txt", ["lpips"]),
    ]

    for filename, columns in metric_specs:
        final_values = read_final_metrics(model_path / filename, columns)
        for column, value in final_values.items():
            if value != "":
                summary[f"final_{column}"] = value

    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def select_experiments(requested):
    by_id = {exp["id"].upper(): exp for exp in EXPERIMENTS}
    by_name = {exp["name"].lower(): exp for exp in EXPERIMENTS}
    selected = []
    seen = set()

    for item in requested:
        key_id = item.upper()
        key_name = item.lower()
        exp = by_id.get(key_id) or by_name.get(key_name)
        if exp is None:
            valid = ", ".join(exp["id"] for exp in EXPERIMENTS)
            raise ValueError(f"Unknown experiment '{item}'. Valid ids: {valid}")
        if exp["id"] not in seen:
            selected.append(exp)
            seen.add(exp["id"])

    return selected


def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"


def build_train_command(args, root, source, masks, exp, model_path):
    command = [
        args.python,
        str(root / "train.py"),
        "-s",
        str(source),
        "-m",
        str(model_path),
        "--init_point_cloud",
        exp["init"],
        "--random_points",
        str(args.random_points),
        "--iterations",
        str(args.iterations),
        "--save_iterations",
        *[str(i) for i in args.save_iterations],
        "--test_iterations",
        *[str(i) for i in args.test_iterations],
    ]

    if exp["use_mask"]:
        command.extend(["--masks", str(masks)])
    if exp["freeze_positions"]:
        command.append("--freeze_gaussian_positions")
    if exp["disable_densification"]:
        command.append("--disable_densification")

    return command


def run_experiment(args, root, source, masks, output_root, exp, index, total):
    model_path = output_root / exp["name"]
    if model_path.exists() and any(model_path.iterdir()):
        raise FileExistsError(f"Refusing to mix runs in existing nonempty directory: {model_path}. Use a new --output-root.")
    model_path.mkdir(parents=True, exist_ok=True)
    command = build_train_command(args, root, source, masks, exp, model_path)

    print()
    print(f"{progress_bar(index, total)} {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Output: {model_path}")
    print("Command:", " ".join(f'"{part}"' if " " in part else part for part in command))
    print()

    return_code = subprocess.run(command, cwd=root, check=False).returncode

    status = "success" if return_code == 0 else f"failed:{return_code}"
    summary = summarize_metrics(exp, model_path, status)
    write_csv(model_path / "summary_metrics.csv", [summary])

    if return_code != 0:
        raise RuntimeError(f"Experiment {exp['name']} failed with exit code {return_code}")

    return summary


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    source = (root / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source).resolve()
    masks = source / "masks"

    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if not masks.exists():
        raise FileNotFoundError(f"Mask folder not found: {masks}")

    if args.output_root:
        output_root = Path(args.output_root)
        output_root = (root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = root / "output" / f"ablation_runs_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    print("3DGS ablation runner")
    print(f"Python executable: {args.python}")
    print(f"Dataset: {source}")
    print(f"Output root: {output_root}")
    experiments = select_experiments(args.experiments)
    print("Selected experiments:", ", ".join(exp["id"] for exp in experiments))

    summaries = []
    total = len(experiments)
    for idx, exp in enumerate(experiments, start=1):
        try:
            summary = run_experiment(args, root, source, masks, output_root, exp, idx, total)
            summaries.append(summary)
        except Exception as exc:
            summary = summarize_metrics(exp, output_root / exp["name"], "failed")
            summaries.append(summary)
            write_csv(output_root / "ablation_summary.csv", summaries)
            print(f"\nERROR: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

        write_csv(output_root / "ablation_summary.csv", summaries)

    print()
    print("All requested experiments finished.")
    print(f"Aggregate summary: {output_root / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
