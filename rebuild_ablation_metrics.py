import argparse
import csv
from pathlib import Path

from run_ablation_experiments import EXPERIMENTS


FIELDS = [
    "experiment", "description", "status",
    "final_loss", "mean_loss", "final_l1", "mean_l1",
    "final_opacity_loss", "mean_opacity_loss",
    "final_psnr", "mean_psnr", "final_ssim", "mean_ssim",
    "final_lpips", "mean_lpips",
]

METRIC_FILES = {
    "loss_results.txt": ("loss", "l1", "opacity_loss"),
    "psnr_results.txt": ("psnr",),
    "ssim_results.txt": ("ssim",),
    "lpips_results.txt": ("lpips",),
}


def read_metrics(path, names):
    values = [[] for _ in names]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) < len(names) + 1:
            raise ValueError(f"Incomplete record in {path}:{line_number}")
        try:
            int(parts[0])
            numbers = [float(value) for value in parts[1:len(names) + 1]]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric record in {path}:{line_number}") from exc
        for column, number in zip(values, numbers):
            column.append(number)
    if not values[0]:
        raise ValueError(f"No metric records in {path}")
    return {name: (column[-1], sum(column) / len(column)) for name, column in zip(names, values)}


def summarize_experiment(experiment, output_root):
    directory = output_root / experiment["name"]
    row = {
        "experiment": experiment["name"],
        "description": experiment["description"],
        "status": "success",
    }
    for filename, names in METRIC_FILES.items():
        for name, (final, mean) in read_metrics(directory / filename, names).items():
            row[f"final_{name}"] = final
            row[f"mean_{name}"] = mean
    return row


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Rebuild ablation CSVs from *_results.txt files.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    rows = [summarize_experiment(experiment, output_root) for experiment in EXPERIMENTS]
    for row in rows:
        write_csv(output_root / row["experiment"] / "summary_metrics.csv", [row])
    write_csv(output_root / "ablation_summary.csv", rows)
    print(f"Rebuilt {len(rows)} experiment summaries and {output_root / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
