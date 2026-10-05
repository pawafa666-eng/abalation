# 3DGS Ablation Experiments

This repository provides training code and a runner for six 3D Gaussian Splatting (3DGS) ablation experiments. It contains the code, environment configuration, source files required to build the extensions, and metric files for the six experiments. Raw experimental data and trained models are not included.

## Experiment configurations

| Group | Point-cloud initialization | Masks | Gaussian positions fixed | Adaptive density control disabled |
| --- | --- | --- | --- | --- |
| A | Random | Yes | No | No |
| B | LiDAR point cloud | Yes | No | No |
| C | LiDAR point cloud | Yes | Yes | No |
| D | LiDAR point cloud | Yes | No | Yes |
| E | LiDAR point cloud | Yes | Yes | Yes |
| F | LiDAR point cloud | No | Yes | Yes |

## Environment and data

`environment.yml` lists the Conda dependencies. Source code for the two CUDA extensions is in `submodules/`. Training requires a CUDA-capable environment and a separately prepared dataset.

The dataset directory must contain at least:

```text
dataset/
├── images/
├── masks/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

The corresponding `.txt` camera files are also supported. LiDAR initialization uses `sparse/0/points3D.ply`; if it is absent, the code attempts to generate it from `points3D.bin`. The current runner requires a `masks/` directory even when running only group F, which does not use masks.

## Running the experiments

From the repository root, run:

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
python run_ablation_experiments.py --source /path/to/dataset
```

By default, the runner executes groups A–F in sequence, trains each group for 30,000 iterations, and saves models at iterations 7,000 and 30,000. To run selected groups, for example:

```bash
python run_ablation_experiments.py \
  --source /path/to/dataset \
  --experiments B C E \
  --iterations 30000 \
  --output-root output/my_ablation
```

`--source` specifies the dataset root. Relative paths are resolved from the repository root. `--output-root` specifies the results directory. The runner will not overwrite an individual experiment directory that already contains files. Run `python run_ablation_experiments.py --help` for more options.

## Outputs

By default, new runs are written to `output/ablation_runs_<timestamp>/`. Each experiment directory contains trained models and a `summary_metrics.csv` file; the run directory contains `ablation_summary.csv`. The repository's `output/` directory also provides `loss_results.txt`, `psnr_results.txt`, `ssim_results.txt`, `lpips_results.txt`, and `summary_metrics.csv` for groups A–F, together with the aggregate `ablation_summary.csv`.

The pretrained weights required for the LPIPS metric are downloaded by the code when first used.

## Rebuilding metric tables from result files

The repository's `output/` directory contains `*_results.txt` files for groups A–F. From the repository root, run:

```bash
python rebuild_ablation_metrics.py
```

The script reads each group's result files and regenerates its `summary_metrics.csv` and `output/ablation_summary.csv`. For each metric, `final_*` is taken from the last record, and `mean_*` is the arithmetic mean of all records. Use `--output-root` to specify another directory containing the result files.

## Licenses

This repository includes the original license files for Gaussian Splatting, the two CUDA extensions, and GLM. Please follow their respective terms when using or redistributing the code.