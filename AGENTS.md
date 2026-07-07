# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python scripts for reproducing and analyzing grokking behavior on modular arithmetic tasks.

- `src/` contains executable experiment and analysis scripts. `src/train.py` defines the Transformer, dataset, training loop, checkpoints, and metric logging. Files such as `src/agop.py`, `src/pca.py`, and `src/spectral_entropy.py` run post-training analyses.
- `utils/` contains plotting scripts and shared plotting configuration. `utils/plot_config.json` controls figure size, colors, axes, and legends.
- Runtime outputs are generated under ignored directories such as `data/` and `results/`. Do not commit checkpoints, metrics, PDFs, NumPy arrays, or cache folders unless requested.

## Build, Test, and Development Commands

There is no package build step. Run scripts directly from the repository root.

- `python src/train.py --operation add` trains modular addition and writes `data/x+y/metric.csv` plus checkpoints.
- `python src/train.py --operation mul` trains multiplication; valid operations are `add`, `sub`, `mul`, and `div`.
- `python src/agop.py --operation add --steps 50000 90000` computes AGOP matrices for selected checkpoints.
- `python utils/trianing_plot.py --operation all` generates training-curve PDFs from existing metrics.

Install dependencies in your environment before running experiments, especially `torch`, `numpy`, and `matplotlib`.

## Coding Style & Naming Conventions

Use Python 3 scripts with 4-space indentation and descriptive names. Follow the existing pattern of small command-line scripts with `argparse`, constants near the top, and a guarded entry point:

```python
if __name__ == "__main__":
    main()
```

Use snake_case for functions, variables, file names, and output directories. Keep operation keys consistent: `add`, `sub`, `mul`, `div`; directory names map to `x+y`, `x-y`, `x_mul_y`, and `x_div_y`.

## Testing Guidelines

No automated test suite is currently tracked. For training or analysis changes, run the smallest relevant command and verify expected outputs exist, for example under `data/{operation}/...`. If adding tests, place them under `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries with optional scope-like prefixes, for example `Restructure: keep only src/ and utils/` and `Cleanup: Organize project documentation and scripts`. Keep commits focused and mention generated-output changes only when intentional.

Pull requests should include the purpose, commands run, affected operations, and important output paths. Include plots or screenshots when visual output under `results/` changes.

## Agent-Specific Instructions

Respect the ignore policy: source and utility scripts are the primary tracked artifacts. Avoid deleting or overwriting local `data/` and `results/` outputs unless the user explicitly asks.
