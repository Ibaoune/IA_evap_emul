# Evaporation Downscaling and Estimation (ds_evap_intern)

This project provides a machine learning and deep learning framework to model and downscale evaporation data over Morocco, leveraging LMDZ climate model predictors (temperature, pressure, wind, etc.) and remote sensing variables.

## Project Structure

- `src/core/`: Contains core configuration, training loops (`PointWiseTrain` and `SpatialTrain`), and utility functions.
- `src/data/`: Modules for data loading, regridding, and pointwise/spatial preprocessing.
- `src/models/`: Implementation of machine learning and deep learning architectures:
  - Point-wise models: `LSTM`, `MLP`, `MLR_RF`, `XGB`
  - Spatial DL models: `AutoEncoder`, `ConvLSTM`, `RCAN`
- `configs/`: `configs/train/` holds the training configs (`common.yaml` with data paths and shared settings, plus one folder of hyperparameter files per architecture); `configs/plots/` holds the figure styles.
- `scripts/`: Shell scripts for batch job submission (e.g., `job_cpu.sh`, `job_gpu.sh`, `submit_gpu_jobs.sh`).
- `train.py`: Main entry point script for training all models.
- `postprocess.py`: Entry point for evaluation and intercomparison of trained models (helpers in `postprocessing/postproc/`).

## Usage

### Training

To train a model, run the `train.py` script from the root directory of the project and pass the path to the corresponding configuration file. 

Example commands:

```bash
# Train XGBoost (Point-wise)
python train.py --config configs/train/xgb/conf0.yaml

# Train Long Short-Term Memory network (LSTM) (Point-wise)
python train.py --config configs/train/lstm/conf2.yaml

# Train Spatial Deep Learning Models
python train.py --config configs/train/autoencoder/conf0.yaml
python train.py --config configs/train/convlstm/conf0.yaml
python train.py --config configs/train/rcan/conf0.yaml
```

### Dry-run Mode
You can quickly validate the data pipeline and model definitions using the `--dry-run` flag. This limits training to 1 epoch and truncates the dataset to a single year.
```bash
python train.py --config configs/train/autoencoder/conf0.yaml --dry-run
```

The training script will:
1. Load the configuration and set up output directories.
2. Load and preprocess the spatial datasets (e.g., apply the Morocco mask, handle missing values, and temporally align variables).
3. Automatically route execution to either `PointWiseTrain` or `SpatialTrain` based on the specified model type.
4. Train the specified model using the parameters defined in the YAML configuration.
5. Save the model predictions, training metrics, and a copy of the configuration to the results directory.

### Command-line help

Both entry points print all their arguments as a table, followed by examples:
```bash
python train.py --help
python postprocess.py --help
python postprocess.py --list-figures   # every figure name and group
```

### Postprocessing

`postprocess.py` evaluates trained configs in two stages:
1. **Compute**: reads `results/<model_type>/<conf>/<model_type>_preds.pkl`, computes per-pixel metrics (annual and seasonal) and caches them with the obs/pred test series in `postprocessing/results/single/<model_type>/<conf>/cache/`. The cache is refreshed automatically when the predictions pickle changes.
2. **Plot**: draws the selected figures from the cache, so changing a figure does not require recomputing anything.

```bash
# Single model: all main + extreme figures
python postprocess.py --config configs/train/xgb/conf0.yaml

# Redraw from the cache only: a group (main, extreme) or specific figures
python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures main
python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures spatial_metrics mean_et_map

# Recompute the cache from the predictions pickle
python postprocess.py --config configs/train/xgb/conf0.yaml --recompute

# Intercomparison (reuses each config's cache)
python postprocess.py --configs configs/train/xgb/conf0.yaml configs/train/lstm/conf0.yaml
python postprocess.py --model xgb
python postprocess.py --all --plot-only --figures intercomp
```

Single-model figures go to `postprocessing/results/single/<model_type>/<conf>/{main,extreme}/`, and intercomparison figures go to `postprocessing/results/intercomparison/<run_label>/`.

### Figure style

Everything about how figures look is set in `configs/plots/default.yaml`: output format and dpi, font sizes, map frame, colors, and for each mapped variable its colorbar. A colorbar can use distinct classes or a continuous gradient, automatic or fixed bounds (`[edges]` or `{vmin, vmax, step}`), a scale centered on 0, and arrows for out-of-range values. To change the style, write an override file that contains only the keys to change and pass it with `--plot-config`. `configs/plots/overrides/paper.yaml` is an example (PDF at 300 dpi, fixed scales for skill scores). Unknown keys are rejected, so a typo is reported instead of being ignored.

```bash
python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures spatial_metrics --plot-config configs/plots/overrides/paper.yaml
```

The style used for each figure, including the colorbar bounds computed automatically, is saved next to the figures in `plot_config_used.yaml`.

## Requirements

Ensure you have the necessary dependencies installed in your environment. Key packages include:
- `numpy`, `pandas`, `xarray`, `rioxarray`, `geopandas`, `shapely` (for data manipulation and geospatial processing)
- `scikit-learn`, `xgboost` (for machine learning models)
- `torch` (for deep learning components)
- `scipy`

## Data Configuration

The pipeline ingests a comprehensive set of predictors spanning climate model outputs and remote sensing data:
- **LMDZ250 (Coarse Grid Predictors)**: 
  - Evapotranspiration (`evap`)
  - Specific humidity (`q2m`)
  - Sea level pressure (`pres`)
  - Surface shortwave & longwave radiation (`swnet`, `lwnet`)
  - 2m air temperature (`t2m`)
  - Soil temperature (`tsol`)
  - Wind components (`u10m`, `v10m`)
  - Surface & total soil moisture (`mrsos`, `mrso`)
- **LMDZ35 (Fine Grid Predictors / Static)**:
  - Target Evapotranspiration (`evap`)
  - Leaf Area Index (`LAImean`)
  - Surface albedo (`albedo_glob`)
  - Topography/Elevation (`altitude`)

File paths for these datasets are securely mapped in `configs/train/common.yaml` and support nested folder structures seamlessly.

Configs follow the structure `configs/train/<model_type>/conf<X>.yaml (lowercase, e.g. `configs/train/mlr_rf/`)` so multiple experiments can easily be tracked and queued.

**Note on Experiments:**
Each model configuration folder (e.g., `configs/train/lstm/`) contains a `doc.txt` file. This document concisely explains the scientific and technical logic behind each prepared experiment (e.g., `conf0`, `conf1`, `conf2`), the hyperparameters changed, and the expected improvements (better generalization, enhanced temporal memory, etc.) without needing to read the YAML files directly.
