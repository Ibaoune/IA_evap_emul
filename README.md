# Evaporation Downscaling and Estimation (ds_evap_intern)

This project provides a machine learning and deep learning framework to model and downscale evaporation data over Morocco, leveraging LMDZ climate model predictors (temperature, pressure, wind, etc.) and remote sensing variables.

## Project Structure

- `src/core/`: Contains core configuration, training loops (`PointWiseTrain` and `SpatialTrain`), and utility functions.
- `src/data/`: Modules for data loading, regridding, and pointwise/spatial preprocessing.
- `src/models/`: Implementation of machine learning and deep learning architectures:
  - Point-wise models: `LSTM`, `MLP`, `MLR_RF`, `XGB`
  - Spatial DL models: `AutoEncoder`, `ConvLSTM`, `RCAN`
- `configs/`: YAML configuration files defining model hyperparameters and data paths for each architecture.
- `scripts/`: Shell scripts for batch job submission (e.g., job_cpu.sh).
- `train.py`: Main entry point script for training all models.

## Usage

To train a model, run the `train.py` script from the root directory of the project and pass the path to the corresponding configuration file. 

Example commands:

```bash
# Train XGBoost (Point-wise)
python train.py --config configs/XGB/conf0.yaml

# Train Long Short-Term Memory network (LSTM) (Point-wise)
python train.py --config configs/LSTM/conf2.yaml

# Train Spatial Deep Learning Models
python train.py --config configs/AutoEncoder/conf0.yaml
python train.py --config configs/ConvLSTM/conf0.yaml
python train.py --config configs/RCAN/conf0.yaml
```

### Dry-run Mode
You can quickly validate the data pipeline and model definitions using the `--dry-run` flag. This limits training to 1 epoch and truncates the dataset to a single year.
```bash
python train.py --config configs/AutoEncoder/conf0.yaml --dry-run
```

The training script will:
1. Load the configuration and set up output directories.
2. Load and preprocess the spatial datasets (e.g., apply the Morocco mask, handle missing values, and temporally align variables).
3. Automatically route execution to either `PointWiseTrain` or `SpatialTrain` based on the specified model type.
4. Train the specified model using the parameters defined in the YAML configuration.
5. Save the model predictions, training metrics, and a copy of the configuration to the results directory.

## Requirements

Ensure you have the necessary dependencies installed in your environment. Key packages include:
- `numpy`, `pandas`, `xarray`, `rioxarray`, `geopandas`, `shapely` (for data manipulation and geospatial processing)
- `scikit-learn`, `xgboost` (for machine learning models)
- `torch` (for deep learning components)
- `scipy`

## Data Configuration

Input data is expected to include:
- LMDZ climate model variables (temperature, pressure, wind components).
- Soil moisture.
- Albedo and Leaf Area Index (LAI).
- Raw DEM (elevation data).

File paths for these datasets are defined in the configuration files (e.g., `configs/common.yaml` or model-specific YAML files).
