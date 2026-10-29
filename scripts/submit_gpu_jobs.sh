#!/bin/bash
configs=(
    "configs/train/autoencoder/conf1.yaml"
    "configs/train/autoencoder/conf2.yaml"
    "configs/train/convlstm/conf1.yaml"
    "configs/train/convlstm/conf2.yaml"
    "configs/train/lstm/conf1.yaml"
    "configs/train/lstm/conf2.yaml"
    "configs/train/mlp/conf1.yaml"
    "configs/train/mlp/conf2.yaml"
    "configs/train/mlr_rf/conf1.yaml"
    "configs/train/mlr_rf/conf2.yaml"
    "configs/train/rcan/conf1.yaml"
    "configs/train/rcan/conf2.yaml"
    "configs/train/xgb/conf1.yaml"
    "configs/train/xgb/conf2.yaml"
)

for config in "${configs[@]}"; do
    echo "Submitting GPU job for $config"
    sbatch scripts/job_gpu.sh "$config"
done
