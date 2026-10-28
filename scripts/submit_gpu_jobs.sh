#!/bin/bash
configs=(
    "configs/autoencoder/conf1.yaml"
    "configs/autoencoder/conf2.yaml"
    "configs/convlstm/conf1.yaml"
    "configs/convlstm/conf2.yaml"
    "configs/lstm/conf1.yaml"
    "configs/lstm/conf2.yaml"
    "configs/mlp/conf1.yaml"
    "configs/mlp/conf2.yaml"
    "configs/mlr_rf/conf1.yaml"
    "configs/mlr_rf/conf2.yaml"
    "configs/rcan/conf1.yaml"
    "configs/rcan/conf2.yaml"
    "configs/xgb/conf1.yaml"
    "configs/xgb/conf2.yaml"
)

for config in "${configs[@]}"; do
    echo "Submitting GPU job for $config"
    sbatch scripts/job_gpu.sh "$config"
done
