#!/bin/bash
configs=(
    "configs/AutoEncoder/conf1.yaml"
    "configs/AutoEncoder/conf2.yaml"
    "configs/ConvLSTM/conf1.yaml"
    "configs/ConvLSTM/conf2.yaml"
    "configs/LSTM/conf1.yaml"
    "configs/LSTM/conf2.yaml"
    "configs/MLP/conf1.yaml"
    "configs/MLP/conf2.yaml"
    "configs/MLR-RF/conf1.yaml"
    "configs/MLR-RF/conf2.yaml"
    "configs/RCAN/conf1.yaml"
    "configs/RCAN/conf2.yaml"
    "configs/XGB/conf1.yaml"
    "configs/XGB/conf2.yaml"
)

for config in "${configs[@]}"; do
    echo "Submitting GPU job for $config"
    sbatch scripts/job_gpu.sh "$config"
done
