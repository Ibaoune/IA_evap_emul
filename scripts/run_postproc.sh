#!/bin/bash
# run_postproc.sh
# Runs the modular postprocessing pipeline for all trained models.

echo "========================================================="
echo " Starting Post-Processing for all models"
echo "========================================================="

# Loop through all config files
for conf in configs/*/*.yaml; do
    echo "---------------------------------------------------------"
    echo " Processing: $conf"
    echo "---------------------------------------------------------"
    # Extract model and conf from the path (e.g., configs/AutoEncoder/conf1.yaml)
    model=$(basename $(dirname "$conf"))
    # lowercase the model name
    model=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    conf_name=$(basename "$conf" .yaml)
    
    # Check if the .pkl file exists
    preds_path="results/${model}/${conf_name}/${model}_preds.pkl"
    if [ ! -f "$preds_path" ]; then
        echo "  [SKIP] No predictions found at $preds_path"
        continue
    fi
    
    /home/mohammad.elaabaribao/.conda/envs/clean_env_Pytorch/bin/python postprocessing/postprocess_single.py --config "$conf"
done

echo "========================================================="
echo " Post-Processing completed for all models!"
echo "========================================================="
