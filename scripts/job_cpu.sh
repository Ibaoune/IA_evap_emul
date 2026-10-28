#!/bin/bash
#SBATCH --job-name=evap_cpu_test
#SBATCH --output=logs/evap_cpu_%j.out
#SBATCH --error=logs/evap_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Load environment
source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh
conda activate clean_env_Pytorch

# Go to project root
cd /srv/data/mohammad.elaabaribao/work/interns/y2026/evap/ds_evap_intern

# Ensure results and logs directories exist
mkdir -p results logs

echo "Starting CPU test for EVAP models"
echo "Date: $(date)"

# Get config from arguments or default to XGB conf0
CONFIG="${1:-configs/xgb/conf0.yaml}"
echo "[INFO] Running CPU training with config: $CONFIG"

# Run the training script
python train.py --config "$CONFIG"

echo "Test completed at: $(date)"
