#!/bin/bash

#SBATCH --job-name=evap_gpu
#SBATCH --output=logs/evap_gpu_%j.out
#SBATCH --error=logs/evap_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

# Load environment
source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh
conda activate clean_env_Pytorch

# Ensure Python logs are flushed immediately
export PYTHONUNBUFFERED=1

# Go to project root
cd /srv/data/mohammad.elaabaribao/work/interns/y2026/evap/ds_evap_intern
mkdir -p results logs

# Record start time
start_time=$(date +%s)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "======================================"

########################################
# GPU INFO
########################################
echo "[INFO] Allocated GPU(s):"
nvidia-smi

########################################
# CONFIG PATH
########################################
CONFIG="${1:-configs/AutoEncoder/conf0.yaml}"
echo "[INFO] Using config: $CONFIG"

########################################
# RUN
########################################
echo "[INFO] Running training..."
python -u train.py --config "$CONFIG"

########################################
# END
########################################
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
