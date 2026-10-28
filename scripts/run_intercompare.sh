#!/bin/bash
# run_intercompare.sh
# Runs the intercomparison post-processing pipeline for specific configurations.

echo "========================================================="
echo " Starting Intercomparison for conf1 (All models)"
echo "========================================================="

# Get all conf1 config files
conf1_files=$(ls configs/*/conf1.yaml)

/home/mohammad.elaabaribao/.conda/envs/clean_env_Pytorch/bin/python postprocess.py --configs $conf1_files

echo "========================================================="
echo " Starting Intercomparison for conf2 (All models)"
echo "========================================================="

# Get all conf2 config files
conf2_files=$(ls configs/*/conf2.yaml)

/home/mohammad.elaabaribao/.conda/envs/clean_env_Pytorch/bin/python postprocess.py --configs $conf2_files

echo "========================================================="
echo " Starting Intercomparison for conf_test_available_data (All models)"
echo "========================================================="

# Get all conf_test_available_data config files
test_files=$(ls configs/*/conf_test_available_data.yaml)

/home/mohammad.elaabaribao/.conda/envs/clean_env_Pytorch/bin/python postprocess.py --configs $test_files

echo "========================================================="
echo " Intercomparison completed!"
echo "========================================================="
