#!/bin/bash
#SBATCH --job-name=ecg_xai_matrix
#SBATCH --partition=gpu-teaching-2h
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/xai-matrix-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u xai.py --model cnn1d --checkpoint outputs/models/cnn1d_recording_eval_fold0.pt --output-name cnn1d_recording_eval_attention --all-classes --attention-summary --notes-name attention_recording_eval_notes.csv
apptainer run --nv /home/bsa01/containers/python_container.sif python -u evaluate_9x9.py --experiment-name recording_eval --models cnn1d,resnet
