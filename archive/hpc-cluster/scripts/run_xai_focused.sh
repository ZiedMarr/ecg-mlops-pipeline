#!/bin/bash
#SBATCH --job-name=ecg_xai_focus
#SBATCH --partition=gpu-teaching-2h
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/xai-focused-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u xai.py --model cnn1d --checkpoint outputs/models/cnn1d_recording_eval_fold0.pt --output-name cnn1d_recording_eval_focus --classes 1,4,6,9 --leads 2,7,12 --qrs --gradcam-only --notes-name xai_recording_eval_notes.csv
apptainer run --nv /home/bsa01/containers/python_container.sif python -u xai.py --model resnet --checkpoint outputs/models/resnet_recording_eval_fold0.pt --output-name resnet_recording_eval_focus --classes 1,4,6,9 --leads 2,7,12 --qrs --gradcam-only --notes-name xai_recording_eval_notes.csv
