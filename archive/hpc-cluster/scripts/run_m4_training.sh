#!/bin/bash
#SBATCH --job-name=ecg_recording_eval
#SBATCH --partition=gpu-teaching-2d
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/recording-eval-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model cnn1d --epochs 100 --class-weights --class-weight-mode sqrt --learning-rate 5e-5 --weight-decay 5e-4 --optimizer adamw --scheduler warmup_cosine --warmup-epochs 5 --augment --recording-eval --early-stopping --early-stopping-metric recording_val_loss --patience 15 --experiment-name recording_eval
apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model resnet --epochs 100 --class-weights --class-weight-mode sqrt --learning-rate 5e-5 --weight-decay 5e-4 --optimizer adamw --scheduler warmup_cosine --warmup-epochs 5 --augment --recording-eval --early-stopping --early-stopping-metric recording_val_loss --patience 15 --experiment-name recording_eval
