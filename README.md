# DL-BSA ECG Classification

This is our code for the DL-BSA ECG classification project. The dataset used is the China Physiological Signal Challenge 2018 dataset.

The main idea is simple:

```text
raw ECG files -> preprocessing -> .npy files -> dataset -> model training
```

The task is multi-label classification, so one ECG recording can have more than one label.

## Data

Put the raw data in this structure:

```text
data/raw/Training_WFDB/
data/raw/REFERENCE.csv
```

After preprocessing, the processed files are saved in:

```text
data/processed/
```

Each saved file looks like this:

```python
{
    "signals": (num_segments, 12, 1500),
    "labels": (num_segments, 9),
    "subject_id": "A0001"
}
```

The signal length is 1500 samples because the ECG is downsampled to 250 Hz and each segment is 6 seconds long.

## What Each File Does

`config.py`

Contains the main settings such as paths, segment length, model name, batch size, epochs, and number of folds.

`preprocessing.py`

Loads the WFDB ECG files, reads the labels, filters the signals, downsamples them, normalizes them, splits them into 6-second segments, and saves `.npy` files.

`dataset.py`

Loads the processed `.npy` files and returns samples for PyTorch.

`models.py`

Contains the two models:

```text
cnn1d  = 1D CNN + BiGRU + Attention
resnet = 1D ResNet
```

Both models take input with shape:

```python
(batch_size, 12, 1500)
```

and output:

```python
(batch_size, 9)
```

`training.py`

Trains the selected model, evaluates it, saves model weights, and saves the results.

`utils.py`

Contains helper functions for creating folders, getting subject IDs, and making train/test splits.

`main.py`

Runs the whole pipeline.

## Models

The default model is:

```text
cnn1d
```

This is a 1D CNN followed by a BiGRU and attention layer.

The second model is:

```text
resnet
```

This uses residual blocks instead of the recurrent layer.


The model output is raw logits. Sigmoid is applied later during evaluation, because the loss function is `BCEWithLogitsLoss`.

## Final Training Setup

The final experiments use this setup:

```text
epochs: 100
batch size: 32
loss: BCEWithLogitsLoss with sqrt class weights
optimizer: AdamW
learning rate: 5e-5
weight decay: 5e-4
scheduler: warmup cosine
warmup epochs: 5
evaluation: 10-fold Group K-Fold
final evaluation: recording-level evaluation
metrics: macro F1, micro F1, macro AUC, per-class F1
```

Group K-Fold is used because each ECG recording is split into several segments. Segments from the same ECG should not appear in both training and testing.

Recording-level evaluation is used for the final result. Segment probabilities from the same original ECG recording are averaged first, then the metrics are calculated.

The final run also uses simple training augmentation:

```text
amplitude scaling
small Gaussian noise
small time shift
```

## How To Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the default model:

```bash
python main.py
```

Choose a model manually:

```bash
python main.py --model cnn1d
python main.py --model resnet
```

Change the number of epochs from the command line:

```bash
python main.py --model cnn1d --epochs 100
python main.py --model resnet --epochs 100
```

If processed `.npy` files already exist, preprocessing is skipped and training starts directly.

To reproduce the final training setup from the command line:

```bash
python main.py --model cnn1d --epochs 100 --class-weights --class-weight-mode sqrt --learning-rate 5e-5 --weight-decay 5e-4 --optimizer adamw --scheduler warmup_cosine --warmup-epochs 5 --augment --recording-eval --early-stopping --early-stopping-metric recording_val_loss --patience 15 --experiment-name recording_eval

python main.py --model resnet --epochs 100 --class-weights --class-weight-mode sqrt --learning-rate 5e-5 --weight-decay 5e-4 --optimizer adamw --scheduler warmup_cosine --warmup-epochs 5 --augment --recording-eval --early-stopping --early-stopping-metric recording_val_loss --patience 15 --experiment-name recording_eval
```


## Outputs

Trained model weights are saved in:

```text
outputs/models/
```

Results are saved in:

```text
outputs/results/
```

For example, when training `cnn1d`, the result files are:

```text
cnn1d_recording_eval_results.json
cnn1d_recording_eval_summary.csv
```

The JSON file keeps the full fold and epoch history. The CSV file is a simpler summary that can be opened in Excel.

The 9x9 class matrices are saved as:

```text
outputs/results/cnn1d_recording_eval_9x9_class_matrix.csv
outputs/results/resnet_recording_eval_9x9_class_matrix.csv
outputs/plots/cnn1d_recording_eval_9x9_class_matrix.png
outputs/plots/resnet_recording_eval_9x9_class_matrix.png
```

The attention weight summary for the CNN-BiGRU-Attention model is saved in:

```text
outputs/xai/cnn1d_recording_eval_attention_all_classes_attention_values.csv
outputs/xai/cnn1d_recording_eval_attention_all_classes_attention_values.png
outputs/xai/cnn1d_recording_eval_attention_all_classes_heatmap.png
```

## Archive

The models were originally trained on a University Cluster (Slurm + Apptainer) ; see archive/hpc/cluster/ for those job scripts.
