
On the Hydra cluster, the same final training can be submitted with:

```bash
sbatch archive/scripts/run_m4_training.sh
```

After training, run the attention summary and 9x9 class matrix script:

```bash
sbatch archive/scripts/run_attention_and_9x9.sh
```