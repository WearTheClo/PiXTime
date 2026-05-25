# PiXTime

PiXTime is a federated time series forecasting framework supporting multiple transformer-based models for long-term multivariate time series forecasting.

## Quick Start

```bash
# Train PiXTime on ETTh1 with 96→96 prediction
python run.py --model PiXTime --data ETTh1 --seq_len 96 --pred_len 96 --features M

# Train with smaller model dimensions
python run.py --model PiXTime --data ETTh1 --seq_len 96 --pred_len 96 \
    --d_model 128 --en_d_ff 512 --de_d_ff 512 --en_layers 1 --de_layers 1
```

## Supported Models

| Model | Description |
|-------|-------------|
| **PiXTime** | Patch embedding + cross-attention decoder with ABS token |
| **PiXTime_No_abs_token** | PiXTime variant without ABS token (for ablation studies) |
| **DLinear** | Linear decomposition baseline |
| **iTransformer** | Inverted transformer — one token per variable |
| **PatchTST** | Patch-based time series transformer |
| **TimeXer** | Patch embedding with token fusion |

## Datasets

| Dataset | `--root_path` |
|---------|---------------|
| ETTh1, ETTh2, ETTm1, ETTm2 | `dataset/ETT-small/` (default) |
| electricity | `./dataset/electricity/` |
| exchange_rate | `./dataset/exchange_rate/` |
| traffic | `./dataset/traffic/` |
| weather | `./dataset/weather/` |

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | PiXTime | Model name |
| `--data` | ETTh1 | Dataset name |
| `--seq_len` | 96 | Input sequence length |
| `--pred_len` | 96 | Prediction length |
| `--patch_len` | 16 | Patch length (must divide seq_len) |
| `--features` | M | M (multivariate), MS (M→S), S (univariate) |
| `--d_model` | 512 | Hidden dimension |
| `--n_heads` | 8 | Attention heads |
| `--en_d_ff` | 2048 | Encoder feed-forward dim |
| `--de_d_ff` | 2048 | Decoder feed-forward dim |
| `--en_layers` | 2 | Encoder layers |
| `--de_layers` | 2 | Decoder layers |
| `--dropout` | 0.1 | Dropout rate |
| `--train_epochs` | 10 | Training epochs |
| `--learning_rate` | 0.0001 | Learning rate |
| `--batch_size` | 32 | Batch size |

## Batch Experiments

Shell scripts are provided for running full benchmark sweeps across all 8 datasets and 4 prediction lengths (96, 192, 336, 720):

```bash
bash pixtime_run.sh        # PiXTime
bash abl_run_for_abs.sh    # PiXTime without ABS token (ablation)
bash dlinear_run.sh        # DLinear
bash itrans_run.sh         # iTransformer
bash patchtst_run.sh       # PatchTST
bash timexer_run.sh        # TimeXer
```

## Overhead Measurement

Measure FLOPs, parameter counts, and peak GPU memory:

```bash
python overhead_run.py --model PiXTime
```

Options: `--model PiXTime` or `--model PiXTime_No_abs_token`.

## Experiment Results

Results are stored in `evaluation/` directory. The three xlsx files record model performance (MSE/MAE) averaged across 4 prediction lengths (96, 192, 336, 720).

## Project Structure

```
PiXTime/
├── run.py                    # Training entry point
├── abl_run_for_abs.py        # Ablation study to abs token entry point
├── overhead_run.py           # FLOPs & memory measurement tool
├── models/                   # Model implementations
├── layers/                   # Reusable transformer components
├── dataset/                  # Data loading and preprocessing
├── utils/                    # Metrics, time features, utilities
├── fl/                       # Federated learning modules
├── checkpoint/               # Saved model checkpoints
├── evaluation/               # Experiment results
└── *.sh                      # Batch experiment scripts
```

## Requirements

- Python 3.8+
- PyTorch 1.11+
- NumPy, pandas

## Important Notes

- `seq_len` must be evenly divisible by `patch_len` for patch-based models (PiXTime, PatchTST, TimeXer)
- `enc_in` (input channels) is auto-detected from the dataset
- PiXTime models automatically save checkpoints to `./checkpoint/` after training
- Results are written to `./evaluation/` directory
