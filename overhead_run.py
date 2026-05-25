import argparse
import warnings
import torch
import torch.nn as nn

import models.PiXTime as PiXTime
import models.PiXTime_No_abs_token as PiXTime_No_abs_token

warnings.filterwarnings("ignore", message=".*Profiler clears events.*")


def format_flops(flops):
    """Format FLOPs count to human-readable string."""
    if flops >= 1e12:
        return f"{flops / 1e12:.4f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.4f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.4f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def format_bytes(num_bytes):
    """Format bytes to human-readable string."""
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.4f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MB"
    else:
        return f"{num_bytes / 1024:.2f} KB"


def count_params(module):
    """Return the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def format_params(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f} K"
    else:
        return str(n)


def build_model(args):
    if args.model == 'PiXTime_No_abs_token':
        model_cls = PiXTime_No_abs_token.Model
    else:
        model_cls = PiXTime.Model
    return model_cls(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        n_vars=args.n_vars,
        d_model=args.d_model,
        dropout=args.dropout,
        factor=args.factor,
        n_heads=args.n_heads,
        en_d_ff=args.en_d_ff,
        de_d_ff=args.de_d_ff,
        en_layers=args.en_layers,
        de_layers=args.de_layers,
    )


def generate_toy_data(args, device):
    """Generate random toy data matching the shapes used in run.py."""
    batch_x = torch.randn(args.batch_size, args.seq_len, args.n_vars, device=device)
    batch_y = torch.randn(args.batch_size, args.label_len + args.pred_len, args.n_vars, device=device)
    # time features: 4 dims for hour-level frequency
    batch_x_mark = torch.randn(args.batch_size, args.seq_len, 4, device=device)
    batch_y_mark = torch.randn(args.batch_size, args.label_len + args.pred_len, 4, device=device)
    # decoder input (same logic as run.py)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
    return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp


def measure_forward_flops(model, batch_x, batch_x_mark, dec_inp, batch_y_mark):
    """Measure forward pass FLOPs using torch.profiler."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    forward_flops = sum(e.flops for e in prof.events() if e.flops is not None)
    return forward_flops, outputs


def measure_backward_flops(loss):
    """Measure backward pass FLOPs using torch.profiler."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        loss.backward()
    backward_flops = sum(e.flops for e in prof.events() if e.flops is not None)
    return backward_flops


def measure_max_memory(model, optimizer, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y, args):
    """Measure peak GPU memory during one full training step."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if args.features == 'MS' else 0
    outputs = outputs[:, -args.pred_len:, f_dim:]
    target = batch_y[:, -args.pred_len:, f_dim:]
    loss = nn.MSELoss()(outputs, target)
    loss.backward()
    optimizer.step()

    max_memory = torch.cuda.max_memory_allocated(device=batch_x.device)
    return max_memory


def run():
    parser = argparse.ArgumentParser(description='PiXTime Overhead Measurement (FLOPs + Memory)')

    # Model selection
    parser.add_argument('--model', type=str, default='PiXTime',
                        choices=['PiXTime', 'PiXTime_No_abs_token'])

    # Toy data shape
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=720)
    parser.add_argument('--pred_len', type=int, default=120)
    parser.add_argument('--label_len', type=int, default=1)
    parser.add_argument('--n_vars', type=int, default=36, help='number of variables (enc_in)')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'MS'])

    # Model structure
    parser.add_argument('--patch_len', type=int, default=24)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--en_d_ff', type=int, default=2048)
    parser.add_argument('--de_d_ff', type=int, default=2048)
    parser.add_argument('--en_layers', type=int, default=1)
    parser.add_argument('--de_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, default=2, help='GPU device id (-1 for CPU)')

    args = parser.parse_args()

    if args.seq_len % args.patch_len != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be divisible by patch_len ({args.patch_len})")

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')

    print("=" * 60)
    print(f"  {args.model} Overhead Measurement")
    print(f"  seq_len={args.seq_len}, pred_len={args.pred_len}, patch_len={args.patch_len}")
    print(f"  n_vars={args.n_vars}, features={args.features}, batch_size={args.batch_size}")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}")
    print(f"  en_layers={args.en_layers}, de_layers={args.de_layers}")
    print(f"  en_d_ff={args.en_d_ff}, de_d_ff={args.de_d_ff}")
    print("=" * 60)

    # --- Build model ---
    model = build_model(args).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Parameter counts ---
    total_params = count_params(model)
    excluded_params = (count_params(model.head)
                       + count_params(model.patch_embedding)
                       + count_params(model.ex_embedding.enc_embedding))
    remaining_params = total_params - excluded_params

    # --- Generate toy data ---
    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = generate_toy_data(args, device)
    print(f"\nInput shapes:")
    print(f"  x_enc:      {list(batch_x.shape)}")
    print(f"  x_mark_enc: {list(batch_x_mark.shape)}")
    print(f"  dec_inp:    {list(dec_inp.shape)}")
    print(f"  y_mark_dec: {list(batch_y_mark.shape)}")

    print(f"\nParameter counts:")
    print(f"  Total                     : {format_params(total_params)}")
    print(f"  Excl head/patch/enc_emb   : {format_params(remaining_params)}")

    # --- Forward FLOPs ---
    forward_flops, outputs = measure_forward_flops(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print(f"\n  Forward FLOPs   : {format_flops(forward_flops)}")

    # --- Backward FLOPs ---
    f_dim = -1 if args.features == 'MS' else 0
    outputs_sliced = outputs[:, -args.pred_len:, f_dim:]
    target = batch_y[:, -args.pred_len:, f_dim:]
    loss = nn.MSELoss()(outputs_sliced, target)
    backward_flops = measure_backward_flops(loss)
    print(f"  Backward FLOPs  : {format_flops(backward_flops)}")

    # --- Max Memory (fresh optimizer state, new run) ---
    # reset model gradients and optimizer
    optimizer.zero_grad()
    model.zero_grad()
    max_memory = measure_max_memory(model, optimizer, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y, args)
    print(f"  Max Memory      : {format_bytes(max_memory)}")

    # --- Summary ---
    total_flops = forward_flops + backward_flops
    print(f"\n  Total FLOPs     : {format_flops(total_flops)}")
    print(f"  Ratio (bwd/fwd) : {backward_flops / forward_flops:.2f}x" if forward_flops > 0 else "")
    print("=" * 60)


if __name__ == '__main__':
    run()
