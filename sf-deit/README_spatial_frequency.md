# Spatial–Frequency Dual-Branch DeiT

This repository includes a DeiT variant matching the supplied spatial–frequency
architecture. The spatial branch keeps the standard DeiT patch embedding and
positional encoding. The frequency branch performs a fixed orthonormal 2D DCT,
removes the DC coefficient, preserves coefficient signs with a signed `log1p`,
and projects non-overlapping frequency blocks into a fixed token grid.

Each complete frequency block is mapped to the DeiT embedding dimension by a
learnable projection. Multi-head cross-attention then uses spatial tokens as
queries and frequency tokens as keys and values. Its output is added back to
the spatial tokens before the standard DeiT encoder blocks and CLS-token head.

## Models

- `deit_tiny_patch16_224_dual`
- `deit_small_patch16_224_dual` (default)
- `deit_base_patch16_224_dual`
- `deit_tiny_patch16_224_concat`
- `deit_small_patch16_224_concat`
- `deit_base_patch16_224_concat`
- `deit_tiny_patch16_224_reverse`
- `deit_small_patch16_224_reverse`
- `deit_base_patch16_224_reverse`

Frequency-only ablations (no patch embedding or cross-attention):

- `deit_tiny_patch16_224_frequency`
- `deit_small_patch16_224_frequency`
- `deit_base_patch16_224_frequency`

`--freq-pool-size K` controls the DCT token grid. The default `K=7` produces
49 frequency tokens. The image size must be divisible by `K`. No coefficients
are averaged: for example, CIFAR with `K=8` projects each complete 4x4x3 DCT
block into one token. Increasing `K` makes the frequency blocks smaller but
raises cross-attention cost linearly in `K²`.

## ImageNet-1k layout

The data loader expects the standard ImageFolder directory layout:

```text
/path/to/imagenet/
  train/
    n01440764/
      *.JPEG
    ...
  val/
    n01440764/
      *.JPEG
    ...
```

## Training

Single-node, 4-GPU training of the small model on ImageNet-1k:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model deit_small_patch16_224_dual \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --batch-size 256 \
  --freq-pool-size 7 \
  --output_dir /path/to/output
```

The existing DeiT recipe supplies ImageNet normalization, RandAugment, random
erasing, Mixup/CutMix, label smoothing, AdamW, and the cosine learning-rate
schedule. `IMNET` fixes the classification head to 1,000 classes.

To train from a standard DeiT spatial checkpoint, pass it with `--finetune`.
The spatial patch embedding, positional embedding, encoder, normalization, and
classification head are loaded; the frequency tokenizer and cross-attention
modules start from their initialized weights.

### Local Apple Silicon training with Metal

On an Apple Silicon Mac, run a single MPS process. The following recipe loads
the standard DeiT-small checkpoint, fine-tunes the spatial backbone, and learns
the new frequency branch from its random initialization:

```bash
conda run -n base python main.py \
  --device mps \
  --model deit_small_patch16_224_dual \
  --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --unscale-lr \
  --warmup-epochs 5 \
  --freq-pool-size 7 \
  --num_workers 4 \
  --no-pin-mem \
  --no-model-ema \
  --output_dir /path/to/output
```

MPS training uses FP32 because the CUDA AMP scaler is not applicable. Use a
recent PyTorch 2.x release: the older PyTorch 1.13 MPS backend can terminate in
native optimizer or attention code at 224×224. Start with batch size 8 on a
32 GB machine. The local command is intentionally non-distributed because MPS
does not use the CUDA/NCCL distributed path in this repository.

The same configuration is available as a convenience script:

```bash
./train_mps_imagenet.sh /path/to/imagenet /path/to/output
```

The script defaults to a one-epoch pilot run. After checking loss, memory, and
checkpoint output, start the full run with `EPOCHS=100`; `BATCH_SIZE` can also
override the default of 8.

## Evaluation

```bash
python main.py --eval \
  --model deit_small_patch16_224_dual \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --resume /path/to/output/best_checkpoint.pth
```

## Frequency-only ablation

Train the frequency branch by itself with the same recipe and token-grid size:

```bash
python main.py \
  --model deit_small_patch16_224_frequency \
  --data-set IMNET \
  --data-path /path/to/imagenet \
  --freq-pool-size 7 \
  --output_dir /path/to/frequency-only-output
```

The model performs the same block-projected DCT tokenization as the dual model, then
prepends a CLS token and sends only those tokens through the DeiT encoder and
classification head. Comparing this run with the matching baseline and dual
run isolates whether the frequency representation is independently useful.

For CIFAR-100, keep the native 32x32 resolution. The training entry point
rejects 224x224 frequency-only CIFAR runs because upsampling before DCT
tokenization severely dilutes the useful spectrum:

```bash
python main.py \
  --model deit_small_patch16_224_frequency \
  --data-set CIFAR \
  --data-path /path/to/cifar100 \
  --input-size 32 \
  --freq-pool-size 8 \
  --epochs 300 \
  --batch-size 128 \
  --output_dir /path/to/cifar-frequency-only
```

The tokenizer preserves signed AC coefficients and applies normalization only
after projecting a complete frequency block into the embedding dimension.
Normalizing the three input channels first would remove absolute spectral
energy; averaging each block would discard the coefficient layout.

To put the same signed-DCT block tokenizer back into the spatial-frequency
model, change only the model name and use a separate output directory:

```bash
python main.py \
  --model deit_small_patch16_224_dual \
  --data-set CIFAR \
  --data-path /path/to/cifar100 \
  --input-size 32 \
  --freq-pool-size 8 \
  --epochs 350 \
  --batch-size 128 \
  --output_dir /path/to/cifar-spatial-frequency-block
```

This model computes ordinary spatial patch tokens in parallel with the 64
signed-DCT tokens. Spatial tokens query frequency tokens as keys and values;
the residual fused spatial sequence is then processed by the DeiT encoder.

## Alternative fusion directions

The concatenation variant keeps all tokens. On 32x32 CIFAR it sends one CLS,
four spatial, and 64 frequency tokens (69 total) through the DeiT encoder:

```bash
python main.py \
  --model deit_small_patch16_224_concat \
  --data-set CIFAR \
  --data-path /path/to/cifar100 \
  --input-size 32 \
  --freq-pool-size 8 \
  --epochs 350 \
  --batch-size 128 \
  --output_dir /path/to/cifar-spatial-frequency-concat
```

The reverse cross-attention variant treats frequency tokens as queries and
spatial tokens as keys and values. It therefore preserves all 64 enhanced
frequency tokens for the main DeiT encoder:

```bash
python main.py \
  --model deit_small_patch16_224_reverse \
  --data-set CIFAR \
  --data-path /path/to/cifar100 \
  --input-size 32 \
  --freq-pool-size 8 \
  --epochs 350 \
  --batch-size 128 \
  --output_dir /path/to/cifar-spatial-frequency-reverse
```
