# AutoDL HAM Pretrain Finetune Guide

This guide runs IRRA on CUHK-PEDES initialized from the HAM ReID pretrained
checkpoint. HAM code is not imported at runtime; only its `.pth` checkpoint is
loaded into the IRRA model.

## Paths

Use the following AutoDL layout:

```text
/root/autodl-tmp/IRRA
/root/autodl-tmp/datasets/CUHK-PEDES
/root/autodl-tmp/checkpoints/ham.pth
```

CUHK-PEDES must contain:

```text
CUHK-PEDES/
  imgs/
  reid_raw.json
```

## Smoke Run

Run one epoch first to verify CUDA, dataset paths, HAM checkpoint loading, logs,
and checkpoint saving.

```bash
cd /root/autodl-tmp/IRRA

CUDA_VISIBLE_DEVICES=0 python train.py \
  --name irra_ham_smoke \
  --root_dir /root/autodl-tmp/datasets \
  --output_dir logs/cuhk_pedes_ham \
  --ham_pretrain /root/autodl-tmp/checkpoints/ham.pth \
  --dataset_name CUHK-PEDES \
  --loss_names 'sdm+mlm+id' \
  --MLM \
  --img_aug \
  --batch_size 128 \
  --lr 5e-6 \
  --num_epoch 1
```

The log should include dataset statistics, `HAM pretrained summary`, epoch loss
messages, and `Saving checkpoint to ... latest.pth`.

## Full Training

HAM recommends fine-tuning IRRA with batch size 128, one GPU, 60 epochs, and
learning rate `5e-6`.

```bash
cd /root/autodl-tmp/IRRA

CUDA_VISIBLE_DEVICES=0 python train.py \
  --name irra_ham \
  --root_dir /root/autodl-tmp/datasets \
  --output_dir logs/cuhk_pedes_ham \
  --ham_pretrain /root/autodl-tmp/checkpoints/ham.pth \
  --dataset_name CUHK-PEDES \
  --loss_names 'sdm+mlm+id' \
  --MLM \
  --img_aug \
  --batch_size 128 \
  --lr 5e-6 \
  --num_epoch 60
```

You can run the bundled script with the same defaults:

```bash
cd /root/autodl-tmp/IRRA
bash run_irra.sh
```

For a one-epoch smoke run with the script:

```bash
NUM_EPOCH=1 bash run_irra.sh
```

## Resume

Each epoch writes `latest.pth`; the best validation Rank-1 checkpoint is saved
as `best.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --resume \
  --resume_ckpt_file logs/cuhk_pedes_ham/CUHK-PEDES/<run_name>/latest.pth \
  --name irra_ham_resume \
  --root_dir /root/autodl-tmp/datasets \
  --output_dir logs/cuhk_pedes_ham \
  --dataset_name CUHK-PEDES \
  --loss_names 'sdm+mlm+id' \
  --MLM \
  --img_aug \
  --batch_size 128 \
  --lr 5e-6 \
  --num_epoch 60
```

When resuming, the IRRA checkpoint already contains model weights, optimizer,
and scheduler state. You do not need to pass `--ham_pretrain` again.

## Troubleshooting

- If `HAM pretrained summary` shows `loaded=0`, the HAM checkpoint keys do not
  match this IRRA model. Check whether the checkpoint is the HAM ReID pretrain
  model rather than the LLaVA captioner checkpoint.
- If CUHK-PEDES is not found, check that `--root_dir` points to the parent
  directory containing `CUHK-PEDES`, not the dataset directory itself.
- If CUDA OOM happens on a 24GB card, retry with `--batch_size 64`. If needed,
  keep the run stable first and then tune batch size upward.
- If training is interrupted, resume from the generated `latest.pth`.
