#!/bin/bash
DATASET_NAME="CUHK-PEDES"
ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/datasets}"
HAM_PRETRAIN="${HAM_PRETRAIN:-/root/autodl-tmp/checkpoints/ham.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/cuhk_pedes_ham}"
NUM_EPOCH="${NUM_EPOCH:-60}"
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra_ham \
--root_dir "$ROOT_DIR" \
--output_dir "$OUTPUT_DIR" \
--ham-pretrain "$HAM_PRETRAIN" \
--img_aug \
--batch_size "$BATCH_SIZE" \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--lr "$LR" \
--num_epoch "$NUM_EPOCH"
