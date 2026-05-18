#!/bin/bash
DATASET_NAME="CUHK-PEDES"
ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/datasets}"
HAM_PRETRAIN="${HAM_PRETRAIN:-/root/autodl-tmp/checkpoints/ham.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/cuhk_pedes_ham}"
NUM_EPOCH="${NUM_EPOCH:-60}"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name irra_ham \
--root_dir "$ROOT_DIR" \
--output_dir "$OUTPUT_DIR" \
--ham_pretrain "$HAM_PRETRAIN" \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--lr 5e-6 \
--num_epoch "$NUM_EPOCH"
