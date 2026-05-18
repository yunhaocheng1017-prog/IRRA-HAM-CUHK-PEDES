# AutoDL 使用说明：IRRA 加载 HAM 预训练并训练 CUHK-PEDES

本文档用于在 AutoDL 云服务器上运行 IRRA，并从本地 HAM ReID 预训练权重初始化模型。

当前方案只修改和运行 IRRA 项目；HAM 项目不参与运行时导入，只需要提供 `.pth` 权重文件。

## 1. 推荐目录结构

建议在 AutoDL 上按下面方式放置项目、数据集和权重：

```bash
/root/autodl-tmp/IRRA-HAM-CUHK-PEDES/IRRA-main
/root/autodl-tmp/datasets/CUHK-PEDES
/root/autodl-tmp/checkpoints/ham.pth
```

CUHK-PEDES 数据集目录需要包含：

```bash
CUHK-PEDES/
  imgs/
  reid_raw.json
```

注意：不要把 `.pth` 或 `.pt` 这类大权重文件上传到 GitHub。HAM 权重放在 AutoDL 本地磁盘即可，运行时通过 `--ham-pretrain` 或脚本里的 `HAM_PRETRAIN` 指定。

## 2. 配置运行环境

在 AutoDL 上创建独立 conda 环境：

```bash
conda create -n irra_ham python=3.8 -y
conda activate irra_ham
```

根据 AutoDL 镜像的 CUDA 版本安装 PyTorch。常见 CUDA 11.8 镜像可以使用：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装 IRRA 需要的其他依赖：

```bash
pip install easydict prettytable tensorboard ftfy regex tqdm pillow numpy scipy scikit-learn opencv-python
```

检查环境是否正常：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import easydict, prettytable, tensorboard, ftfy; print('deps ok')"
```

如果第一条输出里的 `torch.cuda.is_available()` 是 `True`，说明 CUDA 可用。

## 3. 首轮跑通测试

先跑 1 个 epoch，用来确认 CUDA、数据路径、HAM 权重加载、日志、checkpoint 保存和验证流程都正常：

```bash
cd /root/autodl-tmp/IRRA-HAM-CUHK-PEDES/IRRA-main

ROOT_DIR=/root/autodl-tmp/datasets \
HAM_PRETRAIN=/root/autodl-tmp/checkpoints/ham.pth \
OUTPUT_DIR=logs/cuhk_pedes_ham \
NUM_EPOCH=1 \
BATCH_SIZE=128 \
LR=5e-6 \
bash run_irra.sh
```

如果 HAM 权重加载成功，日志里应该出现类似内容，并且 `loaded` 数量大于 0：

```text
Loading HAM pretrained checkpoint from /root/autodl-tmp/checkpoints/ham.pth
HAM pretrained summary: loaded=..., skipped=..., missing=..., unexpected=...
```

如果第 1 个 epoch 完成后能看到 `Validation Results - Epoch: 1` 和 R1/R5/R10/mAP 等指标，说明训练和验证流程已经跑通。

## 4. 完整训练

首轮测试通过后，按 60 epoch 完整训练：

```bash
cd /root/autodl-tmp/IRRA-HAM-CUHK-PEDES/IRRA-main

ROOT_DIR=/root/autodl-tmp/datasets \
HAM_PRETRAIN=/root/autodl-tmp/checkpoints/ham.pth \
OUTPUT_DIR=logs/cuhk_pedes_ham \
NUM_EPOCH=60 \
BATCH_SIZE=128 \
LR=5e-6 \
bash run_irra.sh
```

训练输出会保存在：

```bash
logs/cuhk_pedes_ham/CUHK-PEDES/<时间戳>_irra_ham/
```

主要产物包括：

```bash
latest.pth
best.pth
configs.yaml
events.out.tfevents.*
log.txt
```

其中：

- `latest.pth`：每个 epoch 结束后保存，用于中断后续训。
- `best.pth`：验证集 Rank@1 最好时保存。
- `log.txt`：完整训练日志。
- `configs.yaml`：本次训练参数记录。

## 5. 中断后续训

如果 AutoDL 断开或训练中断，可以从 `latest.pth` 继续：

```bash
python train.py \
  --name irra_ham_resume \
  --root_dir /root/autodl-tmp/datasets \
  --output_dir logs/cuhk_pedes_ham \
  --dataset_name CUHK-PEDES \
  --loss_names 'sdm+mlm+id' \
  --MLM \
  --img_aug \
  --batch_size 128 \
  --lr 5e-6 \
  --num_epoch 60 \
  --resume \
  --resume_ckpt_file /path/to/latest.pth
```

使用 `--resume` 时，checkpoint 里已经包含模型、优化器和学习率调度器状态，会覆盖模型初始化状态。因此续训时一般不需要再传 `--ham-pretrain`。

## 6. 常见问题

- `HAM pretrained summary: loaded=0`：说明 HAM checkpoint 的 key 或模型结构和当前 IRRA 不匹配。先检查日志里 `HAM loaded` / `HAM skipped` 的 key 示例。
- CUHK-PEDES 找不到：`--root_dir` 应该指向包含 `CUHK-PEDES` 的父目录，例如 `/root/autodl-tmp/datasets`，不是 `/root/autodl-tmp/datasets/CUHK-PEDES`。
- 24GB 单卡显存不足：先把 `BATCH_SIZE=128` 改成 `BATCH_SIZE=64`；如果还 OOM，再改成 `BATCH_SIZE=32`。
- Windows PowerShell 不能运行 `npm` 或 `codex`：使用 `npm.cmd` 和 `codex.cmd`。
- GitHub 不要上传大权重文件：HAM 权重放在 AutoDL 本地，例如 `/root/autodl-tmp/checkpoints/ham.pth`。
