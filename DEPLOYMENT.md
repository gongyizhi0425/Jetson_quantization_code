# Jetson Orin NX 部署指南

## 前提条件

| 项目 | 要求 |
|------|------|
| 硬件 | Jetson Orin NX 16 GB |
| 系统 | JetPack 6.x (L4T R36.x, Ubuntu 22.04 aarch64) |
| CUDA | 12.x (JetPack 自带) |
| PyTorch | NVIDIA Jetson 专用 wheel (非 PyPI 版) |

> **统一内存架构**: Jetson 的 CPU 和 GPU 共享 16 GB LPDDR5。系统 + Jupyter 占用 4-6 GB，实验可用约 **10-12 GB**。

---

## 一键部署（从开发机执行）

```bash
# 设置 Jetson 连接信息
export JETSON_USER=gyz
export JETSON_HOST=192.168.x.x    # 替换为 Jetson IP

# 执行部署脚本
bash scripts/deploy_to_jetson.sh
```

脚本会自动：
1. `rsync` 同步项目文件到 Jetson
2. 创建 Python 虚拟环境并安装依赖
3. 安装 JupyterLab 并注册内核

---

## 手动部署步骤

### 1. 安装 NVIDIA PyTorch Wheel

**必须使用 NVIDIA 提供的 Jetson 专用 PyTorch**（PyPI 版本没有 aarch64 CUDA 支持）：

```bash
# JetPack 6.x 对应的 PyTorch 安装
# 查看最新 wheel: https://forums.developer.nvidia.com/t/pytorch-for-jetson/
pip install torch-2.x.x+nvYYMMDD-cp310-cp310-linux_aarch64.whl
```

### 2. 复制项目到 Jetson

```bash
# 从开发机
scp -r /path/to/KV_cache_experiment gyz@jetson-ip:~/
```

### 3. 安装 Python 依赖

```bash
ssh gyz@jetson-ip
cd ~/KV_cache_experiment
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements_jetson.txt
```

### 4. 启动 JupyterLab

```bash
pip install jupyterlab ipykernel
python3 -m ipykernel install --user --name kv_cache --display-name "KV Cache (venv)"
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

在开发机浏览器打开: `http://<jetson-ip>:8888`

### 5. 运行实验 Notebooks

按以下顺序执行（每个 notebook 结束时自动释放模型内存）：

```
00_env_check.ipynb      → 环境检测 + 内存预算
01_data_prep.ipynb      → 下载 PubMedQA 数据集
02_baseline_gqa.ipynb   → GQA 基线 + OOM 探测
03_gqa_paged.ipynb      → PagedAttention (vLLM)
04_gqa_kivi.ipynb       → KIVI 2-bit 量化
05_all_combined.ipynb   → Paged + KIVI 组合
06_analysis.ipynb       → 结果对比分析
```

> **重要**: 不要同时打开多个实验 notebook 的内核。运行完一个再开下一个。

### 6. （可选）安装 vLLM / KIVI

```bash
# vLLM — PagedAttention 引擎
bash scripts/setup_vllm_jetson.sh

# KIVI — 2-bit 量化 CUDA 内核
bash scripts/setup_kivi_jetson.sh
```

---

## 内存监控

```bash
# jetson-stats (推荐)
sudo pip install jetson-stats
sudo jtop

# 或在 notebook 中使用
from src.jetson_utils import print_jetson_summary, get_memory_status_mb
print_jetson_summary()
```

---

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| `torch.cuda.is_available()` 返回 False | 使用 NVIDIA Jetson wheel 重装 PyTorch |
| 模型加载 OOM | 关闭其他 notebook 内核；用 `jtop` 检查内存 |
| vLLM 安装失败 | ARM64 可能需要源码编译，见 `setup_vllm_jetson.sh` |
| KIVI CUDA 编译错误 | 确认 `TORCH_CUDA_ARCH_LIST="8.7"` 已设置 |
| JupyterLab 无法远程访问 | 检查 `--ip=0.0.0.0` 参数或防火墙设置 |
