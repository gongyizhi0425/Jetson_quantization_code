# KV Cache 量化与管理实验 — Jetson Orin NX

在 **NVIDIA Jetson Orin NX 16GB** 边缘设备上，针对医疗问答场景（PubMedQA / 合成 EHR）的大语言模型 KV Cache 优化实验框架。

项目系统对比了四种 KV Cache 策略在统一内存（Unified Memory）架构下的延迟、显存占用与生成质量（PPL）表现。

---

## 📋 目录

- [项目简介](#-项目简介)
- [硬件与环境要求](#-硬件与环境要求)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [实验流程](#-实验流程)
- [四种技术方案](#-四种技术方案)
- [核心评测指标](#-核心评测指标)
- [部署指南](#-部署指南)
- [引用与参考](#-引用与参考)

---

## 🚀 项目简介

在边缘端部署大语言模型时，**KV Cache 是长文本推理的显存瓶颈**。本项目在 Jetson Orin NX（16GB LPDDR5 统一内存，~10–12 GB 可用）上，围绕 **Qwen2.5-1.5B-Instruct**（GQA 架构）系统评测了以下方案：

| 方案 | 核心机制 | 目标 |
|------|---------|------|
| **Baseline** | HuggingFace 默认 `DynamicCache`（FP16 连续张量） | 基准对照 |
| **PagedAttention** | vLLM 引擎，Block-based 显存池 | 消除内存碎片，提升吞吐 |
| **KIVI** | 2-bit 非对称量化 + Residual Window | 极致压缩 KV Cache |
| **Paged + KIVI** | Block 管理 + 旧 Block 量化 | 零碎片 + 低 footprint |

**数据集**：
- [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)（医学研究问答）
- 合成 EHR（电子健康记录）时间线 + 问答对，支持**选择性保留（Selective Retention）**策略

---

## 🖥️ 硬件与环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | NVIDIA Jetson Orin NX 16 GB |
| 系统 | JetPack 6.x (L4T R36.x, Ubuntu 22.04 aarch64) |
| CUDA | 12.x |
| Python | 3.10+ |
| PyTorch | NVIDIA Jetson 专用 wheel（非 PyPI 版） |

> **统一内存提示**：Jetson 的 CPU 与 GPU 共享 16 GB LPDDR5。系统 + Jupyter 占用约 4–6 GB，实验可用约 **10–12 GB**。

---

## 📁 项目结构

```
.
├── configs/
│   └── experiment_config.yaml          # 实验统一配置（模型、内存预算、各阶段参数）
├── notebooks/                          # 实验管线（按序号执行）
│   ├── 00_env_check.ipynb              # 环境检测 + 内存预算
│   ├── 01_data_prep.ipynb              # 数据准备（PubMedQA / EHR）
│   ├── 02_baseline_gqa.ipynb           # GQA 基线 + OOM 探测
│   ├── 03_gqa_paged.ipynb              # PagedAttention (vLLM)
│   ├── 04_gqa_kivi.ipynb               # KIVI 2-bit 量化（LLaMA）
│   └── 04_gqa_kivi_qwen.ipynb          # KIVI 量化（Qwen）
├── preprocessing/
│   └── Final Project/                  # 合成 EHR 预处理管线输出
├── results/                            # 实验结果（CSV / JSON / 图表）
├── scripts/                            # 部署与工具脚本
│   ├── deploy_to_jetson.sh             # 一键部署到 Jetson
│   ├── setup_vllm_jetson.sh            # vLLM Jetson 安装
│   ├── setup_kivi_jetson.sh            # KIVI CUDA 后端编译
│   └── plot_results_comparison.py      # 结果可视化
├── src/                                # 核心源码
│   ├── ehr_bridge.py                   # EHR Prompt 组装（选择性保留）
│   ├── dataset_utils.py                # 数据集加载（PubMedQA / EHR）
│   ├── metrics.py                      # TTFT / TPOT / 内存评测引擎
│   ├── jetson_utils.py                 # Jetson 环境检测与安全加载
│   ├── vllm_runner.py                  # vLLM PagedAttention 封装
│   ├── kivi_cache.py                   # 纯 Python KIVI 2-bit Cache
│   ├── kivi_wrapper.py                 # KIVI 后端包装（CUDA / Python 回退）
│   ├── paged_cache.py                  # 简化版 Paged KV Cache
│   ├── paged_kivi_cache.py             # Paged + KIVI 组合方案
│   ├── perplexity.py                   # PPL（困惑度）质量评估
│   └── qwen_kivi_2.py                  # 现代版 KIVI Attention 注入（Qwen/LLaMA）
├── llama_kivi.py                       # 完整 LLaMA 模型 KIVI 重写
├── mistral_kivi.py                     # Mistral 模型 KIVI 重写
├── utils_quant.py                      # 通用量化基础设施
└── requirements_jetson.txt             # Jetson 依赖
```

---

## ⚡ 快速开始

### 1. 克隆与安装

在 Jetson 上执行：

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements_jetson.txt

# 注册 Jupyter 内核（如需要）
pip install jupyterlab ipykernel
python3 -m ipykernel install --user --name kv_cache --display-name "KV Cache (venv)"
```

> **注意**：PyTorch 必须使用 NVIDIA 提供的 Jetson 专用 wheel，详见 [DEPLOYMENT.md](DEPLOYMENT.md)。

### 2. 启动实验

按顺序执行 `notebooks/` 下的 notebook（**不要同时打开多个内核**）：

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

---

## 🔬 实验流程

| 序号 | Notebook | 内容 |
|------|----------|------|
| 00 | `00_env_check.ipynb` | 检测 JetPack / CUDA / 内存状态，确认预算 |
| 01 | `01_data_prep.ipynb` | 加载数据集，生成并保存 prompts (`results/ehr_prompts.json`) |
| 02 | `02_baseline_gqa.ipynb` | **Baseline**：HF `DynamicCache` + OOM 阈值探测 |
| 03 | `03_gqa_paged.ipynb` | **PagedAttention**：vLLM 引擎评测 |
| 04 | `04_gqa_kivi.ipynb` | **KIVI**：LLaMA 架构 2-bit 量化评测 |
| 04 | `04_gqa_kivi_qwen.ipynb` | **KIVI**：Qwen 架构 2-bit 量化评测 |

每个 notebook 运行结束后会自动释放模型内存，确保下一阶段的显存安全。

---

## 🧪 四种技术方案

### 1. Baseline — HF DynamicCache

- **机制**：使用 HuggingFace 默认的 `DynamicCache`，KV Cache 以 FP16 连续张量存储。
- **问题**：长文本生成时，PyTorch 需频繁申请更大的连续内存块并拷贝旧数据，导致 **大量显存碎片** 和 OOM 风险。
- **适用**：短问答（output < 20 tokens）。

### 2. PagedAttention — vLLM

- **机制**：vLLM 在底层用 C++ 重写显存管理，将 KV Cache 划分为固定大小的 **Blocks（Pages）**，启动时一次性预分配静态显存池。
- **收益**：
  - ✅ 显存碎片从 ~200 MB 降至 ~25 MB（动态压力极限压缩）
  - ✅ Decode 阶段 **TPOT ↓ 33%**，显存占用 **↓ 74%**
- **代价**：
  - ⚠️ Prefill 阶段 **TTFT ↑ 144%**（初始化与调度开销）
- **适用**：长文本生成（output > 100 tokens）、多请求并发。

> **注意**：vLLM 的调度器与 Block Manager 属于底层私有组件，本项目通过反射安全提取其内部统计。

### 3. KIVI — 2-bit 非对称量化

- **机制**：基于 [KIVI 论文](https://github.com/jy-yuan/KIVI)，在 Python/CUDA 层拦截 `past_key_values`：
  - **Key Cache**：Per-channel 量化（沿 seq_len 分组，统计沿 head_dim）
  - **Value Cache**：Per-token 量化（沿 head_dim 分组，统计沿 seq_len）
  - **Residual Window**：最近 `residual_length`（默认 128）tokens 保持 FP16，确保生成质量
- **实现**：
  - 优先加载编译好的 `kivi_gemv` CUDA kernel（Triton 量化 + fused unpack）
  - 若不可用，自动回退到 `src/kivi_cache.py` 的纯 Python 实现
- **收益**：理论 KV Cache 压缩约 **8×**，且 **Prefill 阶段无额外开销**（量化仅在 residual 溢出时触发）

### 4. Paged + KIVI — 组合方案

- **机制**：`src/paged_kivi_cache.py` 将 PagedAttention 的 block-based 分配与 KIVI 量化结合。
- **逻辑**：最新 `residual_blocks` 保持 FP16，更早的完整 block 被量化至 2-bit。
- **目标**：同时获得 **零碎片内存管理** 与 **极低显存占用**。

---

## 📊 核心评测指标

| 指标 | 含义 | 测量方式 |
|------|------|----------|
| **TTFT** | Time To First Token（首 token 延迟） | `time.perf_counter()` 在 prefill 前后同步计时 |
| **TPOT** | Time Per Output Token（每 token 生成时间） | decode 阶段逐 step 计时取平均 |
| **Peak Memory** | 峰值显存占用 | `torch.cuda.max_memory_allocated()` / `psutil` RSS |
| **KV Cache Memory** | KV Cache 实际占用 | 自定义 Cache 暴露 `memory_usage_bytes()` 或从 vLLM Block Manager 反推 |
| **Fragmentation** | 内存碎片率 | 1 - allocated/reserved；vLLM 下使用空闲 block 占比 |
| **PPL** | Perplexity（困惑度） | 滑动窗口计算，监控量化后的质量退化（>5% 视为显著退化） |
| **OOM Threshold** | 最大安全上下文长度 | 递增长度探测，安全跳过而非崩溃 |

---

## 🚀 部署指南

详见 [DEPLOYMENT.md](DEPLOYMENT.md)，包含：

- 一键部署脚本 (`scripts/deploy_to_jetson.sh`)
- vLLM 安装（含 Docker / jetson-containers 方式）
- KIVI CUDA 后端编译（`TORCH_CUDA_ARCH_LIST="8.7"`）
- JupyterLab 远程访问配置
- 故障排查（OOM、CUDA 不可用、编译错误等）

### vLLM Docker 快速启动（可选）

```bash
cd jetson-containers
sudo jetson-containers run -v /home/gyz/KV_cache_experiment:/workspace -p 8889:8888 $(autotag vllm)
```

容器内：
```bash
cd /workspace
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

---

## 📚 引用与参考

- **KIVI**: Liu et al., "KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache", 2024. ([GitHub](https://github.com/jy-yuan/KIVI))
- **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. ([Docs](https://docs.vllm.ai/))
- **PubMedQA**: Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering", EMNLP 2019.
- **Jetson Containers**: [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)

---

> **维护提示**：`README_real.md` 为原始开发笔记，保留供内部参考。本项目文档以 `README.md` 为准。
