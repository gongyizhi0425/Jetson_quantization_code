# 提纲+CLI

## 📖 目录
- [项目简介](#-项目简介)
- [主要特性](#-主要特性)
- [安装指南](#-安装指南)
- [使用说明](#-使用说明)
- [贡献指南](#-贡献指南)

## 🚀 快速进入

> **在jetson上启动jupyter命令:**
> ```bash
> jupyter lab list # 查看所有jupyter lab进程
> sudo systemctl start jupyter # 启动jupyter lab
> ```

**Jupyter Lab 访问链接:**
- 实验结果目录: [http://10.30.54.15:8888/lab/tree/KV_cache_experiment/results](http://10.30.54.15:8888/lab/tree/KV_cache_experiment/results)
- 根目录: [http://10.30.54.15:8888](http://10.30.54.15:8888)




## 🧩 Part 1: EHR Bridge (Prompt 组装)
`ehr_bridge.py` 作为核心接口，使用生成的 timelines 和 qa jsonl 来组装 prompt。
**工作流程：**
  1. 根据定义好的 type weights 在 score_event函数中计算， 让后在selective retain 中按照topk保留事件，再按照时间重排
  2. render context函数开始渲染模块， build chat promot产生对话外壳， 这里要使用tokenizer
  load ehr qa函数把两个文件（timeline和qa）的字段做了映射拼接，直接赋值进返回的 dict
  get selective promopt
        
        from src.ehr_bridge import get_selective_prompts
        prompts = get_selective_prompts(max_samples=50)
        产生了prompts：纯chatML格式+Metadata用于评测（ground truth）
        
        example: prompt"<|im_start|>system You are an experienced medical expert reviewing a patient's longitudinal electronic health                 record (EHR). Provide detailed, evidence-based reasoning before stating your conclusion.<|im_end|> <|im_start|>user                   The following is a selectively retained subset of the patient's medical timeline. Low-value routine events have been                     removed to focus on clinically significant records. Timeline (selective retention, ratio=50%): 2018-01-21 |                         diagnosis | Diagnosed with type 2 diabetes after evaluation. 2018-02-17 | medication_start | Started metformin for                     type 2 diabetes. 2018-03-20 | procedure | Completed HbA1c monitoring for monitoring. 2018-04-26 | abnormal_lab | Abnormal finding: HbA1c elevated. 2018-05-28 | normal_lab | diet and exercise counseling repeated 2018-05-31 | medication_change | Adjusted metformin after abnormal monitoring result. 2018-07-23 | adverse_event | Reported adverse event: hypoglycemia after treatment. 2018-08-05 | hospitalization | Hospitalized for severe hypoglycemia. 2018-08-10 | medication_stop | Stopped metformin after hypoglycemia. 2019-04-23 | normal_lab | diet and exercise counseling repeated 2019-11-22 | routine_followup | routine follow-up with stable symptoms 2020-01-12 | normal_lab | diet and exercise counseling repeated 2020-02-16 | routine_followup | routine follow-up with stable symptoms 2020-07-09 | normal_lab | diet and exercise counseling repeated 2020-12-30 | routine_followup | stable chronic condition noted 2021-01-04 | routine_followup | routine follow-up with stable symptoms 2021-01-22 | routine_followup | routine follow-up with stable symptoms 2021-05-18 | routine_followup | routine follow-up with stable symptoms 2021-09-17 | normal_lab | normal lab panel reviewed 2021-10-18 | routine_followup | stable chronic condition noted 2021-11-01 | routine_followup | stable chronic condition noted 2021-11-11 | normal_lab | diet and exercise counseling repeated 2021-12-08 | normal_lab | diet and exercise counseling repeated 2022-01-07 | routine_followup | stable chronic condition noted 2022-02-04 | normal_lab | vaccination status reviewed 2022-04-05 | normal_lab | administrative insurance form updated 2022-05-04 | normal_lab | preventive care counseling completed 2022-07-12 | routine_followup | routine follow-up with stable symptoms 2022-08-27 | routine_followup | stable chronic condition noted 2022-10-03 | normal_lab | preventive care counseling completed 2022-11-17 | routine_followup | routine follow-up with stable symptoms Question: What diagnosis was first recorded for this patient, and on what date? Based on the retained records, provide your answer with step-by-step medical reasoning.
                        <|im_end|> <|im_start|>assistant "
                        question"What diagnosis was first recorded for this patient, and on what date?"
                        reference_answer"type 2 diabetes on 2018-01-21"
                        final_decision"diagnosis_history"
                        pubid"P00001-DX"
                        context_strategy"selective"
                        num_events_original"62"
                        num_events_retained"31"
                        num_tokens778

        后续所有的notebook中都要： prompts = load_prompts('../results/ehr_prompts.json')







        # Part2 Baseline
        
        OOL detection:    
        最大安全长度: 2048

        Benchmark： 
        Loaded 50 prompts
        TTFT + TPOT
        KV cache memory
        Fragmentation
                

        
        PPL: 12%


        # Create a new docker container for vllm

        
        #在主机上：

        sudo docker ps -a #检查目前已有的镜像
        
        cd jetson-containers  #进入容器文件夹
        
        sudo jetson-containers run -v /home/gyz/KV_cache_experiment:/workspace -p 8889:8888 $(autotag vllm)

        #jetsoncontainers会自动寻找最合适的镜像，并设置好映射文件夹， 比如dustynv//vllm：0.8.6-r36.4-cu128-24.04

        #进入容器中了，先下载基础硬件
        
        pip install pandas datasets jupyterlab matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

        #切换到映射进来的你的实验代码目录
        
        cd /workspace

        #启动内层实验室 (注意这里必须加 --allow-root，因为是 root 用户)

        pkill -f jupyter

        jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

        #PC上连接容器，先进入这个网址

        http://10.30.54.15:8889

        #再输入token

        比如 http://ubuntu:8889/lab?token=ae6a36b549829dd62a0a305e9d6c213714b247ba6950800a



        容器环境：

        vLLM：0.8.6+cu128
        pytorch： 2.7.0+cu128
        torch CUDA ： 12.8
        nvcc ： 12.8.93
        Triton 3.3.0 原装

        #自定义源配置（环境变量）
        PIP_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp6/cu128
        TAR_INDEX_URL=https://apt.jetson-ai-lab.dev/jp6/cu128/24.04

        所有包针对 JetPack 6 + CUDA 12.8 + Orin 预编译
        避免源码编译的兼容性问题  尤其是 Triton 这种含 CUDA kernel 的包
        vllm==0.8.6+cu128 是社区版的 Jetson 移植版本（非官方 release
        Triton 是 NVIDIA 开发的 GPU kernel 编程语言 + 编译器，允许用 Python 编写高性能 CUDA-like 算子，自动优化内存访问和并行策略。






        # Part3 vLLM
                
        在底层用 C++ 配合 PagedAttention 重写了整套显存管理和生成逻辑。
        使用 vllm serve 或 LLM(model="...") API： 绕开了HF
        深入 vLLM 的 block_manager（区块管理器）去抓取真实的碎片情况，同时把理论推导的 KV Cache 大小塞给 kv_cache_memory_mb,（失败）
        命令来自  https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.CacheConfig.calculate_kv_scales
        当 vLLM 启动时（create_vllm_engine），它直接根据 gpu_memory_utilization=0.60 把能用的显存一次性全部划走，做成了一个巨大的“静态 KV Cache
        ！在 Baseline 中，KV Cache 必须在显存中是物理连续的。随着生成越来越长，PyTorch 不得不频繁地申请更大的连续内存块，把旧数据拷过去，释放旧块。这会导致PyTorch 的 Caching Allocator 里产生大量无法回收的碎片
        ！而一旦开了vllm， PyTorch 层面的 torch.cuda.memory_allocated() 根本不会因为 KV Cache 的增加而增长
        ！PyTorch 测算出来的动态显存波动（比如 25MB），其实根本不是完整的 KV Cache，而仅仅是模型在前向传播（Forward Pass）时产生的临时激活值（Activations）
        内存波动大大减小。
        它从根本上消灭了 Baseline 中因为动态连续分配造成的 200多MB 的显存碎片，把运行时的动态内存压力压到了极限的 25MB
                
        ✅ 收益（Decode 阶段）：
        • TPOT ↓ 33%  → 长文本生成更快
        • 显存占用 ↓ 74% → 可跑更长上下文
        

        ⚠️ 代价（Prefill 阶段）：
        • TTFT ↑ 144% → 首词延迟增加
        • 初始化开销 → 短请求不划算

        🎯 适用场景：
        • ✅ 长文本生成（output > 100 tokens）
        • ✅ 多请求并发（vLLM 的调度优势）
        • ❌ 短问答（output < 20 tokens）→ Baseline 更优

        #Peak memory的统计：
                process = psutil.Process(os.getpid())
                process_mem_bytes = process.memory_info().rss  # RSS: 常驻集大小
                peak_mem = process_mem_bytes / (1024 ** 2)     # 转换为 MB

        #KV cache 占用的计算，仅统计了被占用的块 用kv_cache_manager 调用出来

                        if scheduler:
                        #  kv_cache_manager
                        kv_manager = getattr(scheduler, "kv_cache_manager", None)
                        bm = getattr(scheduler, "block_manager", getattr(scheduler, "_block_manager", None))
                        
                        if kv_manager:
                        _tot = getattr(kv_manager, "num_gpu_blocks", 0)
                        num_total_gpu_blocks = int(_tot) if _tot is not None else num_total_gpu_blocks
                        num_free_blocks = int(getattr(kv_manager, "num_free_blocks", 0))


        PagedAttention 不改变PPL的数值

        #优化思路：
        #启用 FP8 KV Cache（如果 Jetson CUDA 支持）
                create_vllm_engine(cache_dtype="fp8_e5m2")
        #但是Ampere 架构不支持，理论上可以将 最大并发并发请求数 或者 最大上下文长度（Max Context Length）直接翻倍， 底层原理是每个 Block 需要的显存减半，
        于是可以有更多的blocks

        #增加 gpu_memory_utilization（在安全范围内）
                create_vllm_engine(gpu_memory_utilization=0.65)  # 默认 0.60

        问题：vllm的调度器（Scheduler）和区块管理器（Block Manager）属于底层私有组件，并不是对外公开的稳定 API


        # Part4 KIVI

                在同一个容器里：
                再开一个虚拟环境（已经自动完成）
                
                pip install triton

                cd ~ && git clone https://github.com/jy-yuan/KIVI.git

                cd ~/KIVI/quant
                
                TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS=4 pip install . --no-build-isolation
                
                #Orin 的 compute capability = 87
                #直接使用 venv 中的 torch/setuptools 
                
                pip install . --no-deps --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
                #安装 Python 主包 kivi

                python3 -c "import torch; import kivi_gemv; print('KIVI CUDA backend OK')"
                #验证安装

                
                
                KIVI 的工作原理：它是作为 Hugging Face transformers 库的一个插件存在的。它接管了原生的 past_key_values，
                在 Python 层面对 KV Cache 进行 2-bit/4-bit 压缩。
                使用 Hugging Face (transformers API）
                
                尝试加载编译C++后端： kivi_gemv 用来实现pack和fused unpack，也就是一个自定义的cuda kernel
                但是KIVI工作的源码只适配llama
                在这里我们通过标准的 PyTorch API 加载大模型，然后把 kivi_gemv 算子强行塞进了 Qwen 模型
                
                把 llama_kivi.py 里最核心的计算流（包括打包逻辑 和 cuda_bmm_fA_qB_outer  C++ 引擎调用）完完整整地翻译并移植到了 Qwen2 的身上

                
                核心1： forwar kivi方法   正常model forward时，需要拦截self attention层，进行动态量化，每次有新的k和v时，用CUDA kernel 算出scale 找最值
                核心2： 




                
                
        # result
        量化操作只在 "overflow" 时触发，Prefill 阶段无额外开销
        


        # Part5 Summary
        




        # Jtop

        1. PID 821115 | root | python3.12

        状态： 占用 1.2G 系统内存 (MEM)，以及高达 3.4G 的 GPU 内存 ([GPU MEM])，并消耗了约 19.5% 的 CPU。

        解析： 这是你当前系统中最核心的计算任务进程。结合你的开发环境，这非常有可能是你正在运行的 AI 推理脚本、模型部署或 CUDA 相关的后端服务（例如你正在部署的量化模型或 KIVI CUDA 后端）。分配给它的 3.4G [GPU MEM] 说明模型权重已经被加载到了 GPU 上下文中准备或正在进行张量计算。
        
        2. PID 3224 | gyz | Xorg

        状态： 占用 39.3M 系统内存，117M GPU 内存。

        解析： Xorg 是 Linux 系统底层的 X11 显示服务器。它负责管理图形界面、驱动显示器输出、处理窗口绘制的硬件加速。它占用 117M 的 GPU 内存是用来渲染你当前看到的整个桌面环境画面的。

        3. PID 3375 | gyz | gnome-shell

        状态： 占用 31.8M 系统内存，49.9M GPU 内存。

        解析： 这是 GNOME 桌面环境的核心用户界面。你看到的顶部任务栏、应用启动器、窗口的阴影和动画等，都是由 gnome-shell 渲染的。它配合底层的 Xorg 工作，利用 GPU 资源来保证桌面动画的流畅。

        4. PID 3617 | gyz | xdg-desktop-por

        状态： 占用 2.9M 系统内存，2.8M GPU 内存。

        解析： 全称是 xdg-desktop-portal。这是一个 D-Bus 服务，主要用于为沙盒化应用（比如 Flatpak 格式的软件）提供安全的系统资源访问接口（比如屏幕共享、打开文件选择器等）。它占用极小的资源，属于正常的系统后台服务。



        # KIVI的原理
        Residual Window 管理逻辑:
                # 每次 update 时检查是否需要量化溢出部分
        res_len = key_residual.shape[-2]  # 当前 FP16 residual 长度
        if res_len >= residual_length + group_size:  # 比如 128+32=160
        # 1. 取出最老的 (res_len - residual_length) 个 token
        k_part = k[:, :, :n_quant, :].contiguous()
        # 2. 按策略量化打包
        k_block = quantize_per_channel(k_part, bits=2, group_size=32)
        # 3. 追加到 quantized list，裁剪 residual
        self._key_quant[layer_idx].append(k_block)
        self._key_residual[layer_idx] = k[:, :, n_quant:, :]  # 只留最近 128 个



        
                
