# Run Local LLMs

*TODO: update project-wide documentation*

**vLLM commands to run LMs:**

```sh
vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.88 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --trust-remote-code \
    --port 8000
```

```sh
vllm serve /mnt/data/model-store/Qwen3.5-35B-A3B \
    --served-model-name Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.88 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --trust-remote-code \
    --port 8000
```

```sh
vllm serve /mnt/data/model-store/Qwen3.5-35B-A3B \
    --served-model-name Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.88 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --trust-remote-code \
    --port 8000
```

```sh
vllm serve /mnt/data/model-store/DeepSeek-R1-Distill-Llama-70B \
    --served-model-name DeepSeek/DeepSeek-R1-Distill-Llama-70B \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.88 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --trust-remote-code \
    --port 8000
```

**SGLang:**

*(Qwen3.5-122B-A10B-GPTQ-Int4)*

`python -m sglang.launch_server     --model-path /mnt/data/model-store/Qwen3.5-122B-A10B-GPTQ-Int4   --served-model-name Qwen/Qwen3.5-122B-A10B-GPTQ-Int4     --port 8000     --tp-size 2     --mem-fraction-static 0.8     --context-length 40960     --reasoning-parser qwen3     --quantization moe_wna16 --attention-backend triton`

*(Qwen3.5-35B-A3B)*

`python -m sglang.launch_server --model-path /mnt/data/model-store/Qwen3.5-35B-A3B  --served-model-name Qwen/Qwen3.5-35B-A3B  --port 8000  --tp-size 2  --mem-fraction-static 0.8  --context-length 40960  --reasoning-parser qwen3  --attention-backend triton`