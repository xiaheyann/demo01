import os
import time
import importlib
import ctypes
from statistics import mean

qpass = importlib.import_module("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass")

def _noop(*args, **kwargs):
    return None

# 尽量兼容不同实现
if hasattr(qpass, "QKNormRopeFusionPass"):
    cls = qpass.QKNormRopeFusionPass
    for meth in ("__call__", "apply", "run"):
        if hasattr(cls, meth):
            setattr(cls, meth, _noop)

for fn in (
    "register_qknorm_rope_fusion_pass",
    "apply_qknorm_rope_fusion_pass",
):
    if hasattr(qpass, fn):
        setattr(qpass, fn, _noop)

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="model-bin/xin7697/Qwen3-14B-W8A8-Ascend",
        tokenizer="model-bin/xin7697/Qwen3-14B-W8A8-Ascend", # 非量化版不需要
        quantization="ascend", # 非量化版不需要
        max_model_len=16384,
        block_size=128,
        gpu_memory_utilization=0.85,
        max_num_seqs=1,
        disable_log_stats=True,
        additional_config={
            "xlite_graph_config": {
                "enabled": False,
                # "full_mode": True,
            },
            "ascend_compilation_config": {
                "fuse_qknorm_rope": False,
            },
        },
    )

    prompt = "你好，尽可能详细的介绍一下你自己。"

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=256,
    )

    results = []

    for i in range(3):
        start_time = time.time()

        outputs = llm.generate([prompt], sampling_params)

        elapsed = time.time() - start_time
        output = outputs[0]

        generated_text = output.outputs[0].text

        prompt_token_ids = output.prompt_token_ids
        output_token_ids = output.outputs[0].token_ids

        prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        completion_tokens = len(output_token_ids) if output_token_ids is not None else 0
        total_tokens = prompt_tokens + completion_tokens

        tps = completion_tokens / elapsed if elapsed > 0 else 0.0
        tpm = tps * 60

        results.append(
            {
                "round": i + 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed": elapsed,
                "tps": tps,
                "tpm": tpm,
                "text": generated_text,
            }
        )

    for item in results:
        print(f"===== 第 {item['round']} 次请求 =====")
        print(f"耗时: {item['elapsed']:.4f} 秒")
        print(f"输入 token 数: {item['prompt_tokens']}")
        print(f"输出 token 数: {item['completion_tokens']}")
        print(f"总 token 数: {item['total_tokens']}")
        print(f"输出吞吐: {item['tps']:.2f} tokens/s")
        print(f"输出吞吐: {item['tpm']:.2f} tokens/min")
        print("生成结果:")
        print(item["text"])
        print()

    avg_prompt_tokens = mean(item["prompt_tokens"] for item in results)
    avg_completion_tokens = mean(item["completion_tokens"] for item in results)
    avg_total_tokens = mean(item["total_tokens"] for item in results)
    avg_elapsed = mean(item["elapsed"] for item in results)
    avg_tps = mean(item["tps"] for item in results)
    avg_tpm = mean(item["tpm"] for item in results)

    print("===== 平均统计 =====")
    print(f"平均耗时: {avg_elapsed:.4f} 秒")
    print(f"平均输入 token 数: {avg_prompt_tokens:.2f}")
    print(f"平均输出 token 数: {avg_completion_tokens:.2f}")
    print(f"平均总 token 数: {avg_total_tokens:.2f}")
    print(f"平均输出吞吐: {avg_tps:.2f} tokens/s")
    print(f"平均输出吞吐: {avg_tpm:.2f} tokens/min")


if __name__ == "__main__":
    main()