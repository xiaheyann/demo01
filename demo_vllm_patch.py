import os
import importlib
import ctypes

# # 如果你还会用 VSCode/debugpy，这几行最好保留
# LIB_DIRS = [
#     "/app/.venv/lib/python3.10/site-packages/xlite/lib",
#     "/workspace/GVirt/xlite/cmake_build/lib",
# ]
# LIBS = [
#     "libxlite_kernels_common_npu.so",
#     "libxlite_kernels_f16_npu.so",
#     "libxlite_kernels_bf16_npu.so",
#     "libxlite.so",
# ]
# os.environ["LD_LIBRARY_PATH"] = ":".join(LIB_DIRS + [os.environ.get("LD_LIBRARY_PATH", "")])
# for lib_dir in LIB_DIRS:
#     for lib in LIBS:
#         p = os.path.join(lib_dir, lib)
#         if os.path.exists(p):
#             ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

# 关键：直接禁用 qknorm+rope 融合 pass
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
        tokenizer="model-bin/xin7697/Qwen3-14B-W8A8-Ascend",
        quantization="ascend",
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

    outputs = llm.generate(
        ["你好，简单介绍一下你自己。"],
        SamplingParams(temperature=0.0, max_tokens=64),
    )

    for o in outputs:
        print(o.outputs[0].text)

if __name__ == "__main__":
    main()