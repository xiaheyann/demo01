import importlib

qpass = importlib.import_module(
    "vllm_ascend.compilation.passes.qknorm_rope_fusion_pass"
)

def _noop(*args, **kwargs):
    return None

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

from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    main()