"""Start the XPolicyLab OLA_SEM (UniWAM) policy server with its umT5-XXL text encoder on the CPU.

Same reason as fastwam_policy_server.py, but stricter: the main model alone takes ~19 GB, and the upstream policy
builds the ~11 GB encoder directly on the GPU, which overflows a 32 GB card before the simulator even starts.
Here the encoder is built on the CPU and its embeddings are cached per instruction (fixed per episode).
Arguments are those of XPolicyLab/setup_policy_server.py.
"""
import runpy
import sys

import torch

from XPolicyLab.policy.OLA_SEM.model import UPSTREAM_POLICY_DIR

sys.path.insert(0, str(UPSTREAM_POLICY_DIR))  # what the adapter's _load_policy does before importing deploy_policy
from XPolicyLab.policy.OLA_SEM.ola_sem.inference.robotwin.Motus import deploy_policy  # noqa: E402


class _CpuCachedT5:
    def __init__(self, **kwargs):
        self.encoder = _T5EncoderModel(**{**kwargs, "device": "cpu"})
        self.cache = {}

    def __call__(self, texts, device):
        key = tuple(texts)
        if key not in self.cache:
            with torch.no_grad():
                self.cache[key] = self.encoder(texts, "cpu")
        return [u.to(device) for u in self.cache[key]]


_T5EncoderModel = deploy_policy.T5EncoderModel
deploy_policy.T5EncoderModel = _CpuCachedT5
sys.argv[0] = "XPolicyLab/setup_policy_server.py"
runpy.run_path("XPolicyLab/setup_policy_server.py", run_name="__main__")
