"""Start the XPolicyLab FastWAM policy server with its umT5-XXL text encoder on the CPU.

The text encoder (~11 GB in bf16) only embeds the instruction, which is fixed per episode, yet it sits on the GPU
next to the video/action experts. With the simulator also rendering on the same 32 GB GPU, renders failed with
"cannot create buffer". Here the encoder moves to the CPU once the model is built, and embeddings are cached per
prompt, so each instruction is encoded once. Everything else matches XPolicyLab/setup_policy_server.py, whose
arguments this script takes unchanged.
"""
import runpy
import sys

import torch

from fastwam.models.wan22.fastwam import FastWAM
from XPolicyLab.policy.FastWAM import model as fastwam_adapter

_cache = {}
_original_init = fastwam_adapter.Model.__init__


def _encode_prompt_on_cpu(self, prompt):
    key = prompt if isinstance(prompt, str) else tuple(prompt)
    if key not in _cache:
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        mask = mask.to(dtype=torch.bool)
        with torch.no_grad():
            prompt_emb = self.text_encoder(ids, mask)
        # Same post-processing as FastWAM.encode_prompt.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        _cache[key] = (prompt_emb, torch.ones_like(mask))
    prompt_emb, mask = _cache[key]
    return prompt_emb.to(device=self.device), mask.to(device=self.device)


def _init_then_offload_text_encoder(self, model_cfg):
    _original_init(self, model_cfg)
    fastwam = getattr(getattr(self, "model", None), "model", None)
    if isinstance(fastwam, FastWAM) and fastwam.text_encoder is not None:
        fastwam.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        print("[fastwam_policy_server] text encoder moved to CPU", flush=True)


FastWAM.encode_prompt = _encode_prompt_on_cpu
fastwam_adapter.Model.__init__ = _init_then_offload_text_encoder
sys.argv[0] = "XPolicyLab/setup_policy_server.py"
runpy.run_path("XPolicyLab/setup_policy_server.py", run_name="__main__")
