"""Deterministic flags in this service's spawned Python workers only."""
import os

if os.environ.get("TERMMAP_VLLM_WORKER") == "1":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random
    import numpy as np
    import torch
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
