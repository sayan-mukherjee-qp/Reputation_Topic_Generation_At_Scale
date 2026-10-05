"""Print what the container can actually see of the GPU.

Run before a long job: `docker run --rm --gpus all reputation-topic-gpu /app/gpu_check.py`
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    import torch

    print(f"torch           {torch.__version__}")
    print(f"built for CUDA  {torch.version.cuda}")
    print(f"cuDNN           {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\nCUDA NOT AVAILABLE -- the run would fall back to CPU.")
        print("Check that the container was started with `--gpus all` (or a compose")
        print("device reservation) and that the host driver is new enough for this")
        print("torch build: cu130 needs driver >= 580, cu126 needs driver >= 560.")
        return 1

    print(f"devices         {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  [{i}] {props.name}  "
            f"{props.total_memory / 1024 ** 3:.1f} GiB  "
            f"sm_{props.major}{props.minor}"
        )

    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    print(f"\nmatmul smoke test OK ({(x @ x).sum().item():.3e})")

    from sentence_transformers import SentenceTransformer

    # The image bakes one model in and runs offline; EMBEDDING_MODEL names it.
    name = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(name, device="cuda")
    emb = model.encode(["gpu smoke test"], normalize_embeddings=True)
    print(f"sentence-transformers OK ({name}, dim={emb.shape[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
