#!/usr/bin/env python3
"""Gate A: verify that the installed PyTorch build can execute on sm_120."""

import platform
import sys

import torch


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"device_name={name}")
    print(f"device_capability={capability}")
    print(f"compiled_arch_list={arch_list}")

    if capability != (12, 0):
        raise RuntimeError(f"unexpected device capability: {capability}")
    if "sm_120" not in arch_list:
        raise RuntimeError("PyTorch build does not list sm_120")

    torch.manual_seed(20260719)
    a = torch.randn((2048, 2048), device="cuda", dtype=torch.float32)
    b = torch.randn((2048, 2048), device="cuda", dtype=torch.float32)
    c = a @ b
    torch.cuda.synchronize()

    checksum = c.abs().mean().item()
    finite = torch.isfinite(c).all().item()
    print(f"matmul_abs_mean={checksum:.6f}")
    print(f"matmul_all_finite={finite}")
    if not finite or checksum <= 0.0:
        raise RuntimeError("CUDA matrix multiplication produced invalid output")

    print("GATE_A=PASS")


if __name__ == "__main__":
    main()

