import platform
import subprocess

def get_cpu_name():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
    except Exception:
        return platform.processor() or "Unknown CPU"

def get_gpu_name():
    # 1. PyTorch (most reliable for CUDA devices)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    # 2. NVIDIA SMI (works in most CUDA environments)
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
    except Exception:
        pass

    # 3. Windows WMIC fallback
    if platform.system() == "Windows":
        try:
            return subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                encoding="utf-8"
            ).split("\n")[1].strip()
        except Exception:
            pass

    # 4. Linux/macOS lspci fallback
    try:
        return subprocess.check_output(
            "lspci | grep -i 'vga\\|3d'",  # also matches discrete GPUs labeled "3D controller"
            shell=True, encoding="utf-8"
        ).split("\n")[0]
    except Exception:
        return "Unknown GPU"