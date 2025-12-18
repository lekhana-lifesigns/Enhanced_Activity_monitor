# scripts/check_hardware.py
"""
Check available hardware acceleration options.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_cuda():
    """Check CUDA/GPU availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            return {
                "available": True,
                "device_name": device_name,
                "device_count": device_count,
                "recommended": "cuda"
            }
        return {"available": False, "recommended": "cpu"}
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}

def check_npu():
    """Check NPU availability."""
    results = {}
    
    # Check Intel NPU
    try:
        import intel_npu_acceleration_library
        results["intel_npu"] = {"available": True}
    except ImportError:
        results["intel_npu"] = {"available": False}
    
    # Check OpenVINO
    try:
        import openvino
        results["openvino"] = {"available": True}
    except ImportError:
        results["openvino"] = {"available": False}
    
    # Check for NPU in system
    import platform
    processor = platform.processor()
    if "Intel" in processor and ("Ultra" in processor or "Core" in processor):
        results["intel_processor"] = {"detected": True, "processor": processor}
    else:
        results["intel_processor"] = {"detected": False}
    
    return results

def check_mps():
    """Check Apple Silicon MPS availability."""
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        if mps_available:
            return {"available": True, "recommended": "mps"}
        return {"available": False}
    except (ImportError, AttributeError):
        return {"available": False}

def main():
    print("=" * 70)
    print("HARDWARE ACCELERATION CHECK")
    print("=" * 70)
    
    # Check CUDA
    print("\n1. CUDA/GPU:")
    print("-" * 70)
    cuda_info = check_cuda()
    if cuda_info.get("available"):
        print(f"  ✅ CUDA Available")
        print(f"  Device: {cuda_info.get('device_name')}")
        print(f"  Count: {cuda_info.get('device_count')}")
        print(f"  Recommended: Set device='cuda' in config/system.yaml")
    else:
        print("  ❌ CUDA Not Available")
        if "error" in cuda_info:
            print(f"  Error: {cuda_info['error']}")
    
    # Check MPS (Apple Silicon)
    print("\n2. Apple Silicon MPS:")
    print("-" * 70)
    mps_info = check_mps()
    if mps_info.get("available"):
        print("  ✅ MPS Available")
        print("  Recommended: Set device='mps' in config/system.yaml")
    else:
        print("  ❌ MPS Not Available")
    
    # Check NPU
    print("\n3. NPU (Neural Processing Unit):")
    print("-" * 70)
    npu_info = check_npu()
    
    if npu_info.get("intel_processor", {}).get("detected"):
        print(f"  ✅ Intel Processor Detected: {npu_info['intel_processor']['processor']}")
        if npu_info.get("intel_npu", {}).get("available"):
            print("  ✅ Intel NPU Acceleration Library Available")
        else:
            print("  ⚠️  Intel NPU Library Not Installed")
            print("     Install: pip install intel-npu-acceleration-library")
        
        if npu_info.get("openvino", {}).get("available"):
            print("  ✅ OpenVINO Available")
        else:
            print("  ⚠️  OpenVINO Not Installed")
            print("     Install: pip install openvino")
    else:
        print("  ❌ Intel NPU Not Detected")
        print("     (NPU only available on Intel Core Ultra processors)")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if cuda_info.get("available"):
        print("\n✅ BEST OPTION: Use CUDA (GPU)")
        print("   Update config/system.yaml:")
        print("   device: \"cuda\"")
        print("   Expected speedup: 5-10x")
    elif mps_info.get("available"):
        print("\n✅ BEST OPTION: Use MPS (Apple Silicon)")
        print("   Update config/system.yaml:")
        print("   device: \"mps\"")
        print("   Expected speedup: 3-5x")
    elif npu_info.get("intel_npu", {}).get("available"):
        print("\n⚠️  OPTION: Use Intel NPU")
        print("   Requires model conversion to OpenVINO IR")
        print("   Expected speedup: 2-3x")
        print("   More complex setup")
    else:
        print("\n✅ CURRENT: Using CPU")
        print("   System works fine on CPU")
        print("   For better performance, consider GPU acceleration")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

