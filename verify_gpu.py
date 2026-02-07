import torch
import sys

print("=" * 50)
print("GPU VERIFICATION")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
    
    # Test GPU computation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("\n✅ GPU COMPUTATION SUCCESSFUL")
        print(f"   Matrix multiplication on GPU completed")
    except Exception as e:
        print(f"\n❌ GPU error: {e}")
        sys.exit(1)
else:
    print("❌ CUDA not available - falling back to CPU")
    print("   (Models will run on CPU)")

print("=" * 50)
