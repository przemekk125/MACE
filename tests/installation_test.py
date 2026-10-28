import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from e3nn import o3

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# Example: Define Irreducible Representations (1x0e + 1x1o)
irreps_in1 = cue.Irreps("O3", "1x0e + 1x1o")
irreps_in2 = cue.Irreps("O3", "1x0e + 1x1o")
irreps_out = cue.Irreps("O3", "1x0e + 1x1o")

# Create a cuEquivariance Tensor Product
tp_layer = cuet.FullyConnectedTensorProduct(
    irreps_in1, irreps_in2, irreps_out, layout=cue.mul_ir, internal_weights=False
)

# Test a forward pass on GPU
x1 = torch.randn(10, irreps_in1.dim)
x2 = torch.randn(10, irreps_in2.dim)
weight = torch.randn(10, tp_layer.weight_numel)

out = tp_layer(x1, x2, weight)
print("Forward pass output shape:", out.shape)
print("cuEquivariance correctly installed and working!")

# 1. Define O(3) irreducible representations (e.g., scalars and vectors)
irreps_in1 = o3.Irreps("2x0e + 1x1o")  # two 0-e (scalars) + one 1-o (vector)
irreps_in2 = o3.Irreps("1x1o")  # one 1-o (vector)
irreps_out = o3.Irreps(
    "2x0e + 1x1o + 1x2e"
)  # two 0-e (scalars) + one 1-o (vector) + one 2-e (tensor)

print(f"Input Irreps 1: {irreps_in1}")
print(f"Dimension of Irreps 1: {irreps_in1.dim}")

# 2. Create an e3nn Fully Connected Tensor Product layer
tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

# 3. Create dummy batch inputs matching the dimensions
batch_size = 5
x1 = torch.randn(batch_size, irreps_in1.dim)
x2 = torch.randn(batch_size, irreps_in2.dim)

# 4. Forward pass
out = tp(x1, x2)

print("\ne3nn successfully installed and working!")
print(f"Output shape: {out.shape}")
