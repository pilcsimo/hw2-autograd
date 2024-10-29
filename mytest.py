import torch
from engine import Tensor

import torch.nn as nn

def test_cross_entropy_loss():
    # Create dummy data
    input_data = torch.randn(3, 5, requires_grad=True)
    target_data = torch.tensor([1, 0, 4])

    # PyTorch cross entropy loss
    criterion_torch = nn.CrossEntropyLoss()
    loss_torch = criterion_torch(input_data, target_data)
    loss_torch.backward()
    grad_torch = input_data.grad.clone()

    # Reset gradients

    # Engine cross entropy loss
    input_tensor = Tensor(input_data.detach().numpy())
    print(input_tensor.data)
    target_tensor = target_data.detach().numpy()
    print(target_tensor)
    #breakpoint()
    loss_engine = input_tensor.cross_entropy_loss(target_tensor)
    loss_engine.backward()
    grad_engine = input_tensor.grad

    # Check if the outputs are close
    print("PyTorch Loss:", loss_torch.item())
    print("Engine Loss:", loss_engine.data)
    print("PyTorch Gradients:", grad_torch)
    print("Engine Gradients:", grad_engine)
    breakpoint()
    assert torch.isclose(torch.tensor(loss_engine.data), loss_torch, atol=1e-6), "Loss outputs are not close"

    # Check if the gradients are close
    assert torch.allclose(torch.tensor(grad_engine), grad_torch, atol=1e-6), "Gradients are not close"

if __name__ == "__main__":
    test_cross_entropy_loss()
    print("All tests finished!")