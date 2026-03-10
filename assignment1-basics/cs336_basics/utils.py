import torch
import os
import colorama
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import numpy as np
from tools.test_frame import log_test


def softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    inv = exp_x.sum(dim=-1, keepdim=True)
    return exp_x / inv


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(x).sum(dim=-1, keepdim=True))
    return x - log_sum_exp


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
        unnormalized logit of jth class for the ith example.
    targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
        Each value must be between 0 and `num_classes - 1`.
    """
    log_probs = log_softmax(inputs)
    loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    n = len(dataset)
    ix = torch.randint(0, n - context_length, (batch_size,)).tolist()
    x_np = np.stack([dataset[i : i + context_length] for i in ix], axis=0)
    y_np = np.stack([dataset[i + 1 : i + 1 + context_length] for i in ix], axis=0)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
    return x, y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out_path: str | os.PathLike | BinaryIO | IO[bytes]):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    checkpoint = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "iteration": iteration
    }
    torch.save(checkpoint, out_path)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


    


@log_test("cross_entropy")
def test_cross_entropy():
    vocab_size = 100
    inputs = torch.randn(64, vocab_size)
    targets = torch.randint(0, 100, (64, 1)).squeeze(-1)
    print(f"targets shape {targets.shape}")
    ce_loss = cross_entropy(inputs, targets)
    print(f"cross_entropy loss {ce_loss.shape}")


@log_test("get_batch")
def test_get_batch():
    dataset = np.arange(100)
    batch_size = 4
    context_length = 3
    device = "cpu"

    inputs, targets = get_batch(dataset, batch_size, context_length, device)

    print(f"inputs shape: {inputs.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"inputs: {inputs}")
    print(f"targets: {targets}")

    assert inputs.shape == (batch_size, context_length), (
        f"Expected shape {(batch_size, context_length)}, got {inputs.shape}"
    )
    assert targets.shape == (batch_size, context_length), (
        f"Expected shape {(batch_size, context_length)}, got {targets.shape}"
    )

    for i in range(batch_size):
        for j in range(context_length - 1):
            assert targets[i, j] == inputs[i, j + 1], f"Mismatch at batch {i}, position {j}"


if __name__ == "__main__":
    colorama.init(autoreset=True)
    test_cross_entropy()
    test_get_batch()
