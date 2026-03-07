import torch
import colorama
from tools.test_frame import test_log

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






@test_log("cross_entropy")
def test_cross_entropy():
    vocab_size = 100
    inputs = torch.randn(64,  vocab_size)
    targets = torch.randint(0, 100, (64, 1)).squeeze(-1)
    print(f"targets shape {targets.shape}")
    ce_loss = cross_entropy(inputs, targets)
    print(f"cross_entropy loss {ce_loss.shape}")

    

if __name__ == "__main__":
    colorama.init(autoreset=True)
    test_cross_entropy()
