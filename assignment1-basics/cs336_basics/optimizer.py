import torch
import math
import torch.nn as nn
from typing import Optional
from collections.abc import Callable, Iterable
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F, init
from tools.test_frame import test_log
from cs336_basics.module import Linear


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate : {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)

                grad = p.grad.clone()

                p.data -= lr / math.sqrt(t + 1) * grad

                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params: list, lr: float = 1e-3, weight_decay: float = 0.01, betas: tuple = (0.9, 0.999), eps: float = 1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate : {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        beta1, beta2 = self.betas

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                grad = p.grad

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = v.sqrt().add_(self.eps)

                with torch.no_grad():
                    p.addcdiv_(m, denom, value=-step_size)
                    p.mul_(1 - lr * self.weight_decay)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


class CosineAnnealingSchedue:
    def __init__(self, optimizer: torch.optim.Optimizer, 
                    max_learning_rate: float, 
                    min_learning_rate: float,
                    warmup_iters: int,
                    cosine_cycle_iters: int):
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def step(self):
        pass

    def get_iter_lr(self, it):
        if it < self.warmup_iters:
            return it / self.warmup_iters * self.max_learning_rate
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate
        else:
            return self.min_learning_rate + (1 + math.cos((it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters) * math.pi)) * (self.max_learning_rate - self.min_learning_rate) / 2
    
        


@test_log("SGD")
def test_SGD():
    in_shape = 3
    out_shape = 4
    bs = 10
    batch_size = 5
    linear = Linear(in_shape, out_shape)
    # generate data
    target_mat = torch.randn((in_shape, out_shape))
    data_in = torch.randn((bs, in_shape))
    data_out = data_in @ target_mat

    opt = SGD(linear.parameters(), lr=1)
    groups = opt.param_groups
    print(f"groups type {type(groups)}")

    for group in groups:
        print(type(group))
        for k, v in group.items():
            print(f"key_type {type(k)}")
            print(f"k {k}, v {v}")

    num_batches = bs // batch_size
    data_in_shuffled = data_in
    data_out_shuffled = data_out
    for t in range(1000):
        opt.zero_grad()

        if t % num_batches == 0:
            perm = torch.randperm(bs)
            data_in_shuffled = data_in[perm]
            data_out_shuffled = data_out[perm]

        batch_idx = t % num_batches
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        index_choose = torch.arange(start_idx, end_idx)

        out = linear(data_in_shuffled[index_choose])

        loss = (data_out_shuffled[index_choose] - out).sum().abs().mean()
        print(f"iter {t} loss {loss}")
        print(type(linear.weight.data.grad))
        loss.backward()

        opt.step()


@test_log("test_adamw_basic_update")
def test_adamw_basic_update():
    """测试 AdamW 基本更新功能"""
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = AdamW(model.parameters(), lr=0.1, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

    initial_weight = model.weight.data.clone()

    x = torch.randn(2, 3)
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt.step()

    assert not torch.allclose(model.weight.data, initial_weight), "Parameters should have been updated"
    print("test_adamw_basic_update: PASSED")


@test_log("test_adamw_with_weight_decay")
def test_adamw_with_weight_decay():
    """测试 weight decay 功能"""
    torch.manual_seed(42)

    x = torch.randn(2, 3)

    model_no_decay = torch.nn.Linear(3, 2, bias=False)
    model_no_decay.weight.data.fill_(1.0)
    opt_no_decay = AdamW(model_no_decay.parameters(), lr=0.1, weight_decay=0.0)

    for _ in range(10):
        opt_no_decay.zero_grad()
        out = model_no_decay(x)
        loss = out.sum()
        loss.backward()
        opt_no_decay.step()

    model_with_decay = torch.nn.Linear(3, 2, bias=False)
    model_with_decay.weight.data.fill_(1.0)
    opt_with_decay = AdamW(model_with_decay.parameters(), lr=0.1, weight_decay=0.1)

    for _ in range(10):
        opt_with_decay.zero_grad()
        out = model_with_decay(x)
        loss = out.sum()
        loss.backward()
        opt_with_decay.step()

    assert torch.norm(model_with_decay.weight.data) < torch.norm(model_no_decay.weight.data), (
        "Weight decay should reduce parameter magnitude"
    )
    print("test_adamw_with_weight_decay: PASSED")

@test_log("test_adamw_matches_pytorch")
def test_adamw_matches_pytorch():
    """测试与 PyTorch AdamW 结果是否接近"""
    torch.manual_seed(42)

    model_pytorch = torch.nn.Linear(3, 2, bias=False)
    opt_pytorch = torch.optim.AdamW(
        model_pytorch.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    model_ours = torch.nn.Linear(3, 2, bias=False)
    opt_ours = AdamW(model_ours.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    with torch.no_grad():
        model_ours.weight.copy_(model_pytorch.weight)

    for _ in range(100):
        x = torch.rand(2, 3)
        y_target = torch.tensor([x[0] + x[1], -x[2]])

        opt_pytorch.zero_grad()
        loss_pytorch = ((model_pytorch(x) - y_target) ** 2).sum()
        loss_pytorch.backward()
        opt_pytorch.step()

        opt_ours.zero_grad()
        loss_ours = ((model_ours(x) - y_target) ** 2).sum()
        loss_ours.backward()
        opt_ours.step()

    is_close = torch.allclose(model_ours.weight, model_pytorch.weight, atol=1e-3)
    max_diff = torch.max(torch.abs(model_ours.weight - model_pytorch.weight)).item()
    print(f"test_adamw_matches_pytorch: max_diff = {max_diff:.6e}")
    assert is_close, f"Parameters differ from PyTorch AdamW by {max_diff}"
    print("test_adamw_matches_pytorch: PASSED")


@test_log("test_adamw_state_persistence")
def test_adamw_state_persistence():
    """测试状态在多次迭代中是否正确保存"""
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = AdamW(model.parameters(), lr=0.1)

    x = torch.randn(2, 3)
    out = model(x)
    loss = out.sum()
    loss.backward()
    opt.step()

    state = opt.state[model.weight]
    assert "m" in state, "First moment estimate should be saved"
    assert "v" in state, "Second moment estimate should be saved"
    assert "t" in state, "Time step should be saved"
    assert state["t"] == 1, f"Time step should be 1, got {state['t']}"

    out = model(x)
    loss = out.sum()
    loss.backward()
    opt.step()

    assert opt.state[model.weight]["t"] == 2, "Time step should be 2 after second step"
    print("test_adamw_state_persistence: PASSED")


@test_log("test_adamw_handles_none_grad")
def test_adamw_handles_none_grad():
    """测试处理无梯度的参数"""
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = AdamW(model.parameters(), lr=0.1)

    model.weight.grad = None

    initial_weight = model.weight.data.clone()
    opt.step()

    assert torch.allclose(model.weight.data, initial_weight), "Parameters with None grad should not be updated"
    print("test_adamw_handles_none_grad: PASSED")


@test_log("test_adamw_lr_validation")
def test_adamw_lr_validation():
    """测试 lr 参数验证"""
    try:
        AdamW([torch.nn.Parameter(torch.randn(3, 2))], lr=-0.1)
        assert False, "Should raise ValueError for negative lr"
    except ValueError as e:
        assert "learning rate" in str(e).lower()
        print("test_adamw_lr_validation: PASSED")


@test_log("test_adamw_different_betas")
def test_adamw_different_betas():
    """测试不同的 betas 参数"""
    torch.manual_seed(42)
    model1 = torch.nn.Linear(3, 2, bias=False)
    model2 = torch.nn.Linear(3, 2, bias=False)

    opt1 = AdamW(model1.parameters(), lr=0.1, betas=(0.5, 0.999))
    opt2 = AdamW(model2.parameters(), lr=0.1, betas=(0.9, 0.999))

    x = torch.randn(2, 3)

    for _ in range(50):
        opt1.zero_grad()
        out = model1(x)
        loss = out.sum()
        loss.backward()
        opt1.step()

        opt2.zero_grad()
        out = model2(x)
        loss = out.sum()
        loss.backward()
        opt2.step()

    assert not torch.allclose(model1.weight.data, model2.weight.data), (
        "Different betas should produce different results"
    )
    print("test_adamw_different_betas: PASSED")



if __name__ == "__main__":
    # test_SGD()
    test_adamw_basic_update()
    test_adamw_with_weight_decay()
    print("\n=== All AdamW tests PASSED ===")

