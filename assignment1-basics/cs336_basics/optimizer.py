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
        self.optimizer: torch.optim.Optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.ti = 0

    def step(self):
        lr = self.get_iter_lr(self.ti)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
            
        self.ti += 1

    def get_iter_lr(self, it):
        if it < self.warmup_iters:
            return it / self.warmup_iters * self.max_learning_rate

        if it < self.cosine_cycle_iters:
            progress = (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_learning_rate + cosine_decay * (self.max_learning_rate - self.min_learning_rate)

        return self.min_learning_rate

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    gdl2 = torch.tensor(0.0)
    for param in parameters:
        if param.grad is None:
            continue
        gdl2 += (param.grad**2).sum()

    gdl2 = torch.sqrt(gdl2)
    
    if gdl2 > max_l2_norm:
        inv = max_l2_norm / (gdl2 + eps)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.mul_(inv)
    
        


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

@test_log("grad_clip")
def test_grad_clip():
    parameters = [nn.Parameter(torch.randn((100, 3))) * 100 for i in range(100)]
    parameters = grad_clip(parameters, 1000)

if __name__ == "__main__":
    # test_SGD()
    test_adamw_basic_update()
    test_adamw_with_weight_decay()
    test_grad_clip()


