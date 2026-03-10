import os
import time
import argparse
import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
import swanlab
import numpy as np
from tqdm import tqdm
from cs336_basics.module import TransformerLM
from cs336_basics.utils import get_batch, cross_entropy, save_checkpoint, load_checkpoint
from cs336_basics.optimizer import AdamW, CosineAnnealingSchedue, grad_clip


@dataclass
class TrainConfig:
    begin_iters: int = 0
    end_iters: int = 10000
    save_interval: int = 10000
    lr: float = 0.001
    max_norm: float = 1.0
    min_lr: float = 6e-5
    batch_size: int = 32
    max_seq_len: int = 1000
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    device: str = "cuda"
    seed: int = 42
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    vocab_path: Optional[str] = None
    output_path: Optional[str] = None
    theta: int = 10000
    vocab_size: int = 10000
    warmup_iter: int = 1000
    use_rmsnorm: bool = True
    use_post_norm: bool = False
    use_rope: bool = True
    use_swiglu: bool = True
    use_silu: bool = True
    config: Optional[str] = field(default=None, repr=False)
    exp_name: Optional[str] = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def validate_config(config: TrainConfig) -> None:
    required_paths = {
        "train_data_path": config.train_data_path,
        "val_data_path": config.val_data_path,
        "output_path": config.output_path,
    }
    missing = [name for name, path in required_paths.items() if not path]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    if config.save_interval <= 0:
        raise ValueError("save_interval must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if config.end_iters < 0:
        raise ValueError("end_iters must be >= 0")
    if config.begin_iters < 0:
        raise ValueError("begin_iters must be >= 0")
    if config.begin_iters > config.end_iters:
        raise ValueError("begin_iters cannot be greater than end_iters")
    if config.warmup_iter < 0:
        raise ValueError("warmup_iter must be >= 0")


def train(config: TrainConfig):
    validate_config(config)
    os.makedirs(config.output_path, exist_ok=True)

    # load dataset
    train_data = np.memmap(config.train_data_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(config.val_data_path, dtype=np.uint16, mode="r")

    if len(train_data) <= config.max_seq_len:
        raise ValueError(
            f"train dataset length ({len(train_data)}) must be > max_seq_len ({config.max_seq_len})"
        )
    if len(valid_data) <= config.max_seq_len:
        raise ValueError(
            f"val dataset length ({len(valid_data)}) must be > max_seq_len ({config.max_seq_len})"
        )

    print(f"train_dataset len {len(train_data)}")
    print(f"valid_dataset len {len(valid_data)}")

    swanlab.init(
        project="cs336_lab1_story", 
        experiment_name=config.exp_name or f"train_{int(time.time())}",
        config=asdict(config),
    )

    print(f"Config {config}")

    # init model optimizer schedule

    model = TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        theta=config.theta,
        vocab_size=config.vocab_size,
        use_rmsnorm=config.use_rmsnorm,
        use_post_norm=config.use_post_norm,
        use_rope=config.use_rope,
        use_swiglu=config.use_swiglu,
        use_silu=config.use_silu,
        device=config.device,
        dtype=torch.float32,
    ).to(config.device)


    optimizer = AdamW(model.parameters(), config.lr, 0.01, (0.9, 0.999))
    schedule = CosineAnnealingSchedue(optimizer, config.lr, config.min_lr, config.warmup_iter, config.end_iters)
    start_iter = config.begin_iters
    ckpt_path = os.path.join(config.output_path, "ckpt.pth")
    if os.path.exists(ckpt_path):
        last_iter = load_checkpoint(ckpt_path, model, optimizer)
        start_iter = max(start_iter, last_iter + 1)
        print(f"Resuming from checkpoint iteration {last_iter}, start from {start_iter}")

    schedule.ti = start_iter

    process_bar = tqdm(range(start_iter, config.end_iters), desc="llm train")
    last_completed_iter = start_iter - 1

    # train loop
    for iter in process_bar:
        # Set per-iteration LR before the optimizer update.
        schedule.step()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_batch(train_data, config.batch_size, config.max_seq_len, config.device)

        outputs = model(inputs)

        loss = cross_entropy(outputs, targets)

        loss.backward()

        grad_clip(model.parameters(), config.max_norm)

        optimizer.step()
        last_completed_iter = iter

        if iter % 100 == 0 or iter == config.end_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(valid_data, config.batch_size, config.max_seq_len, config.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter {iter}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}")
                swanlab.log({
                    "train/loss": loss.item(),
                    "val/loss": v_loss.item(),
                    "iter": iter + 1,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, iter + 1)

                process_bar.set_postfix({"loss": loss.item()})

        if (iter + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, iter, os.path.join(config.output_path, "ckpt.pth"))

    save_checkpoint(model, optimizer, last_completed_iter, os.path.join(config.output_path, "ckpt.pth"))
    process_bar.close()



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # path
    parse.add_argument("--config", type=str, default=None, help="Path to yaml config file")
    parse.add_argument("--begin_iters", type=int, default=0)
    parse.add_argument("--end_iters", type=int, default=10000)
    parse.add_argument("--save_interval", type=int, default=100)
    parse.add_argument("--train_data_path", type=str)
    parse.add_argument("--val_data_path", type=str)
    parse.add_argument("--output_path", type=str)

    # model params
    parse.add_argument("--vocab_path", type=str)
    parse.add_argument("--vocab_size", type=int, default=1000)
    parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--d_model", type=int, default=512)
    parse.add_argument("--d_ff", type=int, default=2048)
    parse.add_argument("--num_layers", type=int, default=6)
    parse.add_argument("--num_heads", type=int, default=8)
    parse.add_argument("--max_seq_len", type=int, default=256)

    # hyper 
    parse.add_argument("--lr", type=float, default=6e-4)
    parse.add_argument("--warmup_iter", type=int, default=1000)
    parse.add_argument("--max_norm", type=float, default=1.0)
    parse.add_argument("--min_lr", type=float, default=6e-5)
    parse.add_argument("--theta", type=int, default=10000)
    parse.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--seed", type=int, default=42)
    parse.add_argument("--exp_name", type=str, default=None)
    parse.add_argument("--use_rmsnorm", type=str2bool, default=True)
    parse.add_argument("--use_post_norm", type=str2bool, default=False)
    parse.add_argument("--use_rope", type=str2bool, default=True)
    parse.add_argument("--use_swiglu", type=str2bool, default=True)
    parse.add_argument("--use_silu", type=str2bool, default=True)

    cli_args = parse.parse_args()

    yaml_config = {}
    if cli_args.config:
        with open(cli_args.config, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    default_config = TrainConfig()
    default_dict = {k: v for k, v in vars(default_config).items() if v is not None}

    merged = {**default_dict, **yaml_config}
    merged = {k: v for k, v in merged.items() if v is not None}

    parse.set_defaults(**merged)

    args = parse.parse_args()

    config = TrainConfig(**vars(args))

    os.makedirs(config.output_path, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train(config)
