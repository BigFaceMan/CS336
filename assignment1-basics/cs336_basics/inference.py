import torch
import argparse
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.module import TransformerLM



@dataclass
class InferConfig:
    max_seq_len: int = 1000
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    window_size: int = 1000
    d_ff: int = 2048
    device: str = "cuda"
    seed: int = 42
    theta: int = 10000
    use_rmsnorm: bool = True
    use_post_norm: bool = False
    use_rope: bool = True
    use_swiglu: bool = True
    use_silu: bool = True
    train_data_path: Optional[str] = None
    model_path: Optional[str] = None
    vocab_path: Optional[str] = None
    vocab_size: int = 10000
    config: Optional[str] = field(default=None, repr=False)



def infer(config: InferConfig):
    tokenizer = Tokenizer.from_files(config.vocab_path + "/vocab.json", config.vocab_path + "/merges.json")
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


    checkpoint_path = config.model_path
    checkpoint = torch.load(config.model_path,  map_location=args.device)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()

    special_tokens = tokenizer.special_tokens
    special_tokens_ids = [tokenizer.bytes_2_id[token.encode("utf-8")] for token in special_tokens]

    while True:
        input_text = input("Enter a prompt (or 'exit' to quit): ")
        if input_text.lower() == "exit":
            break

        input_ids = tokenizer.encode(input_text)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=config.window_size, eos_tokens_id=special_tokens_ids, temperature=0.9, top_p=0.8)

        output_text = tokenizer.decode(output_ids.squeeze().tolist())
        print(f"Generated text: {output_text}")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    # path
    parse.add_argument("--config", type=str, default=None, help="Path to yaml config file")
    parse.add_argument("--model_path", type=str)

    # model params
    parse.add_argument("--vocab_path", type=str)
    parse.add_argument("--vocab_size", type=int, default=1000)
    parse.add_argument("--d_model", type=int, default=512)
    parse.add_argument("--d_ff", type=int, default=2048)
    parse.add_argument("--num_layers", type=int, default=6)
    parse.add_argument("--num_heads", type=int, default=8)
    parse.add_argument("--theta", type=int, default=10000)
    parse.add_argument("--window_size", type=int, default=1000)
    parse.add_argument("--max_seq_len", type=int, default=256)
    parse.add_argument("--use_rmsnorm", type=lambda x: x.lower() == "true", default=True)
    parse.add_argument("--use_post_norm", type=lambda x: x.lower() == "true", default=False)
    parse.add_argument("--use_rope", type=lambda x: x.lower() == "true", default=True)
    parse.add_argument("--use_swiglu", type=lambda x: x.lower() == "true", default=True)
    parse.add_argument("--use_silu", type=lambda x: x.lower() == "true", default=True)

    # hyper 
    parse.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--seed", type=int, default=42)

    cli_args = parse.parse_args()

    yaml_config = {}
    if cli_args.config:
        with open(cli_args.config, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    default_config = InferConfig()
    default_dict = {k: v for k, v in vars(default_config).items() if v is not None}

    merged = {**default_dict, **yaml_config}
    merged = {k: v for k, v in merged.items() if v is not None}

    parse.set_defaults(**merged)

    args = parse.parse_args()

    config = InferConfig(**vars(args))
    infer(config)
