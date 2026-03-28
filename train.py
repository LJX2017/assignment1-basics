import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import wandb

from tokenizer import Tokenizer
from transformer.functions import GPTConfig, cross_entropy, get_batch, save_checkpoint, transformer_lm
from transformer.optimizer import AdamW


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a language model")

    parser.add_argument("--corpus-path", type=Path, required=True, help="Path to a tokenized corpus memmap")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto", help="Training device")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--iteration-per-eval", type=int, default=100, help="Validation interval in steps")
    parser.add_argument("--iteration-per-save", type=int, default=500, help="Checkpoint interval in steps")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model_checkpoint"), help="Checkpoint directory")
    parser.add_argument("--wandb-project", type=str, default="cs336-assignment1-basics", help="Weights & Biases project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode",
    )

    parser.add_argument("--sequence-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size")
    parser.add_argument("--n-layer", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward hidden dimension")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta0", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta1", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="AdamW weight decay")
    return parser


def load_tokenizer(vocab_path, merges_path, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    return Tokenizer(vocab_path, merges_path, special_tokens)


def tokenize_dataset(input_path, output_path):
    raise NotImplementedError("Dataset tokenization is still intentionally left unimplemented.")


def resolve_device(device_name: str) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)


def load_dataset_splits(corpus_path: Path) -> tuple[np.memmap, np.memmap]:
    dataset = np.memmap(corpus_path, dtype=np.int16, mode="r")
    split_idx = int(0.9 * len(dataset))
    return dataset[:split_idx], dataset[split_idx:]


def get_train_batch(corpus_path: Path, device: torch.device, batch_size: int, sequence_len: int):
    train, _ = load_dataset_splits(corpus_path)
    while True:
        yield get_batch(train, batch_size, sequence_len, device)


def get_val_batch(corpus_path: Path, device: torch.device, batch_size: int, sequence_len: int):
    _, val = load_dataset_splits(corpus_path)
    while True:
        yield get_batch(val, batch_size, sequence_len, device)


def build_wandb_config(args, device: torch.device) -> dict[str, object]:
    return {
        "corpus_path": str(args.corpus_path),
        "device": str(device),
        "iterations": args.iterations,
        "iteration_per_eval": args.iteration_per_eval,
        "iteration_per_save": args.iteration_per_save,
        "checkpoint_dir": str(args.checkpoint_dir),
        "sequence_len": args.sequence_len,
        "vocab_size": args.vocab_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_ff": args.d_ff,
        "n_embd": args.n_embd,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta0": args.beta0,
        "beta1": args.beta1,
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }


def train_loop(args):
    device = resolve_device(args.device)
    config = GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        n_embd=args.n_embd,
    )
    model = transformer_lm(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta0, args.beta1),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    train_generator = get_train_batch(args.corpus_path, device, args.batch_size, args.sequence_len)
    val_generator = get_val_batch(args.corpus_path, device, args.batch_size, args.sequence_len)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=build_wandb_config(args, device),
    )

    logging.info("Training on %s", device)
    try:
        for step in range(args.iterations):
            model.train()
            optimizer.zero_grad()

            inputs, outputs = next(train_generator)
            logits = model(inputs)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), outputs.reshape(-1))
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/step": step + 1,
                    "train/lr": args.lr,
                },
                step=step + 1,
            )

            if (step + 1) % args.iteration_per_eval == 0:
                model.eval()
                with torch.no_grad():
                    val_inputs, val_outputs = next(val_generator)
                    val_logits = model(val_inputs)
                    val_loss = cross_entropy(
                        val_logits.reshape(-1, val_logits.size(-1)),
                        val_outputs.reshape(-1),
                    )
                logging.info("step=%d train_loss=%.4f val_loss=%.4f", step + 1, loss.item(), val_loss.item())
                wandb.log({"eval/loss": val_loss.item()}, step=step + 1)

            if (step + 1) % args.iteration_per_save == 0:
                checkpoint_path = args.checkpoint_dir / f"checkpoint_{step + 1}.pt"
                save_checkpoint(model, optimizer, step + 1, checkpoint_path)
                wandb.log({"checkpoint/step": step + 1}, step=step + 1)
    finally:
        wandb_run.finish()

    return model, optimizer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
