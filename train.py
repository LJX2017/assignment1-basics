import argparse
import logging
import pickle

import numpy as np
import torch

from tokenizer import Tokenizer
from transformer.functions import GPTConfig, cross_entropy, get_batch, load_checkpoint, save_checkpoint, transformer_lm
from transformer.optimizer import AdamW

# @dataclass
# class GPTConfig:
#     sequence_len: int = 2048
#     vocab_size: int = 32768
#     n_layer: int = 12
#     n_head: int = 6  # number of query heads
#     d_ff: int = 2048
#     # n_kv_head: int = 6 # number of key/value heads
#     n_embd: int = 768


parser = argparse.ArgumentParser(description="training LM")
# model configs
parser.add_argument("sequence_len", metavar="int", type=int, help="sequence_len", default=2048)
parser.add_argument("vocab_size", metavar="int", type=int, help="vocab_size", default=32768)
parser.add_argument("n_layer", metavar="int", type=int, help="n_layer", default=12)
parser.add_argument("n_head", metavar="int", type=int, help="n_head", default=6)
parser.add_argument("d_ff", metavar="int", type=int, help="d_ff", default=2048)
parser.add_argument("n_embd", metavar="int", type=int, help="n_embd", default=768)
parser.add_argument("batch_size", metavar="int", type=int, help="batch_size", default=12)

# hyparams
# class AdamW(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001):

parser.add_argument("lr", metavar="float", type=float, help="lr", default=1e-3)
parser.add_argument("beta0", metavar="float", type=float, help="beta0", default=0.9)
parser.add_argument("beta1", metavar="float", type=float, help="beta1", default=0.999)
parser.add_argument("eps", metavar="float", type=float, help="eps", default=1e-8)
parser.add_argument("weight_decay", metavar="float", type=float, help="weight_decay", default=0.001)

args = parser.parse_args()


def load_tokenizer(vocab_path, merges_path, special_tokens=["<|endoftext|>"]):
    return Tokenizer(vocab_path, merges_path, special_tokens)


def tokenize_dataset(input_path, output_path):
    pass


def get_train_batch(corpus_path, device):
    ...
    newfp = np.memmap(corpus_path, dtype=np.int16, mode="r")
    examples = len(newfp)
    train = newfp[: 0.9 * examples]
    while True:
        train_data = get_batch(train, args.batch_size, args.sequence_len, device)
        yield train_data


def get_val_batch(corpus_path, device):
    newfp = np.memmap(corpus_path, dtype=np.int16, mode="r")
    examples = len(newfp)
    val = newfp[0.9 * examples :]
    while True:
        val_data = get_batch(val, args.batch_size, args.sequence_len, device)
        yield val_data


def train_loop(iter, iteration_per_eval, iteration_per_save):
    training_config = ...
    model = transformer_lm(training_config)
    optimizer = AdamW(model.parameters, args.lr, ...)
    train_generator = get_train_batch()
    val_generator = get_val_batch()
    for step in range(iter):
        train_data, val_data = load_data(...)
        optimizer.zero_grad()
        inputs, outputs = train_generator.next()
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), outputs.view(-1))
        loss.back()
        optimizer.step()
        if step % iteration_per_eval == iteration_per_eval - 1:
            with torch.no_grad():
                inputs, outputs = val_generator.next()
                logits = model(inputs)  # B T V
                loss = cross_entropy(logits.view(-1, logits.size(-1)), outputs.view(-1))
                logging.log(loss)
        if step % iteration_per_save == iteration_per_save - 1:
            save_checkpoint(model, optimizer, step, "model_checkpoint/checkpoint_" + str(step))
