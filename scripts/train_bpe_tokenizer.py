import argparse
import pickle

from tokenizer.bpe_train import get_tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]

def main() -> None:
    parser = argparse.ArgumentParser(description="train bpe tokenizer")
    parser.add_argument("--data-path", type=str, help="depth path of training task")
    parser.add_argument("--token-num", type=int, default=10000, help="max token number")
    args = parser.parse_args()

    int2byte, merges = get_tokenizer(args.data_path, SPECIAL_TOKENS, args.token_num)

    with open("data/tokenizer.pkl", "wb") as f:
        pickle.dump(int2byte, f)

    with open("data/merges.pkl", "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    main()
