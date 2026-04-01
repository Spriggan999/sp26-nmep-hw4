from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from seq2seq.tokenizer.bpe_tokenizer import BPETokenizer


tokenizer = BPETokenizer(model="gpt2")


class ScreenplayDataset(Dataset):
    def __init__(
        self,
        screenplay_path,
        block_size,
        verbose = False,
    ):
        self.block_size = block_size

        all_tokens = []

        txt_files = sorted(screenplay_path.glob("*.txt"))

        for path in txt_files:
            text = path.read_text(encoding="utf-8")

            file_tokens = []
            file_tokens.append(tokenizer.bos_token_id)
            file_tokens.extend(tokenizer.encode(text).tolist())
            file_tokens.append(tokenizer.eos_token_id)

            all_tokens.extend(file_tokens)

            if verbose:
                print(path, len(file_tokens))

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        # Need block_size + 1 tokens per sample so we can shift by 1
        if len(self.tokens) < self.block_size + 1:
            raise ValueError(
                f"Corpus too small: need at least {self.block_size + 1} tokens, "
                f"got {len(self.tokens)}"
            )

        self.start_idxs = list(
            range(0, len(self.tokens) - (self.block_size + 1) + 1, self.block_size)
        )

        print(f"Total tokens: {len(self.tokens)}")
        print(f"Num samples: {len(self.start_idxs)}")
        print(f"Block size: {self.block_size}")

    def __len__(self):
        return len(self.start_idxs)

    def __getitem__(self, idx: int):
        start = self.start_idxs[idx]
        end = start + self.block_size + 1
        return self.tokens[start:end]


def collate_fn(batch):
    pad_in = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return pad_in


if __name__ == "__main__":
    data = ScreenplayDataset(Path("data/lm/"), 512, verbose=True)

    print("Samples: ", len(data))

    idx = torch.randint(0, len(data), (10,)).tolist()

    for i in idx:
        para = data[i]
        print(tokenizer.decode(para))
        print("-" * 80)
