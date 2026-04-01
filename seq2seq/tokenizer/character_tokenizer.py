from .tokenizer import Tokenizer

import torch


class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        If verbose is True, prints out the vocabulary.

        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = """aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}’•–í€óá«»… º◦©ö°äµ—ø­·òãñ―½¼γ®⇒²▪−√¥£¤ß´úª¾є™，ﬁõ  �►□′″¨³‑¯≈ˆ§‰●ﬂ⇑➘①②„≤±†✜✔➪✖◗¢ไทยếệεληνικαåşıруский 한국어汉语ž¹¿šćþ‚‛─÷〈¸⎯×←→∑δ■ʹ‐≥τ;∆℡ƒð¬¡¦βϕ▼⁄ρσ⋅≡∂≠π⎛⎜⎞ω∗"""

        num = 0
        for char in self.characters:
            self.vocab[char] = num
            num += 1

        if verbose:
            print("Vocabulary:", self.vocab)

        #raise NotImplementedError("Need to implement vocab initialization")

    def encode(self, text: str) -> torch.Tensor:

        text_lower = text.lower()

        tokens = []

        for char in text_lower:
            tokens.append(self.vocab[char])

        tokens_tsr = torch.tensor(tokens)

        return tokens_tsr

        #raise NotImplementedError(
        #    "Need to implement encoder that converts text to tensor of tokens."
        #)

    def decode(self, tokens: torch.Tensor) -> str:

        inverted_vocab = {value : key for key, value in self.vocab.items()}

        chars = ""

        for token in tokens:
            chars += inverted_vocab[token.item()]

        return chars

        #raise NotImplementedError(
        #    "Need to implement decoder that converts tensor of tokens to text."
        #)
