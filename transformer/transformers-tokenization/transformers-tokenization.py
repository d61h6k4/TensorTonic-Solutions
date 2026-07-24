from collections.abc import Generator
from typing import Dict, List

import numpy as np


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def _sentence_to_words(self, text: str) -> Generator[str, None, None]:
        if not text:
            return None
            
        for word in text.strip().lower().split(' '):
            yield word

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.vocab_size = 4
        for text in texts:
            for word in self._sentence_to_words(text):
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.vocab_size += 1

        for ix, word in enumerate(
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
            + sorted(self.word_to_id.keys())
        ):
            self.id_to_word[ix] = word
            self.word_to_id[word] = ix

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        return [self.word_to_id[word] if word in self.word_to_id else self.word_to_id[self.unk_token] for word in self._sentence_to_words(text)]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        return ' '.join([self.id_to_word[ix] if ix in self.id_to_word else self.unk_token for ix in ids ])
