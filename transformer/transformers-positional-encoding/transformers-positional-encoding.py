import numpy as np


def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    base = 10_000
    positions = np.arange(seq_length)[:, None]
    i = np.arange(0, d_model, 2)
    div_term = np.power(base, i / d_model)
    angles = positions / div_term

    pos_enc = np.zeros((seq_length, d_model), dtype=np.float32)
    pos_enc[:, 0::2] = np.sin(angles)
    pos_enc[:, 1::2] = np.cos(angles)
    return pos_enc
