import numpy as np


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d_model = K.shape[-1]
    return softmax(1.0 / np.sqrt(d_model) * Q @ np.swapaxes(K, -1, -2)) @ V


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    batch, seq, d_model = x.shape
    d_k = d_model // num_heads
    x = x.reshape(batch, seq, num_heads, d_k)
    return np.transpose(x, (0, 2, 1, 3))


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch, seq_q, d_model = Q.shape

    Qh = split_heads(Q @ W_q, num_heads)
    Kh = split_heads(K @ W_k, num_heads)
    Vh = split_heads(V @ W_v, num_heads)

    heads = scaled_dot_product_attention(Qh, Kh, Vh)  # (batch, num_heads, seq_q, d_k)

    concat = np.transpose(heads, (0, 2, 1, 3)).reshape(batch, seq_q, d_model)
    return concat @ W_o
