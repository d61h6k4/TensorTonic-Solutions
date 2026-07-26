import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch, seq_q, d_model = Q.shape
    d_k = d_model // num_heads

    def split_heads(t, W):
        t = t @ W
        b, s, _ = t.shape
        return np.transpose(t.reshape(b, s, num_heads, d_k), (0, 2, 1, 3))

    Qh = split_heads(Q, W_q)
    Kh = split_heads(K, W_k)
    Vh = split_heads(V, W_v)

    scores = Qh @ np.swapaxes(Kh, -1, -2) / np.sqrt(d_k)
    heads = softmax(scores, axis=-1) @ Vh

    concat = np.transpose(heads, (0, 2, 1, 3)).reshape(batch, seq_q, d_model)
    return concat @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.maximum(0, x @ W1 + b1)
    return hidden @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)

    ff_out = feed_forward(x, W1, b1, W2, b2)
    return layer_norm(x + ff_out, gamma2, beta2)
