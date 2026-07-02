import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    
    new_G = np.asarray(G) + np.pow(g, 2)
    new_w = np.asarray(w) - lr / np.sqrt(new_G + eps) * np.asarray(g)
    return new_w, new_G
    