import numpy as np

def hybrid_vit_multitask(image, patch_conv_w, patch_conv_b, pos_emb,
                         attn_Wq, attn_Wk, attn_Wv, attn_Wo,
                         num_heads, task_heads):
    image = image.astype(np.float32)
    D, kH, kW, C = patch_conv_w.shape
    H, W, _ = image.shape
    Hp, Wp = H // kH, W // kW

    # Patchify: (H, W, C) -> (N, kH*kW*C), raster order over the patch grid
    patches = (image[:Hp * kH, :Wp * kW]
               .reshape(Hp, kH, Wp, kW, C)
               .transpose(0, 2, 1, 3, 4)
               .reshape(Hp * Wp, kH * kW * C))

    # Weight already in (kH, kW, C) order, so it matches patches directly
    tokens = patches @ patch_conv_w.reshape(D, -1).T + patch_conv_b + pos_emb  # (N, D)

    assert D % num_heads == 0, "D must be divisible by num_heads"
    hd = D // num_heads
    split = lambda x: x.reshape(-1, num_heads, hd).transpose(1, 0, 2)  # (H, N, hd)
    q, k, v = split(tokens @ attn_Wq), split(tokens @ attn_Wk), split(tokens @ attn_Wv)

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)               # (H, N, N)
    scores -= scores.max(axis=-1, keepdims=True)                  # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)

    attn = (weights @ v).transpose(1, 0, 2).reshape(-1, D)        # (N, D)
    out = attn @ attn_Wo + tokens

    return {name: out @ W + b for name, (W, b) in task_heads.items()}