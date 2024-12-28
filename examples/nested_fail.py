import argparse
from typing import cast

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_first_block_mask", action="store_true")
    args = parser.parse_args()

    # define problem size
    B, H, D = 1, 1, 64
    for i, S in enumerate([712, 1478]):
        # create block mask
        block_mask = create_block_mask(mask_mod=mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S)
        if i == 0 and args.skip_first_block_mask:
            continue

        # run forward and backward pass
        q = torch.rand(B, H, S, D, requires_grad=True)
        k = torch.rand(B, H, S, D, requires_grad=True)
        v = torch.rand(B, H, S, D, requires_grad=True)
        grad_out = torch.rand(B, H, S, D)
        flex_out = flex_attention(q, k, v, score_mod=None, block_mask=block_mask)
        flex_out = cast(torch.Tensor, flex_out)
        flex_out.backward(grad_out)


if __name__ == "__main__":
    main()
