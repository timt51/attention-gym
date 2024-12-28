import argparse
from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    _mask_mod_signature,
    BlockMask,
)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def generate_doc_mask_mod(offsets: Tensor) -> _mask_mod_signature:
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    return doc_mask_mod


def block_mask_from_offsets(offsets: torch.Tensor) -> BlockMask:
    S = int(offsets[-1].item())
    mask_mod = generate_doc_mask_mod(offsets)
    block_mask = create_block_mask(mask_mod=mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S)
    return block_mask


def test_offsets(offsets: torch.Tensor, do_warmup: bool = False):
    # define problem size
    B, H, S, D = 1, 1, int(offsets[-1].item()), 64

    # create block mask
    if do_warmup:
        _ = block_mask_from_offsets(offsets[:6])
    block_mask = block_mask_from_offsets(offsets)

    # run forward and backward pass
    q = torch.rand(B, H, S, D, requires_grad=True)
    k = torch.rand(B, H, S, D, requires_grad=True)
    v = torch.rand(B, H, S, D, requires_grad=True)
    grad_out = torch.rand(B, H, S, D)
    flex_out = flex_attention(q, k, v, score_mod=None, block_mask=block_mask)
    flex_out = cast(torch.Tensor, flex_out)
    flex_out.backward(grad_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_warmup", action="store_true")
    args = parser.parse_args()

    test_offsets(
        offsets=torch.tensor([0, 125, 273, 365, 544, 712, 1161, 1311, 1478]),
        do_warmup=args.do_warmup,
    )


if __name__ == "__main__":
    main()
