import argparse
from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    _score_mod_signature,
    BlockMask,
)

from attn_gym.masks.document_mask import (
    generate_doc_mask_mod,
    _offsets_to_doc_ids_tensor,
)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def all_valid_mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    """
    Mask mod that is valid everywhere i.e. doesn't mask anything.
    We're doing this just to make use of generate_doc_mask_mod's interface.
    """
    return q_idx >= 0


def generate_doc_score_mod(
    score_mod: _score_mod_signature, offsets: Tensor
) -> _score_mod_signature:
    """Generates score mods that apply to inputs to flex attention in the sequence
    stacked format.

    Inspired by generate_doc_mask_mod.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_score_mod(score, b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_score = score_mod(score, b, h, q_logical, kv_logical)
        return torch.where(same_doc, inner_score, float("-inf"))

    return doc_score_mod


def create_score_mod_relative(
    relative_pos_bias: Tensor, max_relative_pos: int
) -> _score_mod_signature:
    """
    Generates score mods that add a bias based on the relative positions of q and k.
    """

    def score_mod(
        score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor
    ) -> Tensor:
        pos_bucket = ((q_idx - kv_idx) + max_relative_pos).clip(
            min=0, max=2 * max_relative_pos - 1
        )
        return score + relative_pos_bias[h, pos_bucket]

    return score_mod


def block_mask_from_offsets(offsets: torch.Tensor) -> BlockMask:
    S = int(offsets[-1].item())
    mask_mod = generate_doc_mask_mod(all_valid_mask_mod, offsets=offsets)
    block_mask = create_block_mask(mask_mod=mask_mod, B=1, H=1, Q_LEN=S, KV_LEN=S)
    return block_mask


def test_offsets(offsets: torch.Tensor, do_warmup: bool = False):
    # define problem size
    B, H, S, D = 1, 1, int(offsets[-1].item()), 64

    # create block mask
    if do_warmup:
        _ = block_mask_from_offsets(offsets[:6])
    block_mask = block_mask_from_offsets(offsets)

    # create score mod
    max_relative_pos = 1024
    relative_pos_bias = torch.rand(H, 2 * max_relative_pos, requires_grad=True)
    score_mod = create_score_mod_relative(
        relative_pos_bias=relative_pos_bias, max_relative_pos=max_relative_pos
    )
    score_mod = generate_doc_score_mod(score_mod=score_mod, offsets=offsets)

    # run forward and backward pass
    q = torch.rand(B, H, S, D, requires_grad=True)
    k = torch.rand(B, H, S, D, requires_grad=True)
    v = torch.rand(B, H, S, D, requires_grad=True)
    grad_out = torch.rand(B, H, S, D)
    flex_out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
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
