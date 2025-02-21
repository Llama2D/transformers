import torch
from typing import Any, Optional, Tuple

import numpy as np
import torch.nn as nn

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(
            self,
            num_pos_feats: int = 64,
            scale: Optional[float] = None,
            pin_lbd:bool=False,
            torch_dtype=torch.float32
        ) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        
        matrix = scale * torch.randn((2, num_pos_feats)).to(torch_dtype)
        matrix.requires_grad = False
        # register buffer
        self.register_buffer("positional_encoding_gaussian_matrix", matrix)

        # 0 is for not a point, 1 is for a point
        self.is_a_point_embed = nn.Embedding(2, num_pos_feats*2).to(torch_dtype)

        self.num_pos_feats = num_pos_feats

        # I use a 1d lambda because otherwise torch.empty(*param.size()) fails
        # self.lbd = nn.Parameter(torch.tensor([0.0],requires_grad=True).to(torch_dtype))

        self.pin_lbd = pin_lbd

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        if self.positional_encoding_gaussian_matrix.device != coords.device:
            print("converting pos encodings to device",coords.device,coords.dtype,self.positional_encoding_gaussian_matrix.dtype)
            self.positional_encoding_gaussian_matrix = self.positional_encoding_gaussian_matrix.to(coords.device).to(coords.dtype)

        if self.positional_encoding_gaussian_matrix.dtype != coords.dtype:
            # print("Dtype mismatch",coords.dtype,self.positional_encoding_gaussian_matrix.dtype)
            coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # B x N x C

    def get_rotary_2d_pos_embeds(self,coords):
        assert len(coords.shape) == 3,f"The shape of coords should have dims (batch_size,seq_len,2)"

        # some coords will be [-1,-1] because they have no known position
        # we should not add these coords to the positional embedding
        is_a_point = coords[:,:,0] != -1

        pos_embeds = self.forward_with_coords(coords,(1,1))
        # assert pos_embeds.shape == (bs,seq_len,dim),f"Shape of pos_embeds is {pos_embeds.shape} - shape of coords is {coords.shape} - intended shape is {(bs,seq_len,dim)}"

        is_a_point_embeds = self.is_a_point_embed(is_a_point.long())
        # assert is_a_point_embeds.shape == (bs,seq_len,dim),f"Shape of is_a_point_embeds is {is_a_point_embeds.shape} - shape of coords is {coords.shape}"

        delta = pos_embeds.unsqueeze(1) + is_a_point_embeds.unsqueeze(1)

        return delta

    @staticmethod
    def apply_rotary_2d_pos_emb(q,k,pos_embeds,lbd):

        # shape of q and k: [bs, num_heads, seq_len, dim]
        # aka: B x num_heads x N x C
        # shape of coords: [bs, seq_len, 2]
        bs,num_heads,seq_len,dim = q.shape
        # pos_embeds = pos_embeds.unsqueeze(1)
        # assert dim//2 == self.num_pos_feats,f"Dim of q is {dim}, not {self.num_pos_feats}"

        # add the positional embedding to the query and key
        q = q + pos_embeds * lbd
        k = k + pos_embeds * lbd

        return q,k