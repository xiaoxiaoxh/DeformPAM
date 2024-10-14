import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code
    

class RotaryPositionEncoding2D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary2D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XY):
        '''
        @param XY: [B,N,2]
        @return:
        '''
        bsize, npoint, _ = XY.shape
        x_position, y_position = XY[..., 0:1], XY[..., 1:2]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 2, 2, dtype=torch.float, device=XY.device)
            * (-math.log(10000.0) / (self.feature_dim // 2))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//4]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//4]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)

        sinx, cosx, siny, cosy = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy], dim=-1),  # cos_pos
            torch.cat([sinx, siny], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding6D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary6D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZUVW):
        '''
        @param XY: [B,N,6]
        @return:
        '''
        bsize, npoint, _ = XYZUVW.shape
        x_position, y_position, z_position, u_position, v_position, w_position = XYZUVW.trunk(6, dim=-1)
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 6, 2, dtype=torch.float, device=XYZUVW.device)
            * (-math.log(10000.0) / (self.feature_dim // 6))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//12]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//12]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)
        sinu = torch.sin(u_position * div_term)
        cosu = torch.cos(u_position * div_term)
        sinv = torch.sin(v_position * div_term)
        cosv = torch.cos(v_position * div_term)
        sinw = torch.sin(w_position * div_term)
        cosw = torch.cos(w_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz, sinu, cosu, sinv, cosv, sinw, cosw = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz, sinu, cosu, sinv, cosv, sinw, cosw]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz, cosu, cosv, cosw], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz, sinu, sinv, sinw], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code
    

class LearnedAbsolutePositionEncoding3D(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)


class LearnedAbsolutePositionEncoding3Dv2(nn.Module):
    def __init__(self, input_dim, embedding_dim, norm="none"):
        super().__init__()
        norm_tb = {
            "none": nn.Identity(),
            "bn": nn.BatchNorm1d(embedding_dim),
        }
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            norm_tb[norm],
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)
