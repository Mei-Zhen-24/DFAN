import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import gumbel_softmax


class SeriesDecomp_Last(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        B, C, L = x.shape
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, pad_len)
        end = x[:, :, -1:].repeat(1, 1, pad_len)
        x_padded = torch.cat([front, x, end], dim=2)  # (B, C, L + 2*pad_len)
        moving_mean = self.avg(x_padded)
        res = x - moving_mean
        return res, moving_mean


class Fuzzy_Mahalanobis_mask(nn.Module):
    def __init__(self, input_size, k_ratio=0.5):
        super().__init__()
        frequency_size = input_size // 2 + 1
        self.k_ratio = k_ratio
        self.use_len = round(frequency_size * self.k_ratio)
        self.SeriesDecomp_Last = SeriesDecomp_Last(25)
        self.A = nn.Parameter(torch.randn(self.use_len, self.use_len), requires_grad=True)

    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))
        XF = XF[:, :, : self.use_len]
        XF_max, _ = torch.max(XF, dim=-1, keepdim=True)
        XF = XF / (XF_max.detach() + 1e-10)
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)
        # B x C x C x D
        diff = X1 - X2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)
        dist_n = dist / self.use_len / 2
        membership = torch.exp(-dist_n)
        p = membership * 0.99
        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / (r_flatten_matrix + 1e-10))
        log_r_flatten_matrix = torch.log(r_flatten_matrix / (flatten_matrix + 1e-10))
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        X_sea, X_trend = self.SeriesDecomp_Last(X)
        p = self.calculate_prob_distance(X_trend)
        sample = self.bernoulli_gumbel_rsample(p)
        return sample.unsqueeze(1)


class FuzzyEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, Moe=None, dropout=0.1):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.Moe = Moe

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, atten_ma = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        y = self.norm1(new_x + x)
        balance_loss_all = 0
        loss_moe = 0
        if self.Moe is not None:
            ym, loss_n = self.Moe(y)
            loss_moe += loss_n
            y = self.norm2(y + ym)
        # loss_moe = 0
        return y, balance_loss_all + loss_moe


class Channel_PatchData(nn.Module):
    def __init__(self, patch_len, padding):
        super(Channel_PatchData, self).__init__()
        self.patch_len = patch_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.permute_flag = False

    def do_patch(self, x):
        if self.permute_flag:
            x = x.permute(0, 2, 1)
        self.B, self.C, self.L = x.shape
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        if self.permute_flag:
            x = x.permute(0, 2, 1)
        return x

    def do_inv_patch(self, x):
        if self.permute_flag:
            x = x.permute(0, 2, 1)
        x = x.reshape(self.B, -1)
        x = x.reshape((self.B, self.C, -1))
        if self.permute_flag:
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x, mode=1):
        if mode == 1:
            return self.do_patch(x)
        else:
            return self.do_inv_patch(x)
