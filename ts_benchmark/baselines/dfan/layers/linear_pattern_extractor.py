import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time data
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time data
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SimpleFuzzy(nn.Module):
    def __init__(self, input_size, output_size, rule_size=2):
        super().__init__()
        feature_size = input_size
        self.mu = nn.Parameter(torch.empty(rule_size, feature_size))
        nn.init.xavier_normal_(self.mu)
        self.sigma = nn.Parameter(torch.ones(rule_size) * 1)
        self.experts = nn.ModuleList([MyLinear(input_size, output_size) for _ in range(rule_size)])
        self.scale = 1 / feature_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B * C, L)
        ant = x
        mu = self.mu
        Q_norm_sq = (ant**2).sum(dim=-1, keepdim=True)
        K_norm_sq = (mu**2).sum(dim=-1).unsqueeze(0)
        scores_inner = torch.einsum("bf,hf->bh", ant, mu)
        scores = Q_norm_sq + K_norm_sq - 2 * scores_inner
        scores = (scores / (self.sigma**2 + 1e-10)).clamp(min=0)
        raw_scores = -self.scale * scores
        A = torch.softmax(raw_scores, dim=-1)
        all_expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        V = torch.einsum("bh,bho->bo", A, all_expert_outs)
        V = V.reshape(B, C, self.output_size)
        return V


def cv_squared(x):
    eps = 1e-10
    # if only num_experts = 1

    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


class Fuzzy_extractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.d_model
        self.norm2 = nn.RMSNorm(config.seq_len)
        self.decompsition = series_decomp(25)
        self.Linear_Seasonal = SimpleFuzzy(config.seq_len, config.d_model, config.rule_size)
        self.Linear_Trend = SimpleFuzzy(config.seq_len, config.d_model, config.rule_size)

    def forward(self, x):
        # x: [B, L, D]
        norm_x = x.transpose(1, 2)
        seasonal_init, trend_init = self.decompsition(norm_x.transpose(1, 2))
        seasonal_out = self.Linear_Seasonal(seasonal_init.transpose(1, 2))
        trend_out = self.Linear_Trend(trend_init.transpose(1, 2))
        y = self.norm2((trend_out + seasonal_out + norm_x)).transpose(1, 2)
        return y


class MyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        with torch.no_grad():
            nn.init.constant_(self.weight, 1.0 / in_features)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
