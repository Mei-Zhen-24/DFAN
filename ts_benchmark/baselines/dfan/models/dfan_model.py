from ts_benchmark.baselines.dfan.layers.linear_extractor_cluster import Linear_extractor_cluster
import torch.nn as nn
from einops import rearrange
from ts_benchmark.baselines.dfan.utils.masked_attention import (
    Encoder,
    AttentionLayer,
    FuzzyAttention_S,
)
import torch.nn.functional as F
import torch
from ..layers.RevIN import RevIN
from ..layers.DFANLayers import (
    Fuzzy_Mahalanobis_mask,
    FuzzyEncoderLayer,
    Channel_PatchData,
)


class DFANModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)
        self.channel_mask_generator = Fuzzy_Mahalanobis_mask(config.seq_len, config.k_ratio)
        self.patch_mask_generator = Fuzzy_Mahalanobis_mask(config.patch_len, config.k_ratio)
        self.patch_len = config.patch_len
        self.G = config.seq_len // self.patch_len
        self.patch = Channel_PatchData(self.patch_len, 0)
        self.channel_attention = Encoder(
            [
                FuzzyEncoderLayer(
                    attention=AttentionLayer(
                        FuzzyAttention_S(mask_flag=True, attention_dropout=config.dropout, n_rule=self.n_vars),
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                    ),
                    Moe=None,
                    d_model=config.d_model,
                )
                for _ in range(config.e_layers)
            ],
        )
        self.patch_attention = Encoder(
            [
                FuzzyEncoderLayer(
                    attention=AttentionLayer(
                        FuzzyAttention_S(mask_flag=True, attention_dropout=config.dropout, n_rule=self.G * self.n_vars),
                        d_model=config.patch_len,
                        n_heads=config.n_heads,
                    ),
                    Moe=None,
                    d_model=config.patch_len,
                )
                for _ in range(config.e_layers)
            ],
        )
        self.norm = nn.RMSNorm(config.d_model)
        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout))
        self.weight = nn.Parameter(torch.ones(2))
        self.cluster = Linear_extractor_cluster(config)

    def forward(self, input, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # 32, 7, 512
        input = self.revin(input, "norm")

        # Channel_attention
        channel_input = rearrange(input, "b l c -> b c l")
        channel_mask = self.channel_mask_generator(channel_input)
        channel_attention, channel_att = self.channel_attention(x=channel_input, attn_mask=channel_mask)

        # Patch_attention
        patch_input = rearrange(input, "b l c -> b c l")
        patch_input = self.patch(patch_input, mode=1)
        patch_mask = self.patch_mask_generator(patch_input)
        patch_attention, patch_att = self.patch_attention(x=patch_input, attn_mask=patch_mask)
        patch_attention = self.patch(patch_attention, mode=0)

        # fusion
        weight = F.softmax(self.weight, dim=-1)
        fusion_attention = self.norm(weight[0] * channel_attention + weight[1] * patch_attention)

        # fuzzymoe
        cluster_input = rearrange(fusion_attention, "b c l -> (b c) l 1")
        cluster_output, L_importance = self.cluster(cluster_input)
        cluster_output = rearrange(cluster_output, "(b c) l 1 -> b c l", b=input.shape[0])

        # linear_head
        output = self.linear_head(cluster_output)
        output = rearrange(output, "b c d -> b d c")

        output = self.revin(output, "denorm")
        return output, L_importance
