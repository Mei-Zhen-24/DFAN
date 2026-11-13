import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .linear_pattern_extractor import Fuzzy_extractor as expert
from ..layers.RevIN import RevIN


class MyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        with torch.no_grad():
            nn.init.constant_(self.weight, 1.0 / in_features)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class FuzzyGate(nn.Module):
    def __init__(self, input_size, output_size, rule_size=2):
        super().__init__()
        feature_size = input_size
        self.experts = nn.ModuleList([MyLinear(input_size, output_size) for _ in range(rule_size)])
        self.scale = 1 / feature_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, x, mu, sigma):
        ant = x
        Q_norm_sq = (ant**2).sum(dim=-1, keepdim=True)
        K_norm_sq = (mu**2).sum(dim=-1).unsqueeze(0)
        scores_inner = torch.einsum("bf,hf->bh", ant, mu)
        scores = Q_norm_sq + K_norm_sq - 2 * scores_inner
        scores = (scores / (sigma**2 + 1e-10)).clamp(min=0)
        raw_scores = -self.scale * scores
        A = torch.softmax(raw_scores, dim=-1)
        all_expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        V = torch.einsum("bh,bho->bo", A, all_expert_outs)
        return V


class FuzzyNoise(nn.Module):
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
        return V


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Linear_extractor_cluster(nn.Module):
    def __init__(self, config):
        super(Linear_extractor_cluster, self).__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.input_size = config.seq_len
        self.k = config.k
        # instantiate experts
        self.experts = nn.ModuleList([expert(config) for _ in range(self.num_experts)])
        self.W_h = nn.Parameter(torch.eye(self.num_experts))
        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.gate_fuzzy = FuzzyGate(config.seq_len, self.num_experts, self.num_experts)
        self.noise_fuzzy = FuzzyNoise(config.seq_len, self.num_experts, self.num_experts)
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def get_mu_sigma(self):
        with torch.no_grad():
            mu_list = []
            sigma_list = []
            for i in range(self.num_experts):
                # get mu
                s_mu = self.experts[i].Linear_Seasonal.mu.mean(dim=0).unsqueeze(0)
                s_sigma = self.experts[i].Linear_Seasonal.sigma.mean(dim=0, keepdim=True).unsqueeze(0)
                l_mu = self.experts[i].Linear_Trend.mu.mean(dim=0).unsqueeze(0)
                l_sigma = self.experts[i].Linear_Trend.sigma.mean(dim=0, keepdim=True).unsqueeze(0)
                mu_list.append((s_mu + l_mu))
                sigma_list.append((s_sigma + l_sigma) / 2)
            mu = torch.cat(mu_list, dim=0).squeeze(1)
            sigma = torch.cat(sigma_list, dim=0).squeeze(1)
        return mu, sigma

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        mu, sigma = self.get_mu_sigma()
        clean_logits = self.gate_fuzzy(x.squeeze(-1), mu, sigma)
        # clean_logits = self.gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise_fuzzy(x.squeeze(-1))
            # raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i])
            for i in range(self.num_experts)
            if expert_inputs[i] is not None and expert_inputs[i].numel() > 0
        ]
        y = dispatcher.combine(expert_outputs)
        return y, loss
