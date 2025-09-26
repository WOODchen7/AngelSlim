# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
from torch.special import gammaln


def absmean(x: torch.Tensor):
    scale = x.abs().mean([-1], keepdim=True)
    Delta = scale / 2
    return scale, Delta


def twn(x: torch.Tensor):
    Delta = 0.75 * x.abs().mean([-1], keepdim=True)
    I_pos = x >= Delta
    I_neg = x <= -Delta
    ternary_mask = torch.zeros_like(x)
    ternary_mask[I_pos] = 1
    ternary_mask[I_neg] = -1
    Scale = (x * ternary_mask).sum(dim=1, keepdim=True) / (
        ternary_mask.abs().sum(dim=1, keepdim=True)
    )
    return Scale, Delta


def ggd_quant(x):
    mu, alpha, beta = estimate_parameters(x)
    print(f"Estimated parameters: mu={mu:.3f}, alpha={alpha:.3f}, beta={beta:.3f}")
    if beta <= 1.5:
        Delta = (1.000 - 0.4312 * (beta - 1) + 0.0987 * (beta - 1) ** 2) * alpha
    else:
        Delta = (0.7071 + 0.1234 * (beta - 2) - 0.00456 * (beta - 2) ** 2) * alpha

    I_pos = x >= Delta
    I_neg = x <= -Delta
    Other = (x < Delta) & (x > -Delta)
    x_inter = torch.zeros_like(x)
    x_inter[I_pos] = 1
    x_inter[I_neg] = -1
    x_inter[Other] = 0
    I = x_inter != 0
    scale = x[I].abs().mean()
    return scale, Delta


def estimate_parameters(x):
    """Estimate GGD parameters using moment matching"""
    mu = 0
    m1 = torch.mean(torch.abs(x - mu))
    m2 = torch.mean((x - mu) ** 2)

    # estimator = RInverseApproximator()
    # beta_est = estimator.process((m1**2)/m2)
    def beta_estimator(x):
        return torch.log(-0.6365 / (x - 0.72934))

    beta_est = beta_estimator((m1**2) / m2)
    inv_beta = 1.0 / beta_est
    log_ratio = gammaln(inv_beta) - gammaln(3 * inv_beta)
    alpha_est = torch.sqrt(m2 * torch.exp(log_ratio))

    return mu, alpha_est, beta_est


def build_quantized_linear(in_feature, out_feature, bias, config):
    quant_method = config.quant_method
    kwargs = dict(
        quant_method=quant_method,
        granularity=config.granularity,
        group_size=config.group_size,
        enable_zero_point=config.enable_zero_point,
        range_of_lambada=config.range_of_lambada,
        eps=config.eps,
    )

    if config.w_bits == 0:
        if quant_method in [
            "ultraquant",
            "ultraquantv2",
            "ultraquantv3",
            "ultraquantv4",
        ]:
            return UltraQuantLinear(in_feature, out_feature, bias, **kwargs)
        else:
            raise NotImplementedError
    elif config.w_bits == 4:
        if quant_method in ["absmean", "absmax"]:
            raise NotImplementedError
    else:
        raise NotImplementedError


class StaticQuaternaryQuant(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, quant_method, granularity, group_size, enable_zero_point, eps
    ):

        original_shape = input.shape

        if granularity == "per_tensor":
            x = input.reshape(1, -1)  # [1, N]
        elif granularity == "per_channel":
            x = input.reshape(original_shape[0], -1)  # [C, N]
        elif granularity == "per_group":
            x = input.reshape(-1, group_size)  # [G, group_size]
        else:
            raise NotImplementedError

        alpha = x.abs().mean([-1], keepdim=True)
        delta = alpha / 2

        A = torch.zeros_like(x).to(x.device)
        mask_A_pos = x >= delta
        mask_A_neg = x <= -delta
        A[mask_A_pos] = 1
        A[mask_A_neg] = -1
        A = A * alpha

        B = torch.zeros_like(x).to(x.device)
        mask_B_pos = (x >= 0) & (A == 0)
        mask_B_neg = (x < 0) & (A == 0)
        B[mask_B_pos] = eps
        B[mask_B_neg] = -eps

        A, B = A.reshape(original_shape), B.reshape(original_shape)
        mask_B_pos, mask_B_neg = mask_B_pos.reshape(original_shape), mask_B_neg.reshape(
            original_shape
        )

        ctx.save_for_backward(mask_B_pos, mask_B_neg)
        return A, B

    def backward(ctx, grad_A, grad_B):
        mask_B_pos, mask_B_neg = ctx.saved_tensors
        grad_input = grad_A
        grad_input[mask_B_pos] += grad_B[mask_B_pos]
        grad_input[mask_B_neg] += grad_B[mask_B_neg]

        return grad_input, None, None, None, None, None


class UltraQuantV2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, quant_method, granularity, group_size, enable_zero_point, eps
    ):

        original_shape = input.shape

        if granularity == "per_tensor":
            x = input.reshape(1, -1)  # [1, N]
        elif granularity == "per_channel":
            x = input.reshape(original_shape[0], -1)  # [C, N]
        elif granularity == "per_group":
            x = input.reshape(-1, group_size)  # [G, group_size]
        else:
            raise NotImplementedError
        G_shape = x.shape
        scale, delta = absmean(x)

        A = torch.zeros_like(x).to(x.device)
        mask_A_pos = x >= delta
        mask_A_neg = x <= -delta
        A[mask_A_pos] = 1
        A[mask_A_neg] = -1
        A = A * scale

        B = torch.zeros_like(x).to(x.device)
        mask_B = A == 0
        B[mask_B] = eps * x[mask_B]

        A, B = A.reshape(original_shape), B.reshape(original_shape)

        ctx.save_for_backward(mask_B, eps, scale)
        ctx.other = G_shape
        return A, B

    def backward(ctx, grad_A, grad_B):
        mask_B, eps, scale = ctx.saved_tensors
        G_shape = mask_B.shape
        original_shape = grad_A.shape

        grad_A = grad_A.reshape(G_shape)
        grad_B = grad_B.reshape(G_shape)
        grad_output = grad_A
        grad_output[mask_B] = grad_output[mask_B] + eps * grad_B[mask_B]

        # grad_scale = torch.ones_like(grad_output) * scale
        # grad_scale[mask_B] = 1

        # grad_output = grad_output * grad_scale

        grad_output = grad_output.reshape(original_shape)
        return grad_output, None, None, None, None, None


class UltraQuantV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quant_method, granularity, group_size, enable_zero_point):

        original_shape = input.shape
        if granularity == "per_tensor":
            x = input.reshape(1, -1)  # [1, N]
        elif granularity == "per_channel":
            x = input.reshape(original_shape[0], -1)  # [C, N]
        elif granularity == "per_group":
            x = input.reshape(-1, group_size)  # [G, group_size]
        else:
            raise NotImplementedError
        scale, delta = absmean(x)
        A = torch.zeros_like(x).to(x.device)
        mask_A_pos = x >= delta
        mask_A_neg = x <= -delta
        A[mask_A_pos] = 1
        A[mask_A_neg] = -1
        A = A * scale

        B = torch.zeros_like(x).to(x.device)
        mask_B = A == 0
        B[mask_B] = x[mask_B]

        A, B = A.reshape(original_shape), B.reshape(original_shape)
        ctx.save_for_backward(mask_B, scale)
        # assert not torch.isnan(input).any()
        return A, B

    def backward(ctx, grad_A, grad_B):
        mask_B, scale = ctx.saved_tensors
        G_shape = mask_B.shape
        original_shape = grad_A.shape

        grad_A = grad_A.reshape(G_shape)
        grad_B = grad_B.reshape(G_shape)

        grad_output = grad_A
        grad_output[mask_B] = grad_B[mask_B]

        grad_scale = torch.ones_like(grad_output) * scale
        grad_scale[mask_B] = 1

        grad_output = grad_output * grad_scale
        grad_output = grad_output.reshape(original_shape)
        return grad_output, None, None, None, None


class UltraQuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        quant_method="ultraquant",
        granularity="per_group",
        group_size=128,
        enable_zero_point=False,
        range_of_lambada=0.01,
        eps=1e-5,
    ):
        super(UltraQuantLinear, self).__init__(in_features, out_features, bias=bias)
        self.quant_method = quant_method
        self.granularity = granularity
        self.group_size = group_size
        # params for weight quant
        self.enable_zero_point = enable_zero_point
        if self.quant_method in ["ultraquantv2", "ultraquant"]:
            self.eps = eps
        elif self.quant_method in ["ultraquantv3", "ultraquantv4"]:
            self.Lambada = nn.Parameter(
                torch.randn_like(self.weight) * range_of_lambada, requires_grad=True
            )
            self.optimizer = torch.optim.AdamW([self.Lambada], lr=1e-4)

    def update_lambada_v3(self, input, B):
        # Update lambada
        # reshape input for computation
        input = input.reshape(-1, input.shape[-1])  # [T, in_features]
        T = input.shape[0]  # token 数
        Y = nn.functional.linear(input, B)  # [token * out_features]
        s = torch.sum(self.Lambada * B, dim=-1)  # [out_features]
        loss = torch.sum((Y - s) ** 2) / T
        self.optimizer.zero_grad()
        loss.backward()

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        if self.quant_method == "ultraquant":
            eps = torch.tensor(self.eps, device=input_.device, dtype=input_.dtype)
            A, B = StaticQuaternaryQuant.apply(
                real_weights,
                self.quant_method,
                self.granularity,
                self.group_size,
                self.enable_zero_point,
                eps,
            )

            A = A.to(input_.dtype)
            B = B.to(input_.dtype)

            ones = torch.sign(input_.detach())
            out = nn.functional.linear(input_, A) + nn.functional.linear(ones, B)

            if self.bias is not None:
                out += self.bias.view(1, -1).expand_as(out)

            return out

        elif self.quant_method == "ultraquantv2":
            eps = torch.tensor(self.eps, device=input_.device, dtype=input_.dtype)
            A, B = UltraQuantV2.apply(
                real_weights,
                self.quant_method,
                self.granularity,
                self.group_size,
                self.enable_zero_point,
                eps,
            )
        elif self.quant_method in ["ultraquantv3"]:
            assert not torch.isnan(self.Lambada).any(), f"{self.Lambada}"
            A, B = UltraQuantV3.apply(
                real_weights,
                self.quant_method,
                self.granularity,
                self.group_size,
                self.enable_zero_point,
            )

            if self.training:
                self.update_lambada_v3(input_.detach(), B.detach())
            else:
                pass

            B = B * self.Lambada.detach()

        A = A.to(input_.dtype)
        B = B.to(input_.dtype)

        ones = torch.ones_like(input_, device=input_.device)
        out = nn.functional.linear(input_, A) + nn.functional.linear(ones, B)

        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
