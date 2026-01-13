# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Modified from ParetoQ，https://arxiv.org/abs/2502.02631

import math

import torch
import torch.nn as nn
from torch.special import gammaln
import numpy as np

def absmean(x: torch.Tensor):
    scale = x.abs().mean([-1], keepdim=True)
    Delta = scale / 2
    return scale, Delta

def twn(x: torch.Tensor):
    Delta = 0.75 * x.abs().mean([-1], keepdim=True)
    I_pos = (x >= Delta)
    I_neg = (x <= -Delta)
    ternary_mask = torch.zeros_like(x)
    ternary_mask[I_pos] = 1
    ternary_mask[I_neg] = -1
    Scale = (x * ternary_mask).sum(dim=1, keepdim=True) / (ternary_mask.abs().sum(dim=1, keepdim=True))
    return Scale, Delta

def build_quantized_linear(in_feature, out_feature, bias, config):
    quant_method = config.quant_method
    kwargs = dict(
        quant_method=quant_method,
        granularity=config.granularity,
        group_size=config.group_size,
        enable_zero_point=config.enable_zero_point,
        range_of_lambada=config.range_of_lambada,
        eps=config.eps,
        N=config.N,
        M=config.M,
    )
    if quant_method in ["lsq", "seq", "dlt"]:
        return LSQLinear(in_feature, out_feature, w_bits=config.w_bits, quant_method=quant_method)
    elif config.w_bits == 1:
        return BiLinear(in_feature, out_feature, bias, **kwargs)
    elif config.w_bits == 0:
        if quant_method in ["absmean", "twn"]:
            return StaticTernaryLinear(in_feature, out_feature, bias, **kwargs)
        elif quant_method in ["sherry",]:
            return Arenas(in_feature, out_feature, bias, **kwargs)
        else:
            raise NotImplementedError
    elif config.w_bits == 4:
        if quant_method in ["absmean", "absmax"]:
            raise NotImplementedError
    elif config.w_bits == 16 or quant_method in ["None", "none"]:
        return nn.Linear(in_feature, out_feature, bias)
    else:
        raise NotImplementedError


class BiQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quant_method, granularity, group_size, enable_zero_point):

        original_shape = input.shape

        if granularity == 'per_tensor':
            x = input.reshape(1, -1)  # [1, N]
        elif granularity == 'per_channel':
            x = input.reshape(original_shape[0], -1)  # [C, N]
        elif granularity == 'per_group':
            x = input.reshape(-1, group_size) # [G, group_size]
        else:
            raise NotImplementedError

        scale = x.abs().mean([-1], keepdim=True)
        x_inter = torch.sign(x)
        dequantized_x = x_inter * scale
        output = dequantized_x.reshape(original_shape)

        return output
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class NMQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quant_method, granularity, group_size, enable_zero_point, N, M):

        original_shape = input.shape
        assert len(original_shape) == 2

        weight_reshaped = input.reshape(original_shape[0], original_shape[1] // M, M)
        _, topk_indices = torch.topk(torch.abs(weight_reshaped), N, dim=-1)
        mask = torch.zeros_like(weight_reshaped, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        sparse_weight = weight_reshaped * mask
        sparse_weight = sparse_weight.reshape(original_shape[0], original_shape[1])
        mask = mask.reshape(original_shape[0], original_shape[1])

        if granularity == 'per_tensor':
            x = sparse_weight.reshape(1, -1)  # [1, N]
            mask = mask.reshape(1, -1)
        elif granularity == 'per_channel':
            x = sparse_weight.reshape(original_shape[0], -1)  # [C, N]
            mask = mask.reshape(original_shape[0], -1)
        elif granularity == 'per_group':
            x = sparse_weight.reshape(-1, group_size) # [G, group_size]
            mask = mask.reshape(-1, group_size)
        else:
            raise NotImplementedError

        x_inter = torch.sign(x)
        scale = x.abs().sum([-1], keepdim=True) / (x.shape[-1] / M * N)
        dequantized_x = x_inter * scale
        output = dequantized_x.reshape(original_shape)
        return output

    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None

class StaticTernaryQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quant_method, granularity, group_size, enable_zero_point):

        original_shape = input.shape

        if granularity == 'per_tensor':
            x = input.reshape(1, -1)  # [1, N]
        elif granularity == 'per_channel':
            x = input.reshape(original_shape[0], -1)  # [C, N]
        elif granularity == 'per_group':
            x = input.reshape(-1, group_size) # [G, group_size]
        else:
            raise NotImplementedError

        if quant_method == "absmean":
            scale, delta = absmean(x)
        elif quant_method == "twn":
            scale, delta = twn(x)
        else:
            raise NotImplementedError

        I_pos = (x >= delta)
        I_neg = (x <= -delta)
        Other = (x < delta) & (x > -delta)
        x_inter = torch.zeros_like(x)
        x_inter[I_pos] = 1
        x_inter[I_neg] = -1
        x_inter[Other] = 0
        dequantized_x = x_inter * scale

        output = dequantized_x.reshape(original_shape)
        ctx.save_for_backward(Other, scale)
        return output
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class DLTQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, gamma, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=input.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha + gamma
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        grad_gamma = torch.sum(grad_output, dim=-1, keepdim=True)
        return grad_input, grad_alpha, grad_gamma, None, None


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=input.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=input.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class LSQLinear(nn.Linear):
    def __init__(
        self,
        *args,
        symmetric=True,
        bias=False,
        w_bits=16,
        quant_method = "lsq",
        weight_layerwise=False,

    ):
        super(LSQLinear, self).__init__(*args, bias=False)
        self.w_bits = w_bits
        self.quant_method = quant_method
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
        if quant_method == "dlt":
            self.gamma = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 16:
            weight = self.weight

        if self.quant_method == "lsq":
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        elif self.quant_method == "seq":
            weight = StretchedElasticQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        elif self.quant_method == "dlt":
            weight = DLTQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.gamma,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        else:
            raise NotImplementedError

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BiLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        quant_method="absmean",
        granularity="per_group",
        group_size=128,
        enable_zero_point=False,
        **kwargs
    ):
        super(BiLinear, self).__init__(in_features, out_features, bias=bias)
        self.quant_method = quant_method
        self.granularity = granularity
        self.group_size = group_size
        # params for weight quant
        self.enable_zero_point = enable_zero_point

    def forward(self, input_):
        assert len(self.weight.size()) == 2

        weight = BiQuant.apply(
            self.weight,
            self.quant_method,
            self.granularity,
            self.group_size,
            self.enable_zero_point
        ).to(input_.dtype)

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class Arenas(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        quant_method="absmean",
        granularity="per_group",
        group_size=128,
        enable_zero_point=False,
        N = 3,
        M = 4,
        **kwargs
    ):
        super(Arenas, self).__init__(in_features, out_features, bias=bias)
        self.quant_method = quant_method
        self.granularity = granularity
        self.group_size = group_size
        # params for weight quant
        self.enable_zero_point = enable_zero_point
        self.N = N
        self.M = M

        self.eps = 0
        self.steps = 0
        self.warmup_step = 22111//10
        self.total_steps = 22111
        self.max_eps = 1

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        weight = NMQuant.apply(
                self.weight,
                self.quant_method,
                self.granularity,
                self.group_size,
                self.enable_zero_point,
                self.N,
                self.M
            ).to(input_.dtype)

        self.steps += 1
        if self.training:
            if self.steps <= self.warmup_step:
                self.eps = self.max_eps * self.steps / self.warmup_step
            elif self.steps < self.total_steps:
                progress = (self.steps - self.warmup_step) / (self.total_steps - self.warmup_step)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                self.eps = self.max_eps * cosine_decay
            elif self.step >= self.total_steps:
                self.eps = 0
        else:
            self.eps = 0

        out = nn.functional.linear(input_, weight) + self.eps * nn.functional.linear(input_, self.weight)

        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class StaticTernaryLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        quant_method="absmean",
        granularity="per_group",
        group_size=128,
        enable_zero_point=False,
        **kwargs
    ):
        super(StaticTernaryLinear, self).__init__(in_features, out_features, bias=bias)
        self.quant_method = quant_method
        self.granularity = granularity
        self.group_size = group_size
        # params for weight quant
        self.enable_zero_point = enable_zero_point

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        weight = StaticTernaryQuant.apply(
                self.weight,
                self.quant_method,
                self.granularity,
                self.group_size,
                self.enable_zero_point
        ).to(input_.dtype)

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out