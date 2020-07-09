import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchsummary
import numpy as np
import os
import torch.distributions as td
import math
from extern.normalization import CategoricalConditionalBatchNorm


def get_same_padding(size, kernel_size, stride):
    def check(clen, cpad):
        return (clen + 2 * cpad - (kernel_size - 1) - 1) // stride + 1

    h_pad = 0
    while check(size[0], h_pad) != size[0] / stride:
        h_pad += 1

    w_pad = 0
    while check(size[1], w_pad) != size[1] / stride:
        w_pad += 1

    return h_pad, w_pad


def get_same_padding_transpose(size, kernel_size, stride):
    def check(clen, cpad):
        return (clen - 1) * stride - 2 * cpad + (kernel_size - 1) + 1

    h_pad = 0
    while check(size[0], h_pad) > size[0] * stride:
        h_pad += 1

    w_pad = 0
    while check(size[1], w_pad) > size[1] * stride:
        w_pad += 1

    return (h_pad, w_pad), size[0] * stride - check(size[0], h_pad)


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(start_dim=1)


class VisualEncoderCell(nn.Module):
    def __init__(self, size, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = get_same_padding(size, kernel_size, stride)
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride)

        # Move to CONDITIONAL INSTANCE NORM 2D
        self.instance_norm = CategoricalConditionalBatchNorm(out_channels, 62)
        self.relu = nn.ReLU()

    def forward(self, input, labels):
        conv2d = self.conv2d(input)
        instance_norm = self.instance_norm(conv2d, labels)
        return self.relu(instance_norm)


class VisualEncoderModel(nn.Module):
    # goes from [batch, 64, 64, 1] to [batch, hidden_size]
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.block1 = VisualEncoderCell((64, 64), 1, config.base_depth, 5, 1)
        self.block2 = VisualEncoderCell((64, 64), config.base_depth, config.base_depth, 5, 2)
        self.block3 = VisualEncoderCell((32, 32), config.base_depth, 2 * config.base_depth, 5, 1)
        self.block4 = VisualEncoderCell((32, 32), 2 * config.base_depth, 2 * config.base_depth, 5, 2)

        # new conv layer, to bring shape down
        self.block5 = VisualEncoderCell((16, 16), 2 * config.base_depth, 2 * config.bottleneck_bits, 4, 2)
        # new conv layer, to bring shape down
        self.block6 = VisualEncoderCell((8, 8), 2 * config.bottleneck_bits, 2 * config.bottleneck_bits, 4, 2)

        self.flatten = Flatten()
        self.dense = nn.Linear(1024, 2 * config.bottleneck_bits)

    def forward(self, input, labels):
        out = self.block1(input, labels)
        out = self.block2(out, labels)
        out = self.block3(out, labels)
        out = self.block4(out, labels)
        out = self.block5(out, labels)
        out = self.block6(out, labels)
        flatten = self.flatten(out)
        #print(flatten.shape)
        return self.dense(flatten)


class VisualDecoderCell(nn.Module):
    def __init__(self, size, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding, output_padding = get_same_padding_transpose(size, kernel_size, stride)
        self.conv2d = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         output_padding=output_padding,
                                         stride=stride)

        self.instance_norm = CategoricalConditionalBatchNorm(out_channels, 62)
        self.relu = nn.ReLU()

    def forward(self, input, labels):
        conv2d = self.conv2d(input)
        #print(conv2d.shape)
        instance_norm = self.instance_norm(conv2d, labels)
        return self.relu(instance_norm)


class VisualDecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dense = nn.Linear(config.bottleneck_bits, 1024)
        self.block1 = VisualDecoderCell((4, 4), 64, 2 * config.base_depth, 4, 2)
        self.block2 = VisualDecoderCell((8, 8), 2 * config.base_depth, 2 * config.base_depth, 4, 2)
        self.block3 = VisualDecoderCell((16, 16), 2 * config.base_depth, 2 * config.base_depth, 5, 1)
        self.block4 = VisualDecoderCell((16, 16), 2 * config.base_depth, 2 * config.base_depth, 5, 2)
        self.block5 = VisualDecoderCell((32, 32), 2 * config.base_depth, config.base_depth, 5, 1)
        self.block6 = VisualDecoderCell((32, 32), config.base_depth, config.base_depth, 5, 2)
        self.block7 = VisualDecoderCell((64, 64), config.base_depth, config.base_depth, 5, 1)

        self.conv2d = nn.Conv2d(in_channels=config.base_depth,
                                out_channels=1,
                                kernel_size=5,
                                padding=2)

    def forward(self, input, labels):
        out = self.dense(input)
        out = out.reshape(-1, 64, 4, 4)
        #print(200, out.shape)
        out = self.block1(out, labels)
        out = self.block2(out, labels)
        out = self.block3(out, labels)
        out = self.block4(out, labels)
        out = self.block5(out, labels)
        out = self.block6(out, labels)
        out = self.block7(out, labels)
        out = self.conv2d(out)
        #print("HI", out.shape)
        return out


class VAEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = VisualEncoderModel(config)
        self.decoder = VisualDecoderModel(config)

    def loss(self, logits, features):
        # logits should be dict with 'outputs', which is image.
        targets = features.reshape(-1, 1, 64, 64)
        weights = torch.ones_like(targets)
        loss_num = torch.pow(logits - targets, 2)
        return torch.sum(loss_num * weights), torch.sum(weights)

    def forward(self, inputs, labels, bottleneck=None):
        targets = inputs

        enc_out = self.encoder(inputs, labels)
        enc_out = enc_out.reshape(-1, 2 * self.config.bottleneck_bits)

        # bottleneck
        sampled_bottleneck, b_loss = self.bottleneck(enc_out)
        losses = {'bottleneck_kl': torch.mean(b_loss)}

        if not (bottleneck is None):
            if bottleneck.shape[0] == 0:
                # return bottleneck for interpolation
                # set losses['training'] = 0 so top() isn't called on it
                # potential todo: use losses dict so we have kl_loss here for non stop
                # gradient models
                return sampled_bottleneck, {'training': 0.0}
            else:
                # we want to use the given bottleneck
                sampled_bottleneck = bottleneck

        # finalize bottleneck
        unbottleneck = sampled_bottleneck

        # decoder.
        dec_out = self.decoder(unbottleneck, labels)
        dec_out = td.independent.Independent(td.bernoulli.Bernoulli(dec_out), 3)

        #print(inputs.shape)

        # calculate training loss here lol
        rec_loss = -dec_out.log_prob(inputs)
        elbo = torch.mean(-(b_loss + rec_loss))
        losses['rec_loss'] = torch.mean(rec_loss)
        losses['training'] = -elbo

        #print(dec_out.mean)

        return dec_out.mean, losses

    def bottleneck(self, x):
        z_size = self.config.bottleneck_bits
        x_shape = x.shape
        mu = x[..., :self.config.bottleneck_bits]
        if not self.training:
            return mu, 0.0  # No sampling or kl loss on eval.

        log_sigma = x[..., self.config.bottleneck_bits:]
        #print(x_shape[:-1], [z_size])
        epsilon = torch.randn(list(x_shape[:-1]) + [z_size]).cuda()
        z = (mu + torch.exp(log_sigma / 2) * epsilon).cuda()
        kl = (0.5 * torch.mean(torch.exp(log_sigma) + torch.square(mu) - 1. - log_sigma)).cuda()
        # This is the 'free bits' trick mentioned in Kingma et al. (2016)
        free_bits = self.config.free_bits
        kl_loss = torch.mean(torch.clamp_min(kl - free_bits, 0.0))
        return z, kl_loss * self.config.kl_beta
