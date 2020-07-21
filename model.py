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
from helpers import get_same_padding, get_same_padding_transpose


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(start_dim=1)


########################################################################################################################

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
        self.instance_norm = nn.InstanceNorm2d(out_channels) #CategoricalConditionalBatchNorm(out_channels, 62)
        self.relu = nn.ReLU()

    def forward(self, input, labels):
        conv2d = self.conv2d(input)
        instance_norm = self.instance_norm(conv2d)
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

        self.instance_norm = nn.InstanceNorm2d(out_channels) #CategoricalConditionalBatchNorm(out_channels, 62)
        self.relu = nn.ReLU()

    def forward(self, input, labels):
        conv2d = self.conv2d(input)
        #print(conv2d.shape)
        instance_norm = self.instance_norm(conv2d)
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
        dec_out = td.independent.Independent(td.bernoulli.Bernoulli(logits=dec_out), 3)

        # calculate training loss here lol
        #print(inputs.max())
        #print(inputs.min())
        #print(dec_out.mean)
        rec_loss = -dec_out.log_prob(inputs)
#         print(inputs)
#         print(dec_out)
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
        epsilon = torch.randn(list(x_shape[:-1]) + [z_size]).to(self.config.device)
        z = (mu + torch.exp(log_sigma / 2) * epsilon).to(self.config.device)
        kl = (0.5 * torch.mean(torch.exp(log_sigma) + (mu ** 2) - 1. - log_sigma, dim=-1)).to(self.config.device)
        # This is the 'free bits' trick mentioned in Kingma et al. (2016)
        free_bits = self.config.free_bits
        kl_loss = torch.mean(kl)
        kl_loss = torch.mean(torch.clamp_min(kl - free_bits, 0.0))
        assert kl_loss >= 0
    
        #print("z", torch.sum(torch.abs(z)))
        return z, kl_loss * self.config.kl_beta

########################################################################################################################


class SvgDecoder(nn.Module):
    def __init__(self, config, pretrained_vae):
        super().__init__()
        self.config = config
        self.pretrained_vae = pretrained_vae

    def forward(self, input):
        sampled_bottleneck = self.pretrained_visual_encoder ## TODO: fix
        if self.config.sg_bottleneck:
            sampled_bottleneck = torch.stop_gradient(sampled_bottleneck)

        if 'bottleneck' in features:
            if common_layers.shape_list(features['bottleneck'])[0] == 0:
                # return sampled_bottleneck,
                # set losses['training'] = 0 so self.top() doesn't get called on it
                return sampled_bottleneck, {'training': 0.0}
            else:
                # we want to use the given bottleneck
                sampled_bottleneck = features['bottleneck']

        # finalize bottleneck
        unbottleneck_dim = self.config.hidden_size * 2  # twice because using LSTM
        if self.config.twice_decoder:
            unbottleneck_dim = unbottleneck_dim * 2

        # unbottleneck back to LSTMStateTuple
        dec_initial_state = []

        for hi in range(hparams.num_hidden_layers):
            unbottleneck = self.unbottleneck(sampled_bottleneck, unbottleneck_dim,
                                             name_append='_{}'.format(hi))
            dec_initial_state.append(
                rnn.LSTMStateTuple(
                    c=unbottleneck[:, :unbottleneck_dim // 2],
                    h=unbottleneck[:, unbottleneck_dim // 2:]))

        dec_initial_state = tuple(dec_initial_state)

        shifted_targets = common_layers.shift_right(targets)
        # Add 1 to account for the padding added to the left from shift_right
        targets_length = common_layers.length_from_embedding(shifted_targets) + 1

        # LSTM decoder
        hparams_decoder = copy.copy(hparams)
        if hparams.twice_decoder:
            hparams_decoder.hidden_size = 2 * hparams.hidden_size

        if hparams.mode == tf.estimator.ModeKeys.PREDICT:
            decoder_outputs, _ = self.lstm_decoder_infer(
                common_layers.flatten4d3d(shifted_targets),
                targets_length, hparams_decoder, features['targets_cls'],
                train, initial_state=dec_initial_state,
                bottleneck=sampled_bottleneck)
        else:
            decoder_outputs, _ = self.lstm_decoder(
                common_layers.flatten4d3d(shifted_targets),
                targets_length, hparams_decoder, features['targets_cls'],
                train, initial_state=dec_initial_state,
                bottleneck=sampled_bottleneck)

        ret = tf.expand_dims(decoder_outputs, axis=2)

        return ret, losses


