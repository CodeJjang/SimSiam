import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout


class SPP(nn.Module):

    def __init__(self, backbone):
        super(SPP, self).__init__()

        self.backbone = backbone
        self.output_num = [8, 4, 2, 1]
        self.fc1 = nn.Sequential(
            nn.Linear(10880, 128)
        )


    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    @staticmethod
    def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, x, mode='Normalized'):
        batch_size = x.size(0)
        conv_feats = self.block(self.input_norm(x))
        if self.output_feat_map:
            return conv_feats

        spp = spatial_pyramid_pool(conv_feats, batch_size, [int(conv_feats.size(2)), int(conv_feats.size(3))],
                                   self.output_num)

        spp = Dropout(self.dropout)(spp)

        feature_a = self.fc1(spp).reshape(batch_size, -1)

        if mode == 'Normalized':
            return F.normalize(feature_a, dim=1, p=2)
        else:
            return feature_a
