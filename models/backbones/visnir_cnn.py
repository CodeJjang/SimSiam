import torch
import torch.nn as nn
import math


class VisNirCNN(nn.Module):

    def __init__(self, output_feat_map=False):
        super(VisNirCNN, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),  # stride = 2
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),  # stride = 2  wrong:10368
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # batch_size, 128,8,8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.output_feat_map = output_feat_map
        if not output_feat_map:
            # prev_conv_size = 29
            self.out_pool_size = [8,4,2,1]
            # h_wid = w_wid = int(math.ceil(prev_conv_size / out_pool_size))
            # h_pad = w_pad = int((h_wid * out_pool_size - prev_conv_size + 1) / 2)
            # self.pooling = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            self.fc = nn.Linear(128 * sum([i ** 2 for i in self.out_pool_size]), 2048)

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

    def forward(self, x):
        conv_feats = self.block(self.input_norm(x))
        if self.output_feat_map:
            return conv_feats
        pooled_feats = self.spatial_pyramid_pool(conv_feats, x.shape[0], [int(conv_feats.size(2)), int(conv_feats.size(3))],
                               self.out_pool_size)
        flattened_feats = pooled_feats.view(x.shape[0], -1)
        dropped_feats = torch.nn.Dropout(0.5)(flattened_feats)
        fc = self.fc(dropped_feats)
        fc = torch.nn.functional.normalize(fc, dim=1, p=2)
        return fc
