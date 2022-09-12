import torch
import torch.nn as nn
import math

from models.layers.adain import adain


class VisNirCNNAdaIN(nn.Module):

    def __init__(self, out_pool_size, output_feat_map=False):
        super(VisNirCNNAdaIN, self).__init__()

        # self.block = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),  # stride = 2
        #     nn.BatchNorm2d(64, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),  # stride = 2  wrong:10368
        #     nn.BatchNorm2d(128, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # batch_size, 128,8,8
        #     nn.BatchNorm2d(128, affine=False),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128, affine=False),
        # )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),  # stride = 2
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU()
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU()
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),  # stride = 2  wrong:10368
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
        )

        self.layer_6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
        )

        self.layer_7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # batch_size, 128,8,8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
        )

        self.layer_8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False)
        )

        self.output_feat_map = output_feat_map
        if not output_feat_map:
            # prev_conv_size = 29
            self.out_pool_size = out_pool_size
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

    def map_feats(self, conv_feats):
        pooled_feats = self.spatial_pyramid_pool(conv_feats, conv_feats.shape[0],
                                                 [int(conv_feats.size(2)), int(conv_feats.size(3))],
                                                 self.out_pool_size)
        flattened_feats = pooled_feats.view(conv_feats.shape[0], -1)
        dropped_feats = torch.nn.Dropout(0.5)(flattened_feats)
        fc = self.fc(dropped_feats)
        fc = torch.nn.functional.normalize(fc, dim=1, p=2)
        return fc

    def block_with_feats(self, x):
        feats = [x]
        for i in range(1, 9):
            feats.append(getattr(self, f'layer_{i}')(feats[-1]))
        if self.training:
            return feats[1:]
        return [feats[-1]]


    def forward(self, x1, x2):
        # x1_feats = self.block(self.input_norm(x1))
        # x2_feats = self.block(self.input_norm(x2))
        x1_feats = self.block_with_feats(self.input_norm(x1))
        x2_feats = self.block_with_feats(self.input_norm(x2))
        x1_feats[-1] = adain(content_feat=x1_feats[-1], style_feat=x2_feats[-1])
        x2_feats[-1] = adain(content_feat=x2_feats[-1], style_feat=x1_feats[-1])
        if self.training:
            return self.map_feats(x1_feats[-1]), self.map_feats(x2_feats[-1]), x1_feats, x2_feats
        return self.map_feats(x1_feats[-1]), self.map_feats(x2_feats[-1])


