import torch
import torch.nn as nn
import math


class VisNirCNN(nn.Module):

    def __init__(self):
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
        prev_conv_size = 29
        out_pool_size = 8
        h_wid = w_wid = int(math.ceil(prev_conv_size / out_pool_size))
        h_pad = w_pad = int((h_wid * out_pool_size - prev_conv_size + 1) / 2)
        self.pooling = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        self.fc = nn.Linear(128 * out_pool_size ** 2, 2048)

    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        conv_feats = self.block(self.input_norm(x))
        pooled_feats = self.pooling(conv_feats)
        flattened_feats = pooled_feats.view(x.shape[0], -1)
        dropped_feats = torch.nn.Dropout(0.5)(flattened_feats)
        fc = self.fc(dropped_feats)
        fc = torch.nn.functional.normalize(fc, dim=1, p=2)
        return fc
