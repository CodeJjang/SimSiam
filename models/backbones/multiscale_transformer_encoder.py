import math

import torch.utils.checkpoint

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import Dropout
import math
import copy
from typing import Optional
from torch import nn, Tensor

class MultiscaleTransformerEncoder(nn.Module):

    def __init__(self, dropout, encoder_dim=128, pos_encoding_dim=20, output_attention_weights=False):
        super(MultiscaleTransformerEncoder, self).__init__()

        self.backbone_cnn = BackboneCNN(output_feat_map=True, dropout=dropout, desc_dim=encoder_dim)

        self.query = nn.Parameter(torch.randn(1, encoder_dim))
        self.query_pos_encoding = nn.Parameter(torch.randn(1, encoder_dim))

        self.pos_encoding_x = nn.Parameter(torch.randn(pos_encoding_dim, int(encoder_dim / 2)))
        self.pos_encoding_y = nn.Parameter(torch.randn(pos_encoding_dim, int(encoder_dim / 2)))

        # spp levels; first feature map will be passed in a residual connection to the output
        self.spp_levels = [8, 8, 4, 2, 1]
        # self.spp_levels = [8, 4, 2, 1]

        encoder_layers = 2
        encoder_heads = 2

        self.encoder_layer1 = TransformerEncoderLayer(d_model=encoder_dim, nhead=encoder_heads,
                                                     dim_feedforward=int(encoder_dim),
                                                     dropout=0.1, activation="relu", normalize_before=False)
        self.encoder1 = TransformerEncoder(encoder_layer=self.encoder_layer1, num_layers=encoder_layers)

        self.encoder_layer2 = TransformerEncoderLayer(d_model=encoder_dim, nhead=encoder_heads,
                                                     dim_feedforward=int(encoder_dim),
                                                     dropout=0.1, activation="relu", normalize_before=False)
        self.encoder2 = TransformerEncoder(encoder_layer=self.encoder_layer2, num_layers=encoder_layers)

        # self.SPP_FC = nn.Linear(8576, encoder_dim) # for [8,4,2,1] SPP
        # self.SPP_FC = nn.Linear(8704, encoder_dim)  # for [8,8,4,2,1] SPP
        self.SPP_FC = nn.Linear(8320, encoder_dim)  # for [8,8,4,2,1] SPP with encoder_dim=128
        # self.SPP_FC = nn.Linear(16640, encoder_dim)  # for [8,8,4,2,1] SPP with encoder_dim=256
        self.output_attention_weights = output_attention_weights

    def encoder_spp(self, previous_conv, num_sample, previous_conv_size, encoder):
        attention_weights = []
        residual = None
        multi_level_seq = None
        positional_encodings = None
        for i in range(len(self.spp_levels)):

            # Pooling support
            h_wid = int(math.ceil(previous_conv_size[0] / self.spp_levels[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / self.spp_levels[i]))

            # Padding to retain orthogonal dimensions
            h_pad = int((h_wid * self.spp_levels[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.spp_levels[i] - previous_conv_size[1] + 1) / 2)

            # apply pooling
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            y = maxpool(previous_conv)

            if i == 0:
                residual = y.reshape(num_sample, -1)
            else:
                pos_encoding_2d = prepare_2d_pos_encodings(self.pos_encoding_x,
                                                           self.pos_encoding_y,
                                                           y.shape[2], y.shape[3])

                pos_encoding = pos_encoding_2d.permute(2, 0, 1)
                pos_encoding = pos_encoding[:, 0:y.shape[2], 0:y.shape[3]]
                pos_encoding = pos_encoding.reshape(
                    (pos_encoding.shape[0], pos_encoding.shape[1] * pos_encoding.shape[2]))
                pos_encoding = pos_encoding.permute(1, 0).unsqueeze(1)
                if i == 1:
                    pos_encoding = torch.cat((self.query_pos_encoding.unsqueeze(0), pos_encoding), 0)

                seq = y.reshape((y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
                seq = seq.permute(2, 0, 1)

                if i == 1:
                    query = self.query.repeat(1, seq.shape[1], 1)
                    seq = torch.cat((query, seq), 0)

                if multi_level_seq is not None:
                    multi_level_seq = torch.cat((multi_level_seq, seq), 0)
                    positional_encodings = torch.cat((positional_encodings, pos_encoding), 0)
                else:
                    multi_level_seq = seq
                    positional_encodings = pos_encoding

        enc_output = torch.utils.checkpoint.checkpoint(encoder, multi_level_seq, None, None, positional_encodings)

        cls_token = enc_output[0,]
        pred = torch.cat((residual, cls_token), 1)
        if self.output_attention_weights and i == 1:
            attention_weights = enc_output[1:].transpose(1, 0)

        if self.output_attention_weights:
            return pred, attention_weights
        return pred

    def original_encoder_spp(self, previous_conv, num_sample, previous_conv_size):
        attention_weights = []
        for i in range(len(self.spp_levels)):

            # Pooling support
            h_wid = int(math.ceil(previous_conv_size[0] / self.spp_levels[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / self.spp_levels[i]))

            # Padding to retain orthogonal dimensions
            h_pad = int((h_wid * self.spp_levels[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.spp_levels[i] - previous_conv_size[1] + 1) / 2)

            # apply pooling
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            y = maxpool(previous_conv)

            if i == 0:
                spp = y.reshape(num_sample, -1)
            else:
                pos_encoding_2d = prepare_2d_pos_encodings(self.pos_encoding_x,
                                                           self.pos_encoding_y,
                                                           y.shape[2], y.shape[3])

                pos_encoding = pos_encoding_2d.permute(2, 0, 1)
                pos_encoding = pos_encoding[:, 0:y.shape[2], 0:y.shape[3]]
                pos_encoding = pos_encoding.reshape(
                    (pos_encoding.shape[0], pos_encoding.shape[1] * pos_encoding.shape[2]))
                pos_encoding = pos_encoding.permute(1, 0).unsqueeze(1)
                pos_encoding = torch.cat((self.query_pos_encoding.unsqueeze(0), pos_encoding), 0)

                seq = y.reshape((y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
                seq = seq.permute(2, 0, 1)

                query = self.query.repeat(1, seq.shape[1], 1)
                seq = torch.cat((query, seq), 0)

                enc_output = torch.utils.checkpoint.checkpoint(self.encoder, seq, None, None, pos_encoding)

                cls_token = enc_output[0,]
                if self.output_attention_weights and i == 1:
                    attention_weights = enc_output[1:].transpose(1, 0)

                spp = torch.cat((spp, cls_token.reshape(num_sample, -1)), 1)

        if self.output_attention_weights:
            return spp, attention_weights
        return spp

    def forward_one(self, x, encoder):

        activ_map = self.backbone_cnn(x)

        spp_result = self.encoder_spp(activ_map, x.size(0),
                                      [int(activ_map.size(2)), int(activ_map.size(3))], encoder)
        if self.output_attention_weights:
            spp_activations = spp_result[0]
            attention_weights = spp_result[1]
        else:
            spp_activations = spp_result

        res = self.SPP_FC(spp_activations)
        res = F.normalize(res, dim=1, p=2)

        if self.output_attention_weights:
            return res, attention_weights
        return res

    def forward(self, x1, x2):
        res = dict()
        if not self.output_attention_weights:
            res['Emb1'] = self.forward_one(x1, self.encoder1)
            res['Emb2'] = self.forward_one(x2, self.encoder2)
        else:
            res['Emb1'], res['Emb1Attention'] = self.forward_one(x1)
            res['Emb2'], res['Emb2Attention'] = self.forward_one(x2)
        return res


class BackboneCNN(nn.Module):

    def __init__(self, dropout, output_feat_map=False, desc_dim=128):
        super(BackboneCNN, self).__init__()

        self.pre_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )

        self.block = nn.Sequential(

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(64, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU(),

            nn.Conv2d(64, desc_dim, kernel_size=3, stride=1, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(desc_dim, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU(),

            nn.Conv2d(desc_dim, desc_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(desc_dim, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU(),

            nn.Conv2d(desc_dim, desc_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(desc_dim, affine=False, momentum=0.1 ** 0.5),
            nn.ReLU(),

            nn.Conv2d(desc_dim, desc_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(desc_dim, affine=False, momentum=0.1 ** 0.5),
        )

        self.output_feat_map = output_feat_map
        if not output_feat_map:
            self.output_num = [8, 4, 2, 1]
            self.fc1 = nn.Sequential(
                nn.Linear(10880, desc_dim)
            )

        self.dropout = dropout

        return

    def input_norm(self, x):
        flat = x.reshape(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
            -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x, mode='Normalized'):
        batch_size = x.size(0)
        # conv_feats = self.block(self.input_norm(x))
        conv_feats = self.pre_block(self.input_norm(x))
        conv_feats = torch.utils.checkpoint.checkpoint(self.block, conv_feats)
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

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


def prepare_2d_pos_encodings(pos_enc_x, pos_enc_y, row_num, col_num):
    pos_enc_x = pos_enc_x[0:col_num].unsqueeze(0)  # x=[1,..,20]
    pos_enc_y = pos_enc_y[0:row_num]

    for i in range(row_num):

        curr_y = pos_enc_y[i, :].unsqueeze(0).unsqueeze(0).repeat(1, col_num, 1)

        if i == 0:
            pos_encoding_2d = torch.cat((pos_enc_x, curr_y), 2)
        else:
            curr_pos_encoding_2d = torch.cat((pos_enc_x, curr_y), 2)

            pos_encoding_2d = torch.cat((pos_encoding_2d, curr_pos_encoding_2d), 0)

    return pos_encoding_2d


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")