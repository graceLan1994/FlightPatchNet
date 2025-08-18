import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from layers.Embed import DataEmbedding_inverted
from layers.PatchMixer import PatchEncoder, PredictionHead, PatchDecoder
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from utils.tools import test_params_flop


class FlightPatchNet(nn.Module):
    '''
    FlightPatchNet
    '''

    def __init__(self,
                 in_len,
                 out_len,
                 in_chn,
                 out_chn,
                 patch_sizes,
                 hid_len,
                 hid_pch,
                 hid_pred,
                 e_layers,
                 d_model,
                 norm,
                 last_norm,
                 activ,
                 drop,
                 reduction="sum") -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.reduction = reduction
        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.patch_sizes = patch_sizes
        self.e_layers = e_layers
        self.d_model = d_model
        self.paddings = []
        self.pred_fusion = nn.Linear(len(patch_sizes), 1)
        self.scale_fusion_dim = in_chn * in_len
        self.channel_fusion_dim = in_len * len(patch_sizes)
        self.hid_chn = 16

        for i, patch_size in enumerate(patch_sizes):

            res = in_len % patch_size
            padding = (patch_size - res) % patch_size
            self.paddings.append(padding)
            padded_len = in_len + padding

            self.patch_encoders.append(
                PatchEncoder(padded_len, hid_len, patch_size, hid_pch, in_chn, norm, activ, drop)
            )

            self.patch_decoders.append(
                PatchDecoder(padded_len, hid_len, in_chn, patch_size, hid_pch, norm, activ, drop))

            if out_len != 0 and out_chn != 0:

                self.pred_heads.append(
                    PredictionHead(in_len, out_len, hid_pred,
                                   in_chn, out_chn, self.hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())

        self.time_embedding = DataEmbedding_inverted(self.in_chn, self.d_model)
        # temporal Encoder
        self.global_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=3, attention_dropout=0.1), d_model, n_heads=8),
                    d_model,
                    d_ff=512,
                    dropout=drop,
                    activation=activ
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.channel_projection = nn.Linear(d_model, self.in_chn, bias=True)
          
        self.scale_fusion = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=3, attention_dropout=0.1), self.scale_fusion_dim, n_heads=4),
                    self.scale_fusion_dim,
                    d_ff=512,
                    dropout=drop,
                    activation=activ
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.scale_fusion_dim)
        )
        self.channel_fusion = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=3, attention_dropout=0.1), self.channel_fusion_dim, n_heads=4),
                    self.channel_fusion_dim,
                    d_ff=512,
                    dropout=drop,
                    activation=activ
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.channel_fusion_dim)
        )


    def forward(self, x):
        # B,C,L
        if self.last_norm:
            x_last = x[:, :, [-1]].detach()
            x = x - x_last
        preds = []
        channel_enc_out = self.time_embedding(x.permute(0, 2, 1))
        # [Batch Time d_model]
        channel_enc_out, attns = self.global_encoder(channel_enc_out, attn_mask=None)
        # [Batch Time d_model]
        x = self.channel_projection(channel_enc_out).permute(0, 2, 1)
        # [Batch  channel Time]

        last_comp = torch.zeros_like(x)
        decoders = []
        for i in range(len(self.patch_sizes)):
            x_in = x
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)

            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            decoders.append(comp)
            x = x + comp

        multi_scale_dec = torch.stack(decoders, dim=0)

        scale_fusion_out, scale_fusion = self.scale_fusion(
            multi_scale_dec.view(-1, len(self.patch_sizes), self.scale_fusion_dim))

        # h,b,c*l
        scale_fusion_out = scale_fusion_out.view(len(self.patch_sizes), -1, self.in_chn, self.in_len)

        channel_fusion_out, chn_attn = self.channel_fusion(scale_fusion_out.view(-1, self.in_chn, self.channel_fusion_dim))
        # B,C,H*L
        channel_fusion_out = channel_fusion_out.view(len(self.patch_sizes), -1, self.in_chn, self.in_len)

        # 多头预测
        for i in range(len(self.patch_sizes)):
            last_comp = last_comp + channel_fusion_out[i, ...].squeeze(0)
            pred = self.pred_heads[i](last_comp)

            preds.append(pred)
        if self.out_len != 0 and self.out_chn != 0:

            y_pred = reduce(preds, "h b c l -> b c l",self.reduction)
            if self.last_norm and self.out_chn == self.in_chn:
                y_pred += x_last

            return y_pred
        else:
            return None





if __name__ == '__main__':
    model = FlightPatchNet(60, 10, 6, 6, [60, 30, 20, 10, 5], 128, 128, 256, 3, 128, 'bn', True,
                                           'relu', 0.5)
    x = torch.randn(16, 6, 60)
    y = model(x)
    print(y.shape)
    test_params_flop(model, (6, 60))
