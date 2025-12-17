import torch
from torch import nn

from modules.embed import VisEmbed
from modules.transformer_m import get_hv_mask, Encoder


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout,dataset_name):
        super(VisualEncoder, self).__init__()
        self.dataset_name = dataset_name
        self.v_embed_dim = embed_dim

        # 视觉embedding
        self.vis_embed = VisEmbed(embed_dim=embed_dim, dropout=dropout)

        # 视觉encoder
        self.vis_encoder = Encoder(embed_dim=embed_dim, num_layer=num_layer, num_heads=num_heads,
                                   ff_dim=ff_dim, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.v_embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self,image):
        # 视觉特征提取+嵌入
        B = image.size(0)

        if self.dataset_name == "mimic_cxr_mini":
            hv = self.vis_embed(image.reshape(B, 3, 224, 224))
        elif self.dataset_name == "mimic_cxr":
            hv = self.vis_embed(image.reshape(B, 3, 224, 224))   
        elif self.dataset_name == "iu_xray":
            hv = self.vis_embed(image.reshape(B * 2, 3, 224, 224))
        elif self.dataset_name == "iu_xray_c":
            hv = self.vis_embed(image.reshape(B * 2, 3, 224, 224))
        else:
            raise ValueError(f"{self.dataset_name} is not defined")

        hv = hv.reshape([B, -1, self.v_embed_dim])

        # cls_token
        cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
        hv = torch.cat((cls_tokens, hv), dim=1)

        # 视觉 encode 融合其他patch信息
        hv_mask = get_hv_mask(hv)
        hv = self.vis_encoder(hv, hv_mask)

        return hv,hv_mask