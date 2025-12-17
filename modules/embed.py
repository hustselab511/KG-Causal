import os
import random

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights,densenet121, DenseNet121_Weights
from transformers import AutoTokenizer, AutoModel

from modules.const_var import split_class
from modules.pos_embed import get_2d_sincos_pos_embed
from modules.radgraph_m import RadGraph
from modules.transformer_m import LayerNorm, Embeddings, PositionalEncoding, FBEmbeddings
from utils import tensor_utils
from utils.radgraph_download import RADGRAPH_MODEL_DIR
from utils.tool import format_dict_to_json
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

fixed_labels = [
            "Anatomy::definitely present",
            "Observation::definitely present",
            "Observation::definitely absent",
            "Observation::uncertain",
            "Observation::measurement::definitely present",
            "Anatomy::uncertain",
            "Anatomy::definitely absent",
            "Anatomy::measurement::definitely present",
            "Observation::measurement::definitely absent",
            "Observation::measurement::uncertain",
            "Anatomy::measurement::uncertain"
        ]
# 创建标签到索引的映射字典
label_to_index = {label: idx+1 for idx, label in enumerate(fixed_labels)}




#图像分块与特征提取
class PatchEmbed(nn.Module):
    """
    resnet 1-3 block stem
    """

    def __init__(self, img_size=224, patch_size=16):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches      
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        modules = list(model.children())[:-3]
        self.embed = nn.Sequential(*modules)

    def forward(self, x):
        """
           返回 [B, H'×W',C]
        """
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.embed(x).flatten(2).transpose(1, 2)
        return x



class VisEmbed(nn.Module):
    """
    image embedding with 2d sin-cos position embedding
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=512, dropout=0.):
        super(VisEmbed, self).__init__()

        # --------------------------------------------------------------------------
        # SimVLM encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(1024, embed_dim)
        self.norm = LayerNorm(embed_dim)
        num_patches = self.patch_embed.num_patches
        # use 2d pos embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # self.norm = norm_layer(embed_dim)
        self.initialize_weights()

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.proj(x) + self.pos_embed[:, 1:, :]
        x = self.dropout(self.norm(x))
        return x


class TextEmbed(nn.Module):
    """
    test embedding with 1d sin-cos embedding
    """

    def __init__(self,tokenizer, embed_dim, vocab_size, dropout=0.):
        super(TextEmbed, self).__init__()
        self.embed = FBEmbeddings(embed_dim, vocab_size,tokenizer_m=tokenizer)
        self.pos_encode = PositionalEncoding(embed_dim, dropout)
        self.norm = LayerNorm(embed_dim)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x_ori = self.embed(x)
        x = self.norm(self.pos_encode(x_ori))
        return x
