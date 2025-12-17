import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.models import swin_v2_b

from models.KM_c import KMC
from modules.CIF import CIF
from modules.beam_search import BeamSearch
from modules.embed import VisEmbed, TextEmbed, GraEmbed, TextGraEmbed, GraEncoder
from modules.graph import GraphSelector, GraphEncoder
from modules.transformer_m import Encoder, get_hv_mask, get_ht_mask, Decoder, LayerNorm, get_cross_mask, \
    get_txt_mask, get_hf_mask, MultiDecoder, DEncoder
from modules.vision import VisualEncoder



class KM(nn.Module):
    def __init__(self, args, tokenizer):
        super(KM, self).__init__()
        self.args = args
        self.vocab_size = len(tokenizer.idx2token)
        self.tokenizer = tokenizer

        self.v_embed_dim = args["v_embed_dim"]
        self.t_embed_dim = args["t_embed_dim"]
        self.ff_dim = args["v_embed_dim"] * 4
        self.dropout = args["dropout"]
        self.en_num_layers = args["en_num_layers"]
        self.de_num_layers = args["de_num_layers"]
        self.num_heads = args["num_heads"]
        self.dataset_name = args["dataset_name"]

        # 文字embedding
        self.text_embed = TextEmbed(tokenizer=tokenizer, embed_dim=self.t_embed_dim,
                                            vocab_size=self.vocab_size + 1, dropout=self.dropout)
        pretrained_model = KMC(args)
        checkpoint = torch.load("")
        pretrained_model.load_state_dict(checkpoint)

        self.vis_encoder = pretrained_model.vis_encoder
        self.classifier = pretrained_model.classifier

        self.graph_selector = GraphSelector()
        # 知识图谱encoder
        self.gra_encoder = GraphEncoder(embed_dim=self.t_embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                                    ff_dim=self.ff_dim, dropout=self.dropout,tokenizer=self.tokenizer)

        # 生成注意力矩阵
        self.attention = CIF(embed_dim=self.v_embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                                   ff_dim=self.ff_dim, dropout=self.dropout,k_n=15)
        self.cf_attention = CIF(embed_dim=self.v_embed_dim, num_layer=self.en_num_layers, num_heads=self.num_heads,
                             ff_dim=self.ff_dim, dropout=self.dropout, k_n=15)

        self.decoder = MultiDecoder(embed_dim=self.t_embed_dim, num_layer=self.de_num_layers,
                               num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout)

        self.logit = nn.Linear(self.t_embed_dim, self.vocab_size + 1)

        self.cf_decoder = MultiDecoder(embed_dim=self.t_embed_dim, num_layer=self.de_num_layers,
                                    num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout)

        self.cf_logit = nn.Linear(self.t_embed_dim, self.vocab_size + 1)

        self.beam_search = BeamSearch(args, self.vocab_size)

    def forward(self, image, sims, targets=None, mode="train",rad_list=None):
        # 视觉特征提取+嵌入
        hv,hv_mask = self.vis_encoder(image)

        hv_mean = hv[:, 0, :]

        cl_logits = self.classifier(hv_mean)

        hv= hv[:, 1:, :]
        hv_mask = hv_mask[:, :, 1:]

        hv_d = hv

        if mode == "train":
            inx = pad_sequence([torch.tensor(sublist) for sublist in sims], batch_first=True,padding_value=-1)        
        else:
            inx = self.graph_selector.select_graphs_batch(cl_logits)


        hg, hg_mask,cf_hg,cf_hg_mask = self.gra_encoder(inx, rad_list, True)

        hg_d = hg
        cf_hg_d = cf_hg

        kg_feat, vis_feat = self.attention(img_feat=hv_d, kg_feat=hg_d, img_mask=hv_mask,kg_mask=hg_mask)
        cf_kg_feat, cf_vis_feat = self.cf_attention(img_feat=hv_d, kg_feat=cf_hg_d, img_mask=hv_mask, kg_mask=cf_hg_mask)

        if mode == "train":
            # #知识图谱embedding
            txt_mask = get_txt_mask(targets)
            ht_mask, targets = get_ht_mask(targets)

            ht = self.text_embed(targets)

            hg_cross_mask = get_cross_mask(hg_mask, txt_mask)
            hv_cross_mask = get_cross_mask(hv_mask, txt_mask)

            cf_hg_cross_mask = get_cross_mask(cf_hg_mask, txt_mask)

            # 报告生成    #torch.Size([16, 59, 512]) torch.Size([16, 393, 512]) cmcrl
            out =  self.decoder(ht, vis_feat, kg_feat, self_mask=ht_mask, cross_mask_img=hv_cross_mask,cross_mask_know=hg_cross_mask)
            out_logit = self.logit(out)
            outputs = [F.log_softmax(out_logit, dim=-1)]

            cf_out = self.cf_decoder(ht, cf_vis_feat, cf_kg_feat, self_mask=ht_mask, cross_mask_img=hv_cross_mask,cross_mask_know=cf_hg_cross_mask)
            cf_out_logit = self.cf_logit(cf_out)
            cf_outputs = [F.log_softmax(cf_out_logit, dim=-1)]

            vg_effect = [None,None]
            to_effect = [F.log_softmax(out_logit - cf_out_logit, dim=-1),None]
            o_output=[cl_logits]
            return outputs,vg_effect,to_effect,o_output
        elif mode == "sample":
            self.beam_search.load_model(self.sample_forward, self.logit)

            outputs, _ = self.beam_search.sample_beam(torch.cat([vis_feat, kg_feat], dim=1), torch.cat([hv_mask.squeeze(), hg_mask.squeeze()], dim=-1))
            self.beam_search.clean_model()
            return outputs, None
        else:
            raise ValueError("mode is not defined")

    def sample_forward(self, hf, ht, f_mask, t_mask):
        ht = self.text_embed(ht)

        B, L, D = hf.shape
        V = 196 # 视觉 token 数

        # 1) 直接切片
        visual_feats = hf[:, :V, :] 
        kg_feats = hf[:, V:, :] 

        # 2) 对应掩码同理拆分
        visual_mask = f_mask[:, :, :V] 
        kg_mask = f_mask[:, :, V:]
        return self.decoder(ht, visual_feats, kg_feats, self_mask=t_mask, cross_mask_img=visual_mask, cross_mask_know=kg_mask)