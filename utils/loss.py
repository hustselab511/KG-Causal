"""
Language Model loss
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def calculate_multi_label_loss(model_output, df_gt, output_ids, class_weights=None):
    id_order = {id_val: idx for idx, id_val in enumerate(output_ids)}
    matched_gt = df_gt[df_gt.iloc[:, 0].isin(output_ids)].copy()
    
    matched_gt['sort_key'] = matched_gt.iloc[:, 0].map(id_order)
    matched_gt = matched_gt.sort_values('sort_key').drop('sort_key', axis=1)
    
    gt_labels = matched_gt.iloc[:, 1:].values
    
    if len(gt_labels) != len(output_ids):
        raise ValueError(f"标签数量({len(gt_labels)})与输入ID数量({len(output_ids)})不匹配")
    
    target = torch.tensor(gt_labels, dtype=torch.float32).to(model_output.device)
    
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(model_output.device)
        
    criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    loss = criterion(model_output, target)
    
    return loss

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class CombinedLoss(nn.Module):
    def __init__(self, lm_weight=1.0, cf_weight=0.5, kl_weight=0.3,cos_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.lm_criterion = LanguageModelCriterion()

        self.ce_loss = nn.CrossEntropyLoss()

        self.lm_weight = lm_weight
        self.v_cf_weight = 0.3
        self.g_cf_weight = 0.3
        self.t_cf_weight = 0.3
        self.o_cf_weight = 0.5

        self.kl_weight = kl_weight
        self.cos_weight = cos_weight

    def forward(self, output, target, mask, v_effect,g_effect, t_effect, o_effect):
        # 语言模型损失
        lm_loss = self.lm_criterion(output, target, mask).mean()
        #反事实
        t_cf_loss = self.lm_criterion(t_effect, target, mask).mean()

        total_loss = self.lm_weight * lm_loss + self.t_cf_weight * t_cf_loss
        v_cf_loss, g_cf_loss = 0,0
        return total_loss,lm_loss,v_cf_loss,g_cf_loss


def compute_cb_loss(output, reports_ids, reports_masks,vg_effect, to_effect):
    if isinstance(output, list):
        output = output[0]
        v_effect = vg_effect[0]
        g_effect = vg_effect[1]
        t_effect = to_effect[0]
        o_effect = to_effect[1]
    criterion = CombinedLoss()
    loss,lm_loss,v_cf_loss,g_cf_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:],v_effect,g_effect,t_effect,o_effect)
    # loss, lm_loss, cf_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:], cf_output)
    return loss,lm_loss,v_cf_loss,g_cf_loss


def compute_lm_loss(output, reports_ids, reports_masks):
    if isinstance(output, list):
        output = output[0]
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss



def compute_im_loss(pred, imgs, mask, p=16, norm_pix_loss=False):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = patchify(imgs, p)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # p = self.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

def compute_recon_loss(pred, target, mask, mode='text'):
    if mode == 'text':
        loss = compute_lm_loss(pred, target, mask)
    elif mode == 'img':
        loss = compute_im_loss(pred, target, mask)
    else:
        raise ValueError
    return loss

def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss