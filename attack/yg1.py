import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def L_ex_plus_CE_simple(
    feat_clean: torch.Tensor,   # (B, C, Hf, Wf)
    feat_adv: torch.Tensor,     # (B, C, Hf, Wf)
    logits_adv: torch.Tensor,   # (B, K, Hl, Wl)
    labels: torch.Tensor,       # (B, Hl, Wl)
    lambda_ex: float = 1.0,
    lambda_ce: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict]:
    """
    Simple version:
      L_ex = mean(1 - cosine(f_clean, f_adv))
      L_CE = cross entropy(logits_adv, labels)  (no downsampling)
    Returns total = λ_ex * L_ex + λ_ce * L_CE
    """

    device = feat_clean.device
    B, C, Hf, Wf = feat_clean.shape
    K = logits_adv.shape[1]

    # 1) normalize features along channel
    f_clean = F.normalize(feat_clean, dim=1, p=2)
    f_adv   = F.normalize(feat_adv,   dim=1, p=2)

    # 2) cosine similarity per pixel
    cos_sim = (f_clean * f_adv).sum(dim=1)  # (B, Hf, Wf)
    L_ex = (1.0 - cos_sim).mean()           # scalar

    # 3) cross-entropy (no downsampling)
    L_CE = F.cross_entropy(logits_adv, labels.long(), ignore_index=255, reduction='mean')


    # 4) combine
    # total = lambda_ex * L_ex + lambda_ce * L_CE
    total = lambda_ce * L_CE


    return total
