import torch
import torch.nn.functional as F
from typing import Tuple, Dict

def loss_fpd_with_optional_ce(
    feat_clean: torch.Tensor,   # (B, C, Hf, Wf)
    feat_adv: torch.Tensor,     # (B, C, Hf, Wf)
    logits_clean: torch.Tensor, # (B, K, Hl, Wl)
    logits_adv: torch.Tensor,   # (B, K, Hl, Wl)
    labels: torch.Tensor,       # (B, Hl, Wl)
    *,
    use_ce: bool = False,       # True면 L_FPD + L_CE, False면 L_FPD만
    lambda_fpd: float = 1.0,
    lambda_ce: float = 1.0,
    ignore_index: int = 255,
    min_samples_per_class: int = 8,
) -> Tuple[torch.Tensor, Dict]:
    """
    L_total = λ_fpd * L_FPD  (+ λ_ce * L_CE if use_ce)

    L_FPD (class-conditional, decision-aware):
      For each class c,
        mean_p[(1 - cos(f_clean(p), f_adv(p))) * |P_clean(c|p) - P_adv(c|p)|]
      then average over valid classes.
      If no valid class exists in the batch, fall back to all valid pixels (non-conditional).

    Notes for stability:
      - No .item()/.detach() on tensors used to compute gradients.
      - Fallback ensures L_FPD keeps a gradient path even if class-wise masks are empty.
    """

    device = feat_clean.device
    B, C, Hf, Wf = feat_clean.shape
    K = logits_adv.shape[1]

    # 1) Normalize features (no detach)
    f_c = F.normalize(feat_clean, dim=1)
    f_a = F.normalize(feat_adv,   dim=1)

    # 2) Align labels to feature resolution (nearest, no grad needed)
    labels_feat = F.interpolate(labels.unsqueeze(1).float(),
                                size=(Hf, Wf),
                                mode="nearest").long().squeeze(1)

    # 3) Probabilities at feature resolution (keep graph)
    probs_clean = F.softmax(logits_clean, dim=1)
    probs_adv   = F.softmax(logits_adv,   dim=1)
    probs_clean_res = F.interpolate(probs_clean, size=(Hf, Wf), mode="bilinear", align_corners=False)
    probs_adv_res   = F.interpolate(probs_adv,   size=(Hf, Wf), mode="bilinear", align_corners=False)

    # 4) Flatten for class-conditional masking
    fc_flat = f_c.permute(0, 2, 3, 1).reshape(-1, C)           # (N, C)
    fa_flat = f_a.permute(0, 2, 3, 1).reshape(-1, C)           # (N, C)
    labels_flat = labels_feat.reshape(-1)                      # (N,)
    probs_clean_flat = probs_clean_res.permute(0,2,3,1).reshape(-1, K)  # (N, K)
    probs_adv_flat   = probs_adv_res.permute(0,2,3,1).reshape(-1, K)    # (N, K)

    # 5) Class-conditional FPD
    terms = []
    for c in torch.unique(labels_flat):
        c_id = int(c.item())
        if c_id == ignore_index:
            continue
        mask = (labels_flat == c_id)
        if mask.sum().item() < min_samples_per_class:
            continue

        fc_c = fc_flat[mask]         # (Nc, C)
        fa_c = fa_flat[mask]         # (Nc, C)

        # 1 - cosine
        one_minus_cos = 1.0 - (fc_c * fa_c).sum(dim=1)         # (Nc,)

        # |P_clean(c) - P_adv(c)|
        diff_prob_c = torch.abs(
            probs_clean_flat[mask, c_id] - probs_adv_flat[mask, c_id]
        )                                                      # (Nc,)

        terms.append((one_minus_cos * diff_prob_c).mean())

    if len(terms) > 0:
        L_FPD = torch.stack(terms, dim=0).mean()

    else:
        # Fallback: use all valid pixels (labels != ignore_index)
        valid_mask = (labels_flat != ignore_index)
        if valid_mask.any():
            fc_v = fc_flat[valid_mask]
            fa_v = fa_flat[valid_mask]
            y_v  = labels_flat[valid_mask]                     # (Nv,)

            one_minus_cos_all = 1.0 - (fc_v * fa_v).sum(dim=1) # (Nv,)
            # gather per-pixel GT prob difference
            idx = y_v.unsqueeze(1)                             # (Nv,1)
            pc = probs_clean_flat[valid_mask].gather(1, idx).squeeze(1)  # (Nv,)
            pa = probs_adv_flat[valid_mask].gather(1, idx).squeeze(1)    # (Nv,)
            diff_prob_all = torch.abs(pc - pa)                 # (Nv,)

            L_FPD = (one_minus_cos_all * diff_prob_all).mean()
        else:
            # No valid pixel at all -> define zero that still keeps graph when combined
            # (This scalar has no grad, but total_loss will still depend on CE if use_ce=True)
            L_FPD = probs_adv_flat.sum() * 0.0

    # 6) Optional CE
    if use_ce:
        L_CE = F.cross_entropy(logits_adv, labels.long(),
                               ignore_index=ignore_index, reduction='mean')
        total_loss = lambda_fpd * L_FPD + lambda_ce * L_CE
    else:
        total_loss = lambda_fpd * L_FPD  # pure FPD


    return total_loss
