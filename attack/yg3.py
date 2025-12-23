import torch
import torch.nn.functional as F
from typing import Tuple

def loss_fpd_margin(
    feat_clean: torch.Tensor,   # (B, C, Hf, Wf)
    feat_adv: torch.Tensor,     # (B, C, Hf, Wf)
    logits_clean: torch.Tensor, # (B, K, Hl, Wl)
    logits_adv: torch.Tensor,   # (B, K, Hl, Wl)
    labels: torch.Tensor,       # (B, Hl, Wl)
    *,
    ignore_index: int = 255,
    min_samples_per_class: int = 8,
    use_class_conditional: bool = True,  # True: class별 평균, False: 전체 평균
    use_margin: bool = True,             # True: margin drop 곱함, False: feature-only
) -> torch.Tensor:
    """
    Margin-aware L_FPD for semantic segmentation attack.

    Per-pixel core term:
      feat_term(p) = 1 - cos(f_clean(p), f_adv(p))

      if use_margin:
        # margin-aware decision shift
        m_clean(p) = z_clean[y] - max_{k!=y} z_clean[k]
        m_adv(p)   = z_adv[y]   - max_{k!=y} z_adv[k]
        dec_term(p) = relu(m_clean(p) - m_adv(p))  # margin drop
        base(p) = feat_term(p) * dec_term(p)
      else:
        # feature-only (no decision term)
        base(p) = feat_term(p)

    Aggregation:
      if use_class_conditional:
        - 각 class c에 대해, 픽셀 수 >= min_samples_per_class 이면
            L(c) = mean_{p: y(p)=c} base(p)
        - L = mean_c L(c)
        - 유효 class가 없으면 전체 valid pixel 평균으로 fallback
      else:
        - L = mean over all valid pixels (labels != ignore_index)

    반환값:
      L (scalar), adversarial attack에서 maximize 대상
    """

    device = feat_clean.device
    B, C, Hf, Wf = feat_clean.shape
    K = logits_adv.shape[1]

    # 1) Normalize features
    f_c = F.normalize(feat_clean, dim=1)
    f_a = F.normalize(feat_adv,   dim=1)

    # 2) Align labels to feature resolution (nearest)
    labels_feat = F.interpolate(
        labels.unsqueeze(1).float(),
        size=(Hf, Wf),
        mode="nearest"
    ).long().squeeze(1)  # (B, Hf, Wf)

    # 3) Flatten
    fc_flat = f_c.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
    fa_flat = f_a.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
    labels_flat = labels_feat.reshape(-1)              # (N,)
    N = labels_flat.numel()

    # 4) 1 - cosine (feature misalignment)
    one_minus_cos_flat = 1.0 - (fc_flat * fa_flat).sum(dim=1)  # (N,)

    # 5) valid pixel mask
    valid_mask_all = (labels_flat != ignore_index)

    # 6) base term 초기화
    base_term_flat = torch.zeros_like(one_minus_cos_flat)

    if use_margin and valid_mask_all.any():
        # --- margin-aware decision term 사용 ---

        # Resize logits to feature resolution
        logits_clean_res = F.interpolate(
            logits_clean, size=(Hf, Wf),
            mode="bilinear", align_corners=False
        )
        logits_adv_res = F.interpolate(
            logits_adv, size=(Hf, Wf),
            mode="bilinear", align_corners=False
        )

        zc_flat = logits_clean_res.permute(0, 2, 3, 1).reshape(-1, K)  # (N, K)
        za_flat = logits_adv_res.permute(0, 2, 3, 1).reshape(-1, K)    # (N, K)

        # valid pixels만 뽑기
        zc_v = zc_flat[valid_mask_all]        # (Nv, K)
        za_v = za_flat[valid_mask_all]        # (Nv, K)
        y_v  = labels_flat[valid_mask_all]    # (Nv,)

        # clean margin
        idx_v = torch.arange(zc_v.size(0), device=device)
        zc_y = zc_v[idx_v, y_v]               # (Nv,)
        zc_others = zc_v.clone()
        zc_others[idx_v, y_v] = -1e9
        zc_competitor, _ = zc_others.max(dim=1)   # (Nv,)
        m_clean = zc_y - zc_competitor            # (Nv,)

        # adv margin
        za_y = za_v[idx_v, y_v]               # (Nv,)
        za_others = za_v.clone()
        za_others[idx_v, y_v] = -1e9
        za_competitor, _ = za_others.max(dim=1)   # (Nv,)
        m_adv = za_y - za_competitor              # (Nv,)

        # margin drop (relu)
        dec_term = torch.relu(m_clean - m_adv)    # (Nv,)

        # base = (1 - cos) * decision_term
        base_term_flat[valid_mask_all] = one_minus_cos_flat[valid_mask_all] * dec_term

    else:
        # --- feature-only: decision term 사용 안함 ---
        base_term_flat[valid_mask_all] = one_minus_cos_flat[valid_mask_all]

    # 7) Aggregate: class-conditional vs global
    if not valid_mask_all.any():
        # No valid pixel at all → define 0 with graph path
        L_FPD = one_minus_cos_flat.sum() * 0.0
    else:
        base_term_valid = base_term_flat[valid_mask_all]

        if use_class_conditional:
            terms = []
            for c in torch.unique(labels_flat):
                c_id = int(c.item())
                if c_id == ignore_index:
                    continue

                class_mask = (labels_flat == c_id) & valid_mask_all
                if class_mask.sum().item() < min_samples_per_class:
                    continue

                terms.append(base_term_flat[class_mask].mean())

            if len(terms) > 0:
                L_FPD = torch.stack(terms, dim=0).mean()
            else:
                # no class has enough samples → fallback to global mean
                L_FPD = base_term_valid.mean()
        else:
            # Non class-conditional: just mean over all valid pixels
            L_FPD = base_term_valid.mean()

    return L_FPD
