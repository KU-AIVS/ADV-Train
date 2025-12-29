import torch
import torch.nn.functional as F
from typing import Optional


def loss_fpd_margin_fg_ctx_harmonic(
    feat_clean: torch.Tensor,    # (B, C, Hf, Wf)
    feat_adv: torch.Tensor,      # (B, C, Hf, Wf)
    logits_clean: torch.Tensor,  # (B, K, Hl, Wl)
    logits_adv: torch.Tensor,    # (B, K, Hl, Wl)
    labels: torch.Tensor,        # (B, Hl, Wl)
    *,
    ignore_index: int = 255,
    min_samples_per_class: int = 8,
    use_class_conditional: bool = True,   # True: class별 L_fg, False: global L_fg
    use_margin: bool = True,              # False면 feature-only FPD
    use_log_scale: bool = False,          # log(1+J) 사용할지
    background_class: Optional[int] = 0,  # ctx로 보는 class (ex: 0=background), None이면 ctx 비활성
) -> torch.Tensor:
    r"""
    Foreground / Context-aware harmonic-mean FPD loss.

    핵심 아이디어:
      - 기존 L_FPD는 J(p)=D_f(p)·D_m(p)를 전체 픽셀 평균한 값
        L_FPD = E_all[J(p)]
      - 라벨로 픽셀을 foreground(Ω_fg), context(Ω_ctx)로 나누면
        L_FPD = π_fg·L_fg + π_ctx·L_ctx
        (π_fg, π_ctx: 픽셀 비율, L_fg=E_fg[J], L_ctx=E_ctx[J])

      - 우리는 fg/ctx 둘 다 공격 방향(↑ J)으로 밀되,
        "둘 중 더 약한 쪽" RDC를 상대적으로 올려주기 위해
        조화평균(harmonic mean)을 objective로 사용:

          L_harmonic = 2 L_fg L_ctx / (L_fg + L_ctx)

        * L_fg와 L_ctx가 같으면 L_FPD와 비슷한 scale
        * 둘 중 하나가 작으면 그쪽이 bottleneck이 되어 gradient가 더 강해짐
        * 하이퍼파라미터 추가 없음, fg/ctx를 분리해서 해석 가능

    구현 디테일:
      - D_f: feature grid(Hf×Wf)에서 cos 기반으로 계산 후 logits/label grid(Hl×Wl)로 upsample
      - D_m: logits grid에서 margin drop 계산
      - J(p) = D_f(p) * D_m(p) (use_margin=False면 D_f만 사용)
      - L_fg: foreground 영역에서의 mean J (class-conditional 옵션 지원)
      - L_ctx: background_class에 해당하는 픽셀에서의 mean J (global)

      최종 loss:
        - use_margin=False: 기존 feature-only FPD (class-conditional)로 fallback
        - use_margin=True:
            if ctx 없음 or fg 없음: 그냥 global L_FPD 사용
            else: L_harmonic = 2 L_fg L_ctx / (L_fg + L_ctx)

      공격에서는 이 loss를 maximize.
    """

    device = feat_clean.device
    B, C, Hf, Wf = feat_clean.shape
    _, K, Hl, Wl = logits_adv.shape

    # ----------------------------------------------------
    # 1) Feature drift at feature resolution → upsample
    # ----------------------------------------------------
    f_c = F.normalize(feat_clean, dim=1)
    f_a = F.normalize(feat_adv,   dim=1)

    # (B, 1, Hf, Wf)
    D_f_feat = 1.0 - (f_c * f_a).sum(dim=1, keepdim=True)

    # (B, 1, Hl, Wl)
    D_f_logits = F.interpolate(
        D_f_feat,
        size=(Hl, Wl),
        mode="bilinear",
        align_corners=False
    )

    # ----------------------------------------------------
    # 2) Flatten on logits/label grid
    # ----------------------------------------------------
    D_f_flat = D_f_logits.reshape(B, -1).reshape(-1)   # (N,)
    labels_flat = labels.reshape(B, -1).reshape(-1)    # (N,)
    N = labels_flat.numel()

    valid_mask_all = (labels_flat != ignore_index)
    if not valid_mask_all.any():
        return (D_f_flat * 0.0).sum()

    # ----------------------------------------------------
    # 3) Margin term (logits grid) if use_margin
    # ----------------------------------------------------
    D_m_flat = torch.zeros_like(D_f_flat)

    if use_margin and valid_mask_all.any():
        zc_flat = logits_clean.permute(0, 2, 3, 1).reshape(-1, K)  # (N, K)
        za_flat = logits_adv.permute(0, 2, 3, 1).reshape(-1, K)    # (N, K)

        zc_v = zc_flat[valid_mask_all]          # (Nv, K)
        za_v = za_flat[valid_mask_all]          # (Nv, K)
        y_v  = labels_flat[valid_mask_all]      # (Nv,)
        Nv = zc_v.size(0)

        idx_v = torch.arange(Nv, device=device)

        # clean margin
        zc_y = zc_v[idx_v, y_v]                 # (Nv,)
        zc_others = zc_v.clone()
        zc_others[idx_v, y_v] = -1e9
        zc_competitor, _ = zc_others.max(dim=1)
        m_clean = zc_y - zc_competitor          # (Nv,)

        # adv margin
        za_y = za_v[idx_v, y_v]                 # (Nv,)
        za_others = za_v.clone()
        za_others[idx_v, y_v] = -1e9
        za_competitor, _ = za_others.max(dim=1)
        m_adv = za_y - za_competitor            # (Nv,)

        D_m_v = torch.relu(m_clean - m_adv)     # (Nv,)
        D_m_flat[valid_mask_all] = D_m_v

    # ----------------------------------------------------
    # 4) Joint energy J (or feature-only fallback)
    # ----------------------------------------------------
    J_raw_flat = torch.zeros_like(D_f_flat)
    if use_margin:
        J_raw_flat[valid_mask_all] = D_f_flat[valid_mask_all] * D_m_flat[valid_mask_all]
    else:
        J_raw_flat[valid_mask_all] = D_f_flat[valid_mask_all]

    if use_log_scale:
        J_energy_flat = torch.zeros_like(J_raw_flat)
        J_energy_flat[valid_mask_all] = torch.log1p(J_raw_flat[valid_mask_all])
    else:
        J_energy_flat = J_raw_flat

    # ----------------------------------------------------
    # Helper: 기존 feature-only / global FPD (fallback용)
    # ----------------------------------------------------
    valid_labels = labels_flat[valid_mask_all]

    def _feature_only_fpd():
        if not use_class_conditional:
            return D_f_flat[valid_mask_all].mean()
        terms = []
        for c in torch.unique(valid_labels):
            c_id = int(c.item())
            if c_id == ignore_index:
                continue
            class_mask = (labels_flat == c_id) & valid_mask_all
            if class_mask.sum().item() < min_samples_per_class:
                continue
            terms.append(D_f_flat[class_mask].mean())
        if len(terms) > 0:
            return torch.stack(terms, dim=0).mean()
        else:
            return D_f_flat[valid_mask_all].mean()

    def _global_fpd_from_J():
        # 기존 L_FPD와 거의 동일: 전체 valid pixel의 J_energy 평균
        return J_energy_flat[valid_mask_all].mean()

    # margin 안 쓰면 harmonic-mean RDC의 의미가 약해지니,
    # 그냥 feature-only FPD로 반환
    if not use_margin:
        return _feature_only_fpd()

    # ----------------------------------------------------
    # 5) Foreground / Context mask (logits/label grid)
    # ----------------------------------------------------
    if background_class is not None:
        fg_mask_all = valid_mask_all & (labels_flat != background_class)
        ctx_mask_all = valid_mask_all & (labels_flat == background_class)
    else:
        # context 비활성: 모두 foreground 취급
        fg_mask_all = valid_mask_all
        ctx_mask_all = torch.zeros_like(valid_mask_all, dtype=torch.bool)

    # fg 또는 ctx가 전혀 없으면 harmonic mean이 정의되기 애매하므로
    # 그냥 global FPD로 fallback
    if (not fg_mask_all.any()) or (not ctx_mask_all.any()):
        return _global_fpd_from_J()

    # ----------------------------------------------------
    # 6) L_fg, L_ctx 계산
    # ----------------------------------------------------
    # (1) context: 항상 "global background"로 정의
    J_ctx = J_energy_flat[ctx_mask_all]
    if J_ctx.numel() == 0:
        return _global_fpd_from_J()
    L_ctx = J_ctx.mean()

    # (2) foreground: class-conditional or global
    if not use_class_conditional:
        J_fg = J_energy_flat[fg_mask_all]
        if J_fg.numel() == 0:
            return _global_fpd_from_J()
        L_fg = J_fg.mean()
    else:
        # class-conditional L_fg: 각 object class별 J 평균 후 평균
        terms_fg = []
        for c in torch.unique(valid_labels):
            c_id = int(c.item())
            if c_id == ignore_index:
                continue
            if background_class is not None and c_id == background_class:
                # background class는 fg class로 취급하지 않음
                continue
            class_mask = (labels_flat == c_id) & fg_mask_all
            if class_mask.sum().item() < min_samples_per_class:
                continue
            J_fg_c = J_energy_flat[class_mask]
            terms_fg.append(J_fg_c.mean())
        if len(terms_fg) == 0:
            # foreground class가 없으면 global로 fallback
            return _global_fpd_from_J()
        L_fg = torch.stack(terms_fg, dim=0).mean()

    # ----------------------------------------------------
    # 7) Harmonic-mean RDC loss
    # ----------------------------------------------------
    # L_harmonic = 2 * L_fg * L_ctx / (L_fg + L_ctx)
    # 안전하게 eps 추가
    eps = 1e-6
    denom = (L_fg + L_ctx).clamp_min(eps)
    L_harmonic = 2.0 * L_fg * L_ctx / denom

    return L_harmonic
