import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..attack import Attack


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    [MODIFIED FOR SEMANTIC SEGMENTATION]

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images.
        For segmentation, set higher c (e.g., 1000, 10000) and more steps (e.g., 200, 500).

    Shape:
        - images: :math:`(N, C, H, W)`
        - labels: :math:`(N, H, W)` for Segmentation.
        - output: :math:`(N, C, H, W)`.

    .. note::
        For **Semantic Segmentation Targeted Attack**, target labels :math:`(N, H, W)`
        must be set using `attack.set_target_label(target_labels)` before calling the attack.
    """

    def __init__(self, model, c=10, kappa=0, steps=50, lr=0.01, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], normalization_applied=True):
        super().__init__("CW", model, mean=mean, std=std, normalization_applied=normalization_applied)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)  # (N, H, W)

        if self.targeted:
            if self.targeted_labels is None or len(self.targeted_labels.shape) < 3:
                raise ValueError(
                    "Targeted attack for segmentation requires target labels "
                    "(N, H, W) to be set via `set_target_label()`"
                )
            target_labels = self.targeted_labels  # (N, H, W)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)  # (N,)
        prev_cost = 1e10
        dim = len(images.shape)  # 4

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            adv_images = self.tanh_space(w)

            # [수정 3] L2_loss는 이미지별 총합(sum(dim=1))의 배치 평균(mean())
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)  # (N,)
            L2_loss = current_L2.mean()  # (1,) [수정]

            outputs = self.get_logits(adv_images)  # (N, C, H, W)

            # [수정 3] f_loss는 픽셀+배치 평균(mean())
            if self.targeted:
                f_loss = self.f(outputs, target_labels).mean()  # (1,) [수정]
            else:
                f_loss = self.f(outputs, labels).mean()  # (1,) [수정]

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # --- best_adv_images 업데이트 로직 ---
            pre = torch.argmax(outputs.detach(), dim=1)  # (N, H, W)
            num_classes = outputs.shape[1]

            # [수정 2] 세그맨테이션 및 ignore_index를 고려한 condition 계산
            if self.targeted:
                valid_mask = (target_labels >= 0) & (target_labels < num_classes)
                # 유효한 픽셀이 *모두* 타겟과 일치해야 함
                condition = ((pre == target_labels) | (~valid_mask)).view(len(images), -1).all(dim=1).float()
            else:
                valid_mask = (labels >= 0) & (labels < num_classes)
                # 유효한 픽셀 중 *하나라도* 원본과 달라야 함
                condition = ((pre != labels) & valid_mask).view(len(images), -1).any(dim=1).float()

            mask = condition * (best_L2 > current_L2.detach())  # (N,)
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            # [수정 4] (N,) 마스크를 (N, 1, 1, 1)로 브로드캐스팅 가능하게 변경
            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # --- Early stop ---
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        adv_images = self.tanh_space(w)

        return adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        # outputs: (N, C, H, W)
        # labels: (N, H, W)
        num_classes = outputs.shape[1]

        # [수정 1] 원본 labels 훼손 방지 및 ignore_index 처리
        valid_mask = (labels >= 0) & (labels < num_classes)  # (N, H, W)

        labels_safe = labels.clone()  # 원본 훼손 방지
        labels_safe[~valid_mask] = 0  # ignore 픽셀을 0으로 (어차피 마스킹됨)

        one_hot_labels = F.one_hot(labels_safe, num_classes=num_classes).to(self.device)
        one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)  # (N, C, H, W)

        # (N, H, W)
        other = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 1e9, dim=1)[0]
        # (N, H, W)
        real = (one_hot_labels * outputs).sum(dim=1)

        if self.targeted:
            loss = torch.clamp((other - real), min=-self.kappa)
        else:
            loss = torch.clamp((real - other), min=-self.kappa)

        # [수정 1] ignore 픽셀의 손실을 0으로 마스킹
        loss[~valid_mask] = 0

        return loss