# # import torch
# # import torch.nn as nn
# # from ..attack import Attack
# # import numpy as np
# #
# #
# # class DeepFool(Attack):
# #     r"""
# #     DeepFool을 모든 픽셀에 적용하는 버전. (계산 비용이 매우 높음)
# #     """
# #
# #     def __init__(self, model, steps=10, overshoot=0.02):  # steps를 낮게 설정 권장
# #         super().__init__("DeepFoolSegmentationAllPixels", model)
# #         self.steps = steps
# #         self.overshoot = overshoot
# #         self.supported_mode = ["default"]
# #
# #     def forward(self, images, labels):
# #         images = images.clone().detach().to(self.device)
# #         labels = labels.clone().detach().to(self.device)
# #
# #         batch_size, _, height, width = images.shape
# #         adv_images = images.clone().detach()
# #
# #         # 모델의 클래스 수 확인
# #         num_classes = self.get_logits(images[0:1]).shape[1]
# #
# #         for i in range(batch_size):
# #             adv_image_indiv = images[i:i + 1].clone().detach()
# #             label_mask = labels[i]
# #
# #             # DeepFool 반복 로직 수행
# #             for step in range(self.steps):
# #                 print(f"Image {i + 1}/{batch_size}, Step {step + 1}/{self.steps}...")
# #
# #                 # 이번 스텝에서 모든 픽셀로부터 계산된 섭동을 더할 텐서
# #                 total_perturbation_for_step = torch.zeros_like(adv_image_indiv)
# #
# #                 # 그래디언트 계산을 위해 설정
# #                 adv_image_indiv.requires_grad = True
# #
# #                 # 이번 스텝의 로짓을 한 번만 계산하여 재사용
# #                 all_logits = self.get_logits(adv_image_indiv)
# #
# #                 # --- 여기부터 모든 픽셀을 순회하는 매우 비용이 큰 부분 ---
# #                 for h in range(height):
# #                     for w in range(width):
# #                         target_class = label_mask[h, w].item()
# #
# #                         # ignore 레이블은 건너뜀
# #                         if target_class >= num_classes:
# #                             continue
# #
# #                         # 현재 픽셀의 로짓 벡터
# #                         fs = all_logits[0, :, h, w]
# #                         _, pre_label = torch.max(fs, dim=0)
# #
# #                         # 이미 예측이 틀렸거나, 원래부터 ignore인 픽셀은 건너뜀
# #                         if pre_label != target_class:
# #                             continue
# #
# #                         # 각 픽셀에 대해 Jacobian 계산 (매우 비효율적)
# #                         ws = self._construct_jacobian_pixel(fs, adv_image_indiv)
# #
# #                         # 원본 DeepFool 로직
# #                         f_0 = fs[target_class]
# #                         w_0 = ws[target_class]
# #
# #                         wrong_classes = [c for c in range(len(fs)) if c != target_class]
# #                         f_k = fs[wrong_classes]
# #                         w_k = ws[wrong_classes]
# #
# #                         f_prime = f_k - f_0
# #                         w_prime = w_k - w_0
# #
# #                         w_prime_norm = torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
# #                         w_prime_norm[w_prime_norm < 1e-8] = 1e-8
# #
# #                         value = torch.abs(f_prime) / w_prime_norm
# #                         _, hat_L = torch.min(value, 0)
# #
# #                         # 이 픽셀 하나를 속이기 위한 섭동 계산
# #                         delta = (
# #                                 torch.abs(f_prime[hat_L])
# #                                 * w_prime[hat_L]
# #                                 / (torch.norm(w_prime[hat_L], p=2) ** 2)
# #                         )
# #
# #                         # 계산된 섭동을 총합에 더함
# #                         total_perturbation_for_step += delta
# #                 # --- 모든 픽셀 순회 종료 ---
# #
# #                 # 그래디언트 그래프 분리
# #                 adv_image_indiv = adv_image_indiv.detach()
# #
# #                 # 합산된 섭동을 이미지에 적용
# #                 adv_image_indiv = adv_image_indiv + (1 + self.overshoot) * total_perturbation_for_step
# #                 adv_image_indiv = torch.clamp(adv_image_indiv, min=0, max=1).detach()
# #
# #             adv_images[i] = adv_image_indiv
# #
# #         return adv_images
# #
# #     def _construct_jacobian_pixel(self, y, x):
# #         x_grads = []
# #         # backward() 호출 시, adv_image_indiv의 grad가 누적되지 않도록 매번 초기화 필요
# #         if x.grad is not None:
# #             x.grad.zero_()
# #
# #         for idx, y_element in enumerate(y):
# #             y_element.backward(retain_graph=True)
# #             x_grads.append(x.grad.clone().detach())
# #             x.grad.zero_()  # 다음 y_element 계산을 위해 초기화
# #
# #         return torch.stack(x_grads).squeeze(1)
#
# import torch
# import torch.nn as nn
# from ..attack import Attack
# import numpy as np
#
#
# # '가장 취약한 픽셀' 하나만 공격하는 효율적인 버전
# class DeepFool(Attack):
#     def __init__(self, model, steps=50, overshoot=0.02):
#         super().__init__("DeepFoolSegmentation", model)
#         self.steps = steps
#         self.overshoot = overshoot
#         self.supported_mode = ["default"]
#
#     def forward(self, images, labels):
#         images = images.clone().detach().to(self.device)
#         labels = labels.clone().detach().to(self.device)
#
#         batch_size = len(images)
#         adv_images = images.clone().detach()
#
#         num_classes = self.get_logits(images[0:1]).shape[1]
#
#         for i in range(batch_size):
#             image = images[i:i + 1]
#             label_mask = labels[i]
#
#             # 1. 공격할 목표 픽셀 찾기
#             valid_pixels_mask = (label_mask < num_classes)
#             if not valid_pixels_mask.any():
#                 continue
#
#             target_class = label_mask[valid_pixels_mask][0].item()
#             logits = self.get_logits(image)
#
#             target_class_indices = (label_mask == target_class).nonzero(as_tuple=False)
#             if len(target_class_indices) == 0:
#                 continue
#
#             target_logits = logits[0, target_class, :, :]
#             flat_indices = target_class_indices[:, 0] * logits.shape[3] + target_class_indices[:, 1]
#             max_logit_index = torch.argmax(target_logits.flatten()[flat_indices])
#             target_pixel_coord = target_class_indices[max_logit_index]
#
#             h, w = target_pixel_coord[0], target_pixel_coord[1]
#
#             # 2. DeepFool 반복 로직 수행
#             adv_image_indiv = image.clone().detach()
#
#             for step in range(self.steps):
#                 adv_image_indiv.requires_grad = True
#                 fs = self.get_logits(adv_image_indiv)[0, :, h, w]
#                 _, pre_label = torch.max(fs, dim=0)
#
#                 if pre_label != target_class:
#                     break
#
#                 ws = self._construct_jacobian_pixel(fs, adv_image_indiv)
#                 adv_image_indiv = adv_image_indiv.detach()
#
#                 f_0 = fs[target_class]
#                 w_0 = ws[target_class]
#                 wrong_classes = [c for c in range(len(fs)) if c != target_class]
#                 f_k = fs[wrong_classes]
#                 w_k = ws[wrong_classes]
#                 f_prime = f_k - f_0
#                 w_prime = w_k - w_0
#
#                 w_prime_norm = torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
#                 w_prime_norm[w_prime_norm < 1e-8] = 1e-8
#
#                 value = torch.abs(f_prime) / w_prime_norm
#                 _, hat_L = torch.min(value, 0)
#
#                 delta = (
#                         torch.abs(f_prime[hat_L])
#                         * w_prime[hat_L]
#                         / (torch.norm(w_prime[hat_L], p=2) ** 2)
#                 )
#
#                 adv_image_indiv = adv_image_indiv + (1 + self.overshoot) * delta
#                 adv_image_indiv = torch.clamp(adv_image_indiv, min=0, max=1).detach()
#
#             adv_images[i] = adv_image_indiv
#
#         return adv_images
#
#     def _construct_jacobian_pixel(self, y, x):
#         x_grads = []
#         for idx, y_element in enumerate(y):
#             if x.grad is not None:
#                 x.grad.zero_()
#             retain_graph = (idx + 1 < len(y))
#             y_element.backward(retain_graph=retain_graph)
#             x_grads.append(x.grad.clone().detach())
#         return torch.stack(x_grads).squeeze(1)

import torch
import torch.nn as nn
from ..attack import Attack
from torch.func import jacrev

import numpy as np


class DeepFool(Attack):
    r"""
    DeepFool을 모든 픽셀에 적용하는 버전. (계산 비용이 매우 높음)
    """
    def __init__(self, model, steps=10, overshoot=0.02, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], normalization_applied = True): # steps를 낮게 설정 권장
        super().__init__("DeepFoolSegmentationAllPixels", model,mean=mean, std=std, normalization_applied=normalization_applied)
        self.steps = steps
        self.overshoot = overshoot
        self.supported_mode = ["default"]

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)

        if isinstance(self.model, nn.DataParallel):
            logits = self.model.module(inputs)
        else:
            logits = self.model(inputs)
        return logits

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size, _, height, width = images.shape
        adv_images = images.clone().detach()

        # 모델의 클래스 수 확인
        num_classes = self.get_logits(images[0:1]).shape[1]

        adv_image_indiv = images.clone().detach()
        label_mask = labels[0]

        # DeepFool 반복 로직 수행
        for step in range(self.steps):

            # 이번 스텝에서 모든 픽셀로부터 계산된 섭동을 더할 텐서
            total_perturbation_for_step = torch.zeros_like(adv_image_indiv)

            # 그래디언트 계산을 위해 설정
            adv_image_indiv.requires_grad = True

            # 이번 스텝의 로짓을 한 번만 계산하여 재사용
            all_logits = self.get_logits(adv_image_indiv)

            # --- 여기부터 모든 픽셀을 순회하는 매우 비용이 큰 부분 ---
            for h in range(height):
                for w in range(width):
                    target_class = label_mask[h, w].item()

                    # ignore 레이블은 건너뜀
                    if target_class >= num_classes:
                        continue

                    # 현재 픽셀의 로짓 벡터
                    fs = all_logits[0, :, h, w]
                    _, pre_label = torch.max(fs, dim=0)

                    # 이미 예측이 틀렸거나, 원래부터 ignore인 픽셀은 건너뜀
                    if pre_label != target_class:
                        continue

                    def get_pixel_logits_func(img):
                        return self.get_logits(img)[0, :, h, w]
                    ws = jacrev(get_pixel_logits_func)(adv_image_indiv)

                    # # 각 픽셀에 대해 Jacobian 계산 (매우 비효율적)
                    # ws = self._construct_jacobian_pixel(fs, adv_image_indiv)

                    # 원본 DeepFool 로직
                    f_0 = fs[target_class]
                    w_0 = ws[target_class]

                    wrong_classes = [c for c in range(len(fs)) if c != target_class]
                    f_k = fs[wrong_classes]
                    w_k = ws[wrong_classes]

                    f_prime = f_k - f_0
                    w_prime = w_k - w_0

                    w_prime_norm = torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
                    w_prime_norm[w_prime_norm < 1e-8] = 1e-8

                    value = torch.abs(f_prime) / w_prime_norm
                    _, hat_L = torch.min(value, 0)

                    # 이 픽셀 하나를 속이기 위한 섭동 계산
                    delta = (
                            torch.abs(f_prime[hat_L])
                            * w_prime[hat_L]
                            / (torch.norm(w_prime[hat_L], p=2) ** 2)
                    )

                    # 계산된 섭동을 총합에 더함
                    total_perturbation_for_step += delta
            # --- 모든 픽셀 순회 종료 ---

            # 그래디언트 그래프 분리
            adv_image_indiv = adv_image_indiv.detach()

            # 합산된 섭동을 이미지에 적용
            adv_image_indiv = adv_image_indiv + (1 + self.overshoot) * total_perturbation_for_step
            adv_image_indiv = torch.clamp(adv_image_indiv, min=0, max=1).detach()

        adv_images = adv_image_indiv

        return adv_images

    def _construct_jacobian_pixel(self, y, x):
        x_grads = []
        # backward() 호출 시, adv_image_indiv의 grad가 누적되지 않도록 매번 초기화 필요
        if x.grad is not None:
            x.grad.zero_()

        for idx, y_element in enumerate(y):
            y_element.backward(retain_graph=True)
            x_grads.append(x.grad.clone().detach())
            x.grad.zero_() # 다음 y_element 계산을 위해 초기화

        return torch.stack(x_grads).squeeze(1)