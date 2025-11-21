import torch
import math
import torch.nn.functional as F


class Attack:
    def __init__(self):
        pass

    """
    Function to take one attack step in the l-infinity norm constraint

    perturbed_image: Float tensor of shape [batch size, channels, (image spatial resolution)]
    epsilon: Float tensor: permissible epsilon range
    data_grad: gradient on the image input to the model w.r.t. the loss backpropagated
    orig_image: Float tensor of shape [batch size, channels, (image spatial resolution)]: Original unattacked image, before adding any noise
    alpha: Float tensor: attack step size
    targeted: boolean: Targeted attack or not
    clamp_min: Float tensor: minimum clip value for clipping the perturbed image back to the permisible input space
    clamp_max: Float tensor: maximum clip value for clipping the perturbed image back to the permisible input space
    grad_scale: tensor either single value or of the same shape as data_grad: to scale the added noise
    """

    @staticmethod
    def step_inf(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            targeted,
            std,
            mean,
            clamp_min,
            clamp_max,
            grad_scale=None
    ):
        if clamp_max == 1:
            assert epsilon == 0.03 and alpha==0.01, "[0,1] epsilon should be 0.03, alpha should be 0.01"
            sign_data_grad = alpha * data_grad.sign()
            unnorm_perturbed_image = perturbed_image.detach()
            unnorm_perturbed_image[:, 0, :, :] = unnorm_perturbed_image[:, 0, :, :] * std[0] + mean[0]
            unnorm_perturbed_image[:, 1, :, :] = unnorm_perturbed_image[:, 1, :, :] * std[1] + mean[1]
            unnorm_perturbed_image[:, 2, :, :] = unnorm_perturbed_image[:, 2, :, :] * std[2] + mean[2]
            unnorm_perturbed_image = unnorm_perturbed_image + sign_data_grad

            unnorm_orig_image = orig_image.detach()
            unnorm_orig_image[:, 0, :, :] = unnorm_orig_image[:, 0, :, :] * std[0] + mean[0]
            unnorm_orig_image[:, 1, :, :] = unnorm_orig_image[:, 1, :, :] * std[1] + mean[1]
            unnorm_orig_image[:, 2, :, :] = unnorm_orig_image[:, 2, :, :] * std[2] + mean[2]
            delta = torch.clamp(unnorm_perturbed_image - unnorm_orig_image, min=-epsilon, max=epsilon)

            n_pertured = torch.clamp(unnorm_orig_image + delta, clamp_min, clamp_max).detach()
            n_pertured[:, 0, :, :] = (n_pertured[:, 0, :, :] - mean[0]) / std[0]
            n_pertured[:, 1, :, :] = (n_pertured[:, 1, :, :] - mean[1]) / std[1]
            n_pertured[:, 2, :, :] = (n_pertured[:, 2, :, :] - mean[2]) / std[2]
            return n_pertured.detach()

        elif clamp_max == 255:
            assert epsilon == 8 and alpha==3, "[0,255] epsilon should be 8, alpha should be 3"
            sign_data_grad = alpha * data_grad.sign()
            if targeted:
                sign_data_grad *= -1
            if grad_scale is not None:
                sign_data_grad *= grad_scale
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_image = perturbed_image.detach() + sign_data_grad
            # Adding clipping to maintain [0,1] range
            delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
            return perturbed_image

    @staticmethod
    def init_linf(
            images,
            epsilon,
            mean,
            std,
            clamp_min=0,
            clamp_max=1,
    ):
        adversarial_example = images.clone().detach()
        if clamp_max == 1:
            noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device)
            adversarial_example[:, 0, :, :] = adversarial_example[:, 0, :, :] * std[0] + mean[0]
            adversarial_example[:, 1, :, :] = adversarial_example[:, 1, :, :] * std[1] + mean[1]
            adversarial_example[:, 2, :, :] = adversarial_example[:, 2, :, :] * std[2] + mean[2]
            adversarial_example = adversarial_example + noise
            adversarial_example = adversarial_example.clamp(clamp_min, clamp_max)
            adversarial_example[:, 0, :, :] = (adversarial_example[:, 0, :, :] - mean[0]) / std[0]
            adversarial_example[:, 1, :, :] = (adversarial_example[:, 1, :, :] - mean[1]) / std[1]
            adversarial_example[:, 2, :, :] = (adversarial_example[:, 2, :, :] - mean[2]) / std[2]

            return adversarial_example

        elif clamp_max == 255:
            noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device)
            adversarial_example = adversarial_example + noise
            adversarial_example = adversarial_example.clamp(clamp_min, clamp_max)
            return adversarial_example

    @staticmethod
    def segpgd_scale(
            predictions,
            labels,
            loss,
            iteration,
            iterations,
            targeted=False,
    ):
        lambda_t = iteration / (2 * iterations)
        output_idx = torch.argmax(predictions, dim=1)
        if targeted:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    lambda_t * loss,
                    (1 - lambda_t) * loss
                )
            ) / (predictions.shape[-2] * predictions.shape[-1])
        else:
            loss = torch.sum(
                torch.where(
                    output_idx == labels,
                    (1 - lambda_t) * loss,
                    lambda_t * loss
                )
            ) / (predictions.shape[-2] * predictions.shape[-1])
        return loss

    """
    Scaling of the pixel-wise loss as implemeted by: 
    Agnihotri, Shashank, et al. "CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks." 
    arXiv preprint arXiv:2302.02213 (2023).

    predictions: Float tensor of shape [batch size, channel, (image spatial resolution)]: Predictions made by the model
    labels: The ground truth/target labels, for semantic segmentation index tensor of the shape: [batch size, channel, (image spatial resolution)].
                                     for pixel-wise regression tasks, same shape as predictions
    loss: Float tensor: The loss between the predictions and the ground truth/target
    num_classes: int: For semantic segmentation the number of classes. None for pixel-wise regression tasks
    targeted: boolean: Targeted attack or not
    one_hot: boolean: To use one-hot encoding, SHOULD BE TRUE FOR SEMANTIC SEGMENTATION and FALSE FOR pixel-wise regression tasks
    """

    @staticmethod
    def cospgd_scale(
            predictions,
            labels,
            loss,
            num_classes=None,
            targeted=False,
            one_hot=True,
    ):
        if one_hot:
            transformed_target = torch.nn.functional.one_hot(
                torch.clamp(labels, labels.min(), num_classes - 1),
                num_classes=num_classes
            ).permute(0, 3, 1, 2)
        else:
            transformed_target = torch.nn.functional.softmax(labels, dim=1)
        cossim = torch.nn.functional.cosine_similarity(
            torch.nn.functional.softmax(predictions, dim=1),
            transformed_target,
            dim=1
        )
        if targeted:
            cossim = 1 - cossim  # if performing targeted attacks, we want to punish for dissimilarity to the target
        loss = cossim.detach() * loss
        return loss

    @staticmethod
    def fspgd_scale(
            mid_original,
            mid_adv,
            iteration,
            iterations,
            cosine=3,
    ):
        lambda_t = iteration / iterations
        n, c, h, w = mid_original.size()
        hw = h * w

        f_orig = F.normalize(mid_original.view(n, c, hw), dim=1, eps=1e-8)
        f_adv = F.normalize(mid_adv.view(n, c, hw), dim=1, eps=1e-8)
        del mid_original, mid_adv

        # (1) Diagonal similarity
        f_diag = (f_orig * f_adv).sum(dim=1)

        # (2) Cosine similarity matrix
        sim_matrix = torch.bmm(f_orig.transpose(1, 2), f_orig)  # (n, hw, hw)
        threshold_val = math.cos(math.pi / cosine)

        # (3) Mask & threshold
        if 'eye_mat' not in locals() or eye_mat.shape[1] != hw:
            eye_mat = torch.eye(hw, device=sim_matrix.device).unsqueeze(0)
        mask = 1.0 - eye_mat
        W = ((sim_matrix > threshold_val) & (mask.bool())).float()
        del sim_matrix, mask, eye_mat  # 메모리 절약
        W_sum = W.sum(dim=(1, 2))

        # (4) Similarity with f_adv
        sim_adv = torch.bmm(f_adv.transpose(1, 2), f_adv)
        f_combi = (W * sim_adv).sum(dim=(1, 2)) / (W_sum + 1e-6) / 2
        f_combi = f_combi.mean()
        del sim_adv
        del f_orig, f_adv
        loss = - lambda_t * f_diag.mean() - (1 - lambda_t) * f_combi

        return loss.mean()

    def rppgd_scale(self, predictions, labels, loss, mid_adv, iteration, iterations, S, b=0.75):
        with torch.no_grad():
            current_preds = predictions.argmax(1)
            is_correct = (current_preds == labels)

            # 상태 업데이트: True(0) 상태였던 픽셀이 오답이 되면 Boundary(1)로 변경.
            # Boundary(1)나 False(2) 상태는 그대로 유지됩니다.
            S[(S == 0) & (~is_correct)] = 1

            mask_T = (S == 0).float()
            mask_B = (S == 1).float()
            mask_F = (S == 2).float()

        lambda_1 = (2 * iterations - iteration) / (2 * iterations)
        lambda_2 = (iteration - 1) / (2 * iterations)
        lambda_3 = 1 / (2 * iterations)

        loss_R = (lambda_1 * (loss * mask_T).sum() +
                  lambda_2 * (loss * mask_B).sum() +
                  lambda_3 * (loss * mask_F).sum()) / predictions.shape[0]

        loss_P = torch.tensor(0.0).to(predictions.device)

        if mask_T.sum() > 0:
            # True Region에 존재하는 클래스들의 프로토타입을 계산합니다.
            prototypes = []
            # True Region에 있는 픽셀들의 실제 레이블
            true_region_labels = labels[mask_T.bool()]
            # 그 중 고유한 클래스 레이블들
            unique_labels_in_T = torch.unique(true_region_labels)

            if len(unique_labels_in_T) > 1:
                b, c, h, w = mid_adv.shape
                # 마스크와 레이블 맵을 특징 맵의 크기로 리사이즈
                mask_T_resized = F.interpolate(mask_T.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                labels_resized = F.interpolate(labels.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(
                    1).long()

                for cls in unique_labels_in_T:
                    # True Region 내 현재 클래스에 해당하는 픽셀 마스크
                    class_mask_T = (labels_resized == cls) & (mask_T_resized.bool())

                    if class_mask_T.sum() > 0:
                        # 마스크된 평균 풀링(Masked Average Pooling)으로 프로토타입 계산
                        proto = mid_adv[class_mask_T.unsqueeze(1).expand_as(mid_adv)].view(c,
                                                                                           -1).mean(1)
                        prototypes.append(proto)

                if len(prototypes) > 1:
                    proto_tensor = torch.stack(prototypes)
                    proto_tensor = F.normalize(proto_tensor, p=2, dim=1)
                    # 프로토타입 간의 쌍별 코사인 유사도 계산
                    cosine_sim = F.cosine_similarity(proto_tensor.unsqueeze(1), proto_tensor.unsqueeze(0), dim=2)
                    # 유사도를 최대화해야 하므로, 손실은 유사도의 합으로 정의 (경사 상승법 적용)
                    loss_P = torch.triu(cosine_sim, diagonal=1).sum()

            # --- 4. 최종 손실 결합 (L_RP) ---
        loss_RP = b * loss_R + (1 - b) * loss_P

        return loss_RP, S

    @staticmethod
    def pgd_scale(
            loss,
            predictions=None,
            labels=None,
            num_classes=None,
            targeted=False,
            one_hot=True,
    ):
        loss = loss.mean(dim=(1,2))
        return loss
