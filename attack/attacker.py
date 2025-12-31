import torch
from attack import torchattacks
from attack.pgds import Attack
import re
import numpy as np
import os
from PIL import Image
import apex


mid_output = None

def get_source_layer(model, source_layer):
    layers = []
    layer_text = source_layer.split('_')
    model_backbone = model.module

    layer = model_backbone
    for i in layer_text:
        try:
            layer = getattr(layer, i)
        except:
            layer = layer[int(i)]
    layers.append(layer)
    return layers



def attacker(input, target, model, optimizer,
             attack, k_number, source_layer, classes,
             std, mean, eps=0.03, alpha=0.01, result_path=None, args=None, normalize_layer=None, training=True, fusion=False):
    ignore_label = 255
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none').cuda()
    if training:
        clip_min = 0
        clip_max = 1
    else:
        clip_min = 0
        clip_max = 255
        eps= 8
        alpha = 3

    model.eval()

    if attack =='cw':
        method = torchattacks.CW(model, steps=k_number, normalization_applied=training)
        method.set_normalization_used(mean=mean, std=std)
        adversarial_examples = method(input, target)
    elif attack == 'df':
        method = torchattacks.DeepFool(model, steps=k_number, normalization_applied=training)
        method.set_normalization_used(mean=mean, std=std)
        adversarial_examples = method(input, target)
    else: # pgd
        adversarial_examples = torch.zeros_like(input)
        for i in range(input.shape[0]):
            orig_image  = input[i:i+1].detach().clone()

            if source_layer is not None:
                def get_mid_output(m, i, o):
                    global mid_output
                    mid_output = o
                feature_layer = get_source_layer(model, source_layer)[0]
                h = feature_layer.register_forward_hook(get_mid_output)  # layer select
            if training:
                orig_result_max, _, _, orig_result = model(orig_image, y=target[i:i+1], indicate=1)
            else:
                orig_result = model(normalize_layer(orig_image))
            orig_result = orig_result.detach()
            orig_loss = criterion(orig_result, target[i:i+1])

            if 'rp' in attack:
                orig_result_max = torch.argmax(orig_result, dim=1)
                S = (orig_result_max != target[i:i+1]).long() * 2

            if source_layer is not None:
                mid_original = torch.zeros(mid_output.size()).cuda()
                mid_original.copy_(mid_output.detach())
            functions = Attack()

            adversarial_example = functions.init_linf(
                input[i:i+1],
                epsilon=eps,
                mean=mean,
                std=std,
                clamp_min=clip_min,
                clamp_max=clip_max
            )
            adversarial_example = adversarial_example.detach().clone()
            # adversarial_example = input[i:i+1].detach().clone()

            for mm in range(k_number):

                model.zero_grad()
                adversarial_example.requires_grad = True

                if training:
                    result_max, loss1, loss2, result = model(adversarial_example, y=target[i:i+1], indicate=1)
                else:
                    result = model(normalize_layer(adversarial_example))
                # #-----
                # output = result[0].data.cpu().numpy()
                # output = output.transpose(1, 2, 0)
                # prediction = np.argmax(output, axis=2)
                # gray = np.uint8(prediction)
                #
                # def colorize(gray, palette):
                #     # gray: numpy array of the label and 1*3N size list palette
                #     color = Image.fromarray(gray.astype(np.uint8)).convert('P')
                #     color.putpalette(palette)
                #     return color
                #
                # colors = np.loadtxt(args.colors_path).astype('uint8')
                # color = colorize(gray, colors)
                # color.save(f'{i}_{mm}temp.png')
                #-----
                if source_layer is not None:
                    mid_adv = torch.zeros(mid_output.size()).cuda()
                    mid_adv.copy_(mid_output)

                if training:
                    loss_our = criterion(result.float(), target[i:i + 1].detach())
                    loss = loss_our + loss1 * 0 + loss2 * 0
                    optimizer.zero_grad()
                else:
                    loss = criterion(result.float(), target[i:i + 1].detach())

                if fusion and mm==0:
                    loss = functions.yg_loss(
                        feat_clean=mid_original,
                        feat_adv=mid_adv,
                        logits_clean=orig_result,
                        logits_adv=result,
                        labels=target[i:i + 1].long(),
                    )
                else:
                    if attack == 'cospgd':
                        loss = functions.cospgd_scale(
                            predictions=result,
                            labels=target[i:i+1],
                            loss=loss,
                            num_classes=classes,
                            targeted = False,
                            one_hot = True
                        )
                    elif attack == 'segpgd':
                        loss = functions.segpgd_scale(
                            predictions=result,
                            labels=target[i:i+1],
                            loss=loss,
                            iteration=mm,
                            iterations=k_number,
                            targeted=False
                        )
                    elif attack == 'pgd':
                        loss = functions.pgd_scale(loss=loss)
                    elif attack == 'fspgd':
                        loss = functions.fspgd_scale(
                            mid_original=mid_original,
                            mid_adv=mid_adv,
                            iteration=mm,
                            iterations=k_number,
                        )
                    elif attack == 'rppgd':
                        loss, S = functions.rppgd_scale(
                            predictions=result,
                            labels=target[i:i+1],
                            loss = loss,
                            mid_adv=mid_adv,
                            iteration=mm,
                            iterations=k_number,
                            S=S
                        )
                    elif attack == 'fs_yg':
                        loss = functions.yg_loss(
                            feat_clean=mid_original,
                            feat_adv=mid_adv,
                            logits_clean=orig_result,
                            logits_adv=result,
                            labels=target[i:i+1].long(),
                        )

                if args.use_apex and args.multiprocessing_distributed and training:
                    with apex.amp.scale_loss(loss.mean(), optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.mean().backward()

                adversarial_example = functions.step_inf(
                    perturbed_image=adversarial_example,
                    epsilon=eps,
                    data_grad=adversarial_example.grad,
                    orig_image=orig_image,
                    alpha=alpha,
                    targeted=False,
                    mean=mean,
                    std=std,
                    clamp_min=clip_min,
                    clamp_max=clip_max,
                )
                adversarial_example = adversarial_example.detach()

            adversarial_examples[i:i+1] = adversarial_example.detach()
            if source_layer is not None:
                del mid_original, mid_adv
            del loss

            if source_layer is not None:
                h.remove()

            # ------Check adversarial examples
            # if not training:
            #     np_adv_example = adversarial_example[0].detach().cpu().numpy()
            #     for c, (mean_c, std_c) in enumerate(zip(mean_origin, std_origin)):
            #         np_adv_example[c, :, :] *= std_c
            #         np_adv_example[c, :, :] += mean_c
            #     np_adv_example = np_adv_example * 255
            #     np_adv_example = np_adv_example.transpose(1, 2, 0)
            #     np_adv_example = np_adv_example.astype(np.uint8)
            #     if not os.path.exists(os.path.join(result_path, 'example')):
            #         os.makedirs(os.path.join(result_path, 'example'))
            #     Image.fromarray(np_adv_example).save(os.path.join(result_path, f'example/{i}.png'))


    if training:
        adversarial_examples = adversarial_examples.detach()
        result_max_final, _, _, _ = model(adversarial_examples, y=target, indicate=1)
        model.zero_grad()
        model.train()
        return adversarial_examples, result_max_final
    else:
        return adversarial_examples