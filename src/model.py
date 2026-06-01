import copy
import math

import torch


def configure_pafkip_resnet50(model):
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def collect_bn_affine_params(model):
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ("weight", "bias"):
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def entropy(logits):
    p = torch.softmax(logits, dim=1).clamp(min=1.0e-6)
    return -(p * torch.log(p)).sum(dim=1)


def paf_loss(main_logits, ema_logits, ent_thr_ratio=0.4, alpha_ood=2.0):
    num_classes = main_logits.shape[1]
    threshold = ent_thr_ratio * math.log(float(num_classes))
    ent_main = entropy(main_logits)
    ent_ema = entropy(ema_logits)
    mask_min = ent_main < threshold
    mask_max = torch.logical_and(ent_main >= threshold, ent_ema >= threshold)
    mask_min_f = mask_min.to(torch.float32)
    mask_max_f = mask_max.to(torch.float32)
    count_min = mask_min_f.sum().clamp(min=1.0)
    count_max = mask_max_f.sum().clamp(min=1.0)
    coeff = torch.exp(-((ent_ema.detach() - threshold).clamp(min=-10.0, max=10.0)))
    loss_ind = (ent_main * mask_min_f * coeff).sum() / count_min
    loss_ood = (ent_main * mask_max_f).sum() / count_max
    return loss_ind - alpha_ood * loss_ood


def kip(main_logits, ema_logits, anchor_logits, kip_alpha=0.1):
    conf_main = torch.softmax(main_logits, dim=1).max(dim=1).values
    conf_ema = torch.softmax(ema_logits, dim=1).max(dim=1).values
    conf_anchor = torch.softmax(anchor_logits, dim=1).max(dim=1).values
    conf_mean = (conf_main + conf_ema + conf_anchor) / 3.0
    w_main = (1.0 / 3.0) + kip_alpha * (conf_main - conf_mean)
    w_ema = (1.0 / 3.0) + kip_alpha * (conf_ema - conf_mean)
    w_anchor = (1.0 / 3.0) + kip_alpha * (conf_anchor - conf_mean)
    return (
        w_main[:, None] * main_logits
        + w_ema[:, None] * ema_logits
        + w_anchor[:, None] * anchor_logits
    )


def make_resnet50_tta_models_with_weights(classes=1000, weights_name="none"):
    import torchvision.models as models

    weights = None
    if weights_name == "default":
        if classes != 1000:
            raise ValueError("torchvision default ResNet50 weights require classes=1000")
        weights = models.ResNet50_Weights.DEFAULT
    elif weights_name != "none":
        raise ValueError(f"unsupported weights_name: {weights_name}")

    main = configure_pafkip_resnet50(models.resnet50(weights=weights, num_classes=classes))
    ema = copy.deepcopy(main).eval()
    anchor = copy.deepcopy(main).eval()
    for model in (ema, anchor):
        model.requires_grad_(False)
    return main, ema, anchor


class FlatBatchNorm2d(torch.nn.Module):
    def __init__(self, source_bn, weight_offset, bias_offset):
        super().__init__()
        self.num_features = int(source_bn.num_features)
        self.eps = float(source_bn.eps)
        self.weight_offset = int(weight_offset)
        self.bias_offset = int(bias_offset)

    def forward(self, x, flat_bn_params):
        weight = flat_bn_params[
            self.weight_offset : self.weight_offset + self.num_features
        ].reshape(1, self.num_features, 1, 1)
        bias = flat_bn_params[
            self.bias_offset : self.bias_offset + self.num_features
        ].reshape(1, self.num_features, 1, 1)
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=(0, 2, 3), keepdim=True)
        return centered * torch.rsqrt(var + self.eps) * weight + bias


class FlatBNBottleneck(torch.nn.Module):
    def __init__(self, source_block, offset_by_name, prefix):
        super().__init__()
        self.conv1 = source_block.conv1
        self.bn1 = FlatBatchNorm2d(
            source_block.bn1,
            offset_by_name[f"{prefix}.bn1.weight"],
            offset_by_name[f"{prefix}.bn1.bias"],
        )
        self.conv2 = source_block.conv2
        self.bn2 = FlatBatchNorm2d(
            source_block.bn2,
            offset_by_name[f"{prefix}.bn2.weight"],
            offset_by_name[f"{prefix}.bn2.bias"],
        )
        self.conv3 = source_block.conv3
        self.bn3 = FlatBatchNorm2d(
            source_block.bn3,
            offset_by_name[f"{prefix}.bn3.weight"],
            offset_by_name[f"{prefix}.bn3.bias"],
        )
        self.downsample_conv = None
        self.downsample_bn = None
        if source_block.downsample is not None:
            self.downsample_conv = source_block.downsample[0]
            self.downsample_bn = FlatBatchNorm2d(
                source_block.downsample[1],
                offset_by_name[f"{prefix}.downsample.1.weight"],
                offset_by_name[f"{prefix}.downsample.1.bias"],
            )

    def forward(self, x, flat_bn_params):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, flat_bn_params)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, flat_bn_params)
        out = torch.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, flat_bn_params)

        if self.downsample_conv is not None:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity, flat_bn_params)

        out = out + identity
        return torch.relu(out)


class FlatBNResNet50PAFLoss(torch.nn.Module):
    def __init__(self, classes=1000, weights_name="none"):
        super().__init__()
        source, _, _ = make_resnet50_tta_models_with_weights(classes, weights_name)
        params, names = collect_bn_affine_params(source)
        self.bn_param_names = names
        offsets = {}
        offset = 0
        for name, param in zip(names, params):
            offsets[name] = offset
            offset += param.numel()
        self.bn_param_count = len(names)
        self.bn_scalar_count = int(offset)

        self.conv1 = source.conv1
        self.bn1 = FlatBatchNorm2d(source.bn1, offsets["bn1.weight"], offsets["bn1.bias"])
        self.maxpool = source.maxpool
        self.layer1 = self._make_layer(source.layer1, offsets, "layer1")
        self.layer2 = self._make_layer(source.layer2, offsets, "layer2")
        self.layer3 = self._make_layer(source.layer3, offsets, "layer3")
        self.layer4 = self._make_layer(source.layer4, offsets, "layer4")
        self.avgpool = source.avgpool
        self.fc = source.fc
        self.requires_grad_(False)

    @staticmethod
    def _make_layer(source_layer, offsets, prefix):
        return torch.nn.ModuleList(
            [
                FlatBNBottleneck(block, offsets, f"{prefix}.{index}")
                for index, block in enumerate(source_layer)
            ]
        )

    def forward_logits(self, images, flat_bn_params):
        x = self.conv1(images)
        x = self.bn1(x, flat_bn_params)
        x = torch.relu(x)
        x = self.maxpool(x)
        for block in self.layer1:
            x = block(x, flat_bn_params)
        for block in self.layer2:
            x = block(x, flat_bn_params)
        for block in self.layer3:
            x = block(x, flat_bn_params)
        for block in self.layer4:
            x = block(x, flat_bn_params)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def forward(self, images, ema_logits, flat_bn_params):
        return paf_loss(self.forward_logits(images, flat_bn_params), ema_logits)


class AllBNSGDUpdate(torch.nn.Module):
    """Flat SGD update for every ResNet50 BatchNorm affine tensor."""

    def forward(self, bn_params, bn_grads, lr):
        return bn_params - lr * bn_grads


def flatten_bn_params(params):
    flats = [p.detach().reshape(-1).to(torch.float32) for p in params]
    return torch.cat(flats) if flats else torch.empty(0, dtype=torch.float32)


def flatten_bn_grads(params):
    flats = []
    for p in params:
        if p.grad is None:
            flats.append(torch.zeros_like(p.detach()).reshape(-1).to(torch.float32))
        else:
            flats.append(p.grad.detach().reshape(-1).to(torch.float32))
    return torch.cat(flats) if flats else torch.empty(0, dtype=torch.float32)
