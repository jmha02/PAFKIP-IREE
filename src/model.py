import copy
import math

import torch
import torch.nn.functional as F


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


def sanitize_finite(values, limit=10000.0):
    safe = torch.where(values == values, values, torch.zeros_like(values))
    return torch.clamp(safe, min=-limit, max=limit)


def entropy(logits):
    logits_f32 = sanitize_finite(logits.to(torch.float32), 80.0)
    p = torch.softmax(logits_f32, dim=1).clamp(min=1.0e-6)
    return -(p * torch.log(p)).sum(dim=1)


def paf_loss(train_logits, main_filter_logits, ema_filter_logits, ent_thr_ratio=0.4, alpha_ood=2.0):
    num_classes = train_logits.shape[1]
    threshold = ent_thr_ratio * math.log(float(num_classes))
    ent_train = entropy(train_logits)
    ent_main_filter = entropy(main_filter_logits)
    ent_ema_filter = entropy(ema_filter_logits)
    mask_min = ent_main_filter < threshold
    mask_max = torch.logical_and(ent_main_filter >= threshold, ent_ema_filter >= threshold)
    mask_min_f = mask_min.to(torch.float32)
    mask_max_f = mask_max.to(torch.float32)
    count_min = mask_min_f.sum().clamp(min=1.0)
    count_max = mask_max_f.sum().clamp(min=1.0)
    coeff = torch.exp(-((ent_ema_filter.detach() - threshold).clamp(min=-10.0, max=10.0)))
    loss_ind = (ent_train * mask_min_f * coeff).sum() / count_min
    loss_ood = (ent_train * mask_max_f).sum() / count_max
    return loss_ind - alpha_ood * loss_ood


def kip(main_logits, ema_logits, anchor_logits, kip_alpha=0.1):
    output_dtype = main_logits.dtype
    main_f32 = sanitize_finite(main_logits.to(torch.float32), 80.0)
    ema_f32 = sanitize_finite(ema_logits.to(torch.float32), 80.0)
    anchor_f32 = sanitize_finite(anchor_logits.to(torch.float32), 80.0)
    conf_main = torch.softmax(main_f32, dim=1).max(dim=1).values
    conf_ema = torch.softmax(ema_f32, dim=1).max(dim=1).values
    conf_anchor = torch.softmax(anchor_f32, dim=1).max(dim=1).values
    conf_mean = (conf_main + conf_ema + conf_anchor) / 3.0
    w_main = (1.0 / 3.0) + kip_alpha * (conf_main - conf_mean)
    w_ema = (1.0 / 3.0) + kip_alpha * (conf_ema - conf_mean)
    w_anchor = (1.0 / 3.0) + kip_alpha * (conf_anchor - conf_mean)
    return (
        w_main[:, None] * main_f32
        + w_ema[:, None] * ema_f32
        + w_anchor[:, None] * anchor_f32
    ).to(output_dtype)


def stable_energy(logits):
    logits_f32 = sanitize_finite(logits.to(torch.float32), 80.0)
    max_logits = torch.max(logits_f32, dim=1, keepdim=True).values
    shifted = logits_f32 - max_logits
    energy = max_logits.squeeze(1) + torch.log(torch.exp(shifted).sum(dim=1))
    return energy.to(logits.dtype)


def seeded_pafkip_transform_specs(seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    specs = []
    for _ in range(3):
        top = int(torch.randint(0, 9, (), generator=generator).item())
        left = int(torch.randint(0, 9, (), generator=generator).item())
        flip = bool(torch.rand((), generator=generator).item() < 0.5)
        specs.append((top, left, flip))
    return specs


def fixed_random_crop_hflip(images, top: int, left: int, flip: bool, padding: int = 4):
    padded = F.pad(images, (padding, padding, padding, padding), mode="constant", value=0.0)
    cropped = padded[:, :, top : top + images.shape[2], left : left + images.shape[3]]
    if flip:
        columns = [cropped[:, :, :, i : i + 1] for i in range(cropped.shape[3] - 1, -1, -1)]
        cropped = torch.cat(columns, dim=3)
    return cropped


class SeededPAFKIPTTAViews(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        self.specs = seeded_pafkip_transform_specs(seed)

    def forward(self, images):
        train = fixed_random_crop_hflip(images, *self.specs[0])
        main_filter = fixed_random_crop_hflip(images, *self.specs[1])
        ema_filter = fixed_random_crop_hflip(images, *self.specs[2])
        anchor = images
        return train, main_filter, ema_filter, anchor


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


def _npu_torch_dtype(npu_dtype: str):
    if npu_dtype == "f16":
        return torch.float16
    if npu_dtype == "bf16":
        return torch.bfloat16
    return None


def _conv2d_npu_island(x, conv, npu_dtype: str):
    island_dtype = _npu_torch_dtype(npu_dtype)
    if island_dtype is None:
        return conv(x)
    y = F.conv2d(
        x.to(island_dtype),
        conv.weight.to(island_dtype),
        None if conv.bias is None else conv.bias.to(island_dtype),
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )
    return y.to(torch.float32)


def _linear_npu_island(x, linear, npu_dtype: str):
    island_dtype = _npu_torch_dtype(npu_dtype)
    if island_dtype is None:
        return linear(x)
    y = F.linear(
        x.to(island_dtype),
        linear.weight.to(island_dtype),
        None if linear.bias is None else linear.bias.to(island_dtype),
    )
    return y.to(torch.float32)


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
        compute_dtype = torch.float32
        x_stats = x.to(compute_dtype)
        weight = weight.to(compute_dtype)
        bias = bias.to(compute_dtype)
        mean = x_stats.mean(dim=(0, 2, 3), keepdim=True)
        centered = x_stats - mean
        var = (centered * centered).mean(dim=(0, 2, 3), keepdim=True)
        return (centered * torch.rsqrt(var + self.eps) * weight + bias).to(x.dtype)


class FlatBNBottleneck(torch.nn.Module):
    def __init__(self, source_block, offset_by_name, prefix, npu_dtype="f32"):
        super().__init__()
        self.npu_dtype = npu_dtype
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

        out = _conv2d_npu_island(x, self.conv1, self.npu_dtype)
        out = self.bn1(out, flat_bn_params)
        out = torch.relu(out)

        out = _conv2d_npu_island(out, self.conv2, self.npu_dtype)
        out = self.bn2(out, flat_bn_params)
        out = torch.relu(out)

        out = _conv2d_npu_island(out, self.conv3, self.npu_dtype)
        out = self.bn3(out, flat_bn_params)

        if self.downsample_conv is not None:
            identity = _conv2d_npu_island(x, self.downsample_conv, self.npu_dtype)
            identity = self.downsample_bn(identity, flat_bn_params)

        out = out + identity
        return torch.relu(out)


class FlatBNResNet50PAFLoss(torch.nn.Module):
    def __init__(self, classes=1000, weights_name="none", npu_dtype="f32"):
        super().__init__()
        if npu_dtype not in ("f32", "f16", "bf16"):
            raise ValueError(f"unsupported npu_dtype: {npu_dtype}")
        self.npu_dtype = npu_dtype
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
        self.layer1 = self._make_layer(source.layer1, offsets, "layer1", npu_dtype)
        self.layer2 = self._make_layer(source.layer2, offsets, "layer2", npu_dtype)
        self.layer3 = self._make_layer(source.layer3, offsets, "layer3", npu_dtype)
        self.layer4 = self._make_layer(source.layer4, offsets, "layer4", npu_dtype)
        self.avgpool = source.avgpool
        self.fc = source.fc
        self.requires_grad_(False)

    @staticmethod
    def _make_layer(source_layer, offsets, prefix, npu_dtype):
        return torch.nn.ModuleList(
            [
                FlatBNBottleneck(block, offsets, f"{prefix}.{index}", npu_dtype)
                for index, block in enumerate(source_layer)
            ]
        )

    def forward_logits(self, images, flat_bn_params):
        x = _conv2d_npu_island(images, self.conv1, self.npu_dtype)
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
        return _linear_npu_island(x, self.fc, self.npu_dtype)

    def forward(self, train_images, main_filter_images, ema_filter_logits, flat_bn_params):
        train_logits = self.forward_logits(train_images, flat_bn_params)
        main_filter_logits = self.forward_logits(main_filter_images, flat_bn_params).detach()
        return paf_loss(train_logits, main_filter_logits, ema_filter_logits)


class AllBNSGDMomentumUpdate(torch.nn.Module):
    """Flat SGD+momentum update for every ResNet50 BatchNorm affine tensor."""

    def forward(self, bn_params, bn_grads, velocity, lr, sgd_momentum):
        safe_params = sanitize_finite(bn_params)
        safe_grads = sanitize_finite(bn_grads)
        safe_velocity = sanitize_finite(velocity)
        new_velocity = sanitize_finite(sgd_momentum * safe_velocity + safe_grads)
        new_bn_params = sanitize_finite(safe_params - lr * new_velocity)
        return new_bn_params, new_velocity


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
