from .simsiam import SimSiam, SimSiamAdaIN
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, resnet50_cifar_variant1, visnir_cnn_backbone, \
    visnir_cnn_adain, multiscale_transformer


def get_backbone(backbone, params=None, castrate=True):
    params = params or {}
    backbone = globals()[backbone](**params)

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):
    if 'simsiam_adain' in model_cfg.name:
        model = SimSiamAdaIN(get_backbone(model_cfg.backbone, vars(model_cfg.params), castrate=True))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif 'simsiam' in model_cfg.name:
        model = SimSiam(get_backbone(model_cfg.backbone, vars(model_cfg.params), castrate=True))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif 'byol' in model_cfg.name:
        model = BYOL(get_backbone(model_cfg.backbone))
    elif 'simclr' in model_cfg.name:
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif 'swav' in model_cfg.name:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
