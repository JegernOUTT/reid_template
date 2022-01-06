import importlib
from logging import warning
from typing import Tuple

import kornia.augmentation as Kornia
import pretrainedmodels
import pytorch_lightning.callbacks as LightningCallbacks
import pytorch_loss as PytorchExtraLosses
import pytorch_metric_learning.distances as Distances
import pytorch_metric_learning.losses as MetricLearningLosses
import pytorch_metric_learning.miners as Miners
import pytorch_metric_learning.reducers as LossReducers
import pytorch_metric_learning.regularizers as LossRegularizers
import timm
import torch
import torch.nn as TorchNNModules
import torch.optim as OptimizerLib
import torch.optim.lr_scheduler as LRSchedulerLib
import torchmetrics as PLMetrics

import person_reid.callbacks as Callbacks
import person_reid.data.module as DataModules
import person_reid.data.samplers as Samplers
import person_reid.loss.softmax_losses as SoftmaxLosses
import person_reid.metrics as CustomMetrics
import person_reid.modelling.backbones as Backbones
import person_reid.modelling.heads as Heads
import person_reid.modelling.transforms as Transforms
import person_reid.modules as Modules

__all__ = [
    'build_main_module',
    'build_data_module',
    'build_backbone',
    'build_head',
    'build_transform',
    'build_miner',
    'build_distance',
    'build_loss_regularizer',
    'build_loss_reducer',
    'build_loss',
    'build_metric',
    'build_optimizer',
    'build_lr_scheduler',
    'build_callbacks',
    'build_sampler'
]


def _base_build(config, modules_to_find):
    if config is None:
        return None

    assert isinstance(config, dict) and 'type' in config, f'Check config type validity: {config}'

    args = config.copy()
    obj_type_name = args.pop('type')

    real_type = None
    for module in modules_to_find:
        if not hasattr(module, obj_type_name):
            continue
        real_type = getattr(module, obj_type_name)
        if real_type:
            break

    assert real_type is not None, f'{obj_type_name} is not registered type in any modules: {modules_to_find}'
    return real_type(**args)


def _try_import(*module_names):
    modules = []
    for n in module_names:
        try:
            modules.append(importlib.import_module(n))
        except ModuleNotFoundError:
            warning(f'Module {n} is not installed, skipping it')
    return modules


def build_main_module(config):
    return _base_build(config, [Modules])


def build_data_module(config):
    return _base_build(config, [DataModules])


def build_backbone(config) -> Tuple[torch.nn.Module, int]:
    args = config.copy()
    backbone_type_name = args.pop('type')

    if hasattr(Backbones, backbone_type_name):
        backbone = getattr(Backbones, backbone_type_name)(**args)
        output_channels = backbone.output_channels
    elif backbone_type_name in pretrainedmodels.__dict__:
        backbone = pretrainedmodels.__dict__[backbone_type_name](**args)
        if 'squeezenet' in backbone_type_name:
            backbone = backbone.features
            output_channels = 512
        else:
            backbone.forward = backbone.features
            output_channels = backbone.last_linear.in_features
    elif backbone_type_name in timm.list_models():
        backbone = timm.create_model(backbone_type_name, **args)
        backbone.forward = backbone.forward_features
        output_channels = backbone.num_features
    else:
        assert False, f'{backbone_type_name} not found in backbones factory'

    return backbone, output_channels


def build_head(input_channels: int, config):
    config = config.copy() if config is not None else config
    config['input_channels'] = input_channels
    return _base_build(config, [Heads])


def build_transform(config):
    def _builder(cfg):
        modules = [Kornia, Transforms]

        if 'transforms' in cfg:
            cfg['transforms'] = [
                _builder(transform_cfg) for transform_cfg in cfg['transforms']
            ]

        return _base_build(cfg, modules)

    config = config.copy() if config is not None else config
    return _builder(config)


def build_miner(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [Miners])


def build_distance(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [Distances])


def build_loss_regularizer(config):
    config = config.copy() if config is not None else config if config is not None else config
    return _base_build(config, [LossRegularizers])


def build_loss_reducer(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [LossReducers])


def build_loss(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [MetricLearningLosses, SoftmaxLosses, TorchNNModules, PytorchExtraLosses])


def build_metric(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [CustomMetrics, PLMetrics])


def build_optimizer(config, params):
    config = config.copy() if config is not None else config
    modules = [OptimizerLib, *_try_import('adabelief_pytorch', 'ranger_adabelief', 'ranger')]
    config['params'] = params
    return _base_build(config, modules)


def build_lr_scheduler(optimizer, config):
    config = config.copy() if config is not None else config
    config['optimizer'] = optimizer
    return _base_build(config, [LRSchedulerLib])


def build_callbacks(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [Callbacks, LightningCallbacks])


def build_sampler(config):
    config = config.copy() if config is not None else config
    return _base_build(config, [Samplers])
