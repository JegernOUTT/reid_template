from abc import ABC
from typing import Any, Dict, List, Tuple

import kornia.augmentation as KA
import torch

from person_reid.modelling.common import l2_norm

__all__ = ['ModuleBaseMixin', 'OnnxFreezable', 'KorniaAugmentations', 'ModuleBuilders', 'CfgT']

CfgT = Dict[str, Any]


class OnnxFreezable(ABC):
    def forward_postprocess(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return l2_norm(x, axis=1)

    @property
    def output_names(self) -> List:
        return ['output']


class KorniaAugmentations(torch.nn.Module):
    def __init__(self, transforms=None, with_keypoints_and_masks=False):
        super().__init__()
        from person_reid.utils.builders import build_transform

        if transforms is None:
            self.noop = True
            return
        else:
            self.noop = False

        self.with_keypoints_and_masks = with_keypoints_and_masks
        if with_keypoints_and_masks:
            self.transforms = KA.AugmentationSequential(
                *build_transform(transforms),
                data_keys=['input', 'mask', 'keypoints'],
                return_transform=False,
                same_on_batch=False
            )
        else:
            self.transforms = KA.AugmentationSequential(
                *build_transform(transforms),
                data_keys=['input']
            )

    def _imgs_preprocess(self, images):
        return images.permute(0, 3, 1, 2)

    def _imgs_postprocess(self, images):
        return images / 255.

    @torch.no_grad()
    def forward(self, images, masks=None, keypoints=None):
        images = self._imgs_preprocess(images)
        if self.noop:
            return self._imgs_postprocess(images)

        if self.with_keypoints_and_masks:
            return self._imgs_postprocess(self.transforms(images, masks, keypoints))
        else:
            return self._imgs_postprocess(self.transforms(images))


class ModuleBuilders:
    @staticmethod
    def build_miner(miner_distance_cfg: CfgT,
                    miner_cfg: CfgT) -> torch.nn.Module:
        from person_reid.utils.builders import build_distance, build_miner
        if miner_cfg is None:
            return None

        miner_cfg.update({
            'distance': build_distance(miner_distance_cfg)
        })
        return build_miner(miner_cfg)

    @staticmethod
    def build_loss(loss_name: str,
                   loss_distance_cfg: CfgT,
                   loss_regularizer_cfg: CfgT,
                   loss_reducer_cfg: CfgT,
                   loss_cfg: CfgT) -> torch.nn.Module:
        from person_reid.utils.builders import (build_distance, build_loss_regularizer,
                                                build_loss_reducer, build_loss)
        loss_cfg.update({
            'distance': build_distance(loss_distance_cfg),
            'embedding_regularizer': build_loss_regularizer(loss_regularizer_cfg),
            'reducer': build_loss_reducer(loss_reducer_cfg)
        })
        loss = build_loss(loss_cfg)
        loss.name = loss_name
        return loss

    @staticmethod
    def build_metric(metric_cfg: CfgT) -> torch.nn.Module:
        from person_reid.utils.builders import build_metric
        return build_metric(metric_cfg)

    @staticmethod
    def build_models(backbone_cfg: CfgT,
                     head_cfg: CfgT) -> Tuple[torch.nn.Module, torch.nn.Module]:
        from person_reid.utils.builders import build_backbone, build_head

        backbone, backbone_ch = build_backbone(backbone_cfg)
        head = build_head(backbone_ch, head_cfg)
        return backbone, head

    @staticmethod
    def build_optimizers(optimization_params,
                         optimizer_cfg: CfgT,
                         scheduler_cfg: CfgT,
                         scheduler_update_params: CfgT) -> Tuple[torch.nn.Module, torch.nn.Module]:
        from person_reid.utils.builders import build_optimizer, build_lr_scheduler

        optimizer = build_optimizer(optimizer_cfg, optimization_params)
        scheduler = build_lr_scheduler(optimizer, scheduler_cfg)
        lr_scheduler_info = {'scheduler': scheduler, **scheduler_update_params}
        return optimizer, lr_scheduler_info


class ModuleBaseMixin:
    def _register_module_list(self, module_list):
        for module in module_list:
            module_type_name = type(module).__name__.lower()
            self.__setattr__(module_type_name, module)
