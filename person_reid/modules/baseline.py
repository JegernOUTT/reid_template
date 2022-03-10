import pytorch_lightning as pl
import torch
import wandb

from person_reid.modules.base import OnnxFreezable, ModuleBaseMixin, ModuleBuilders, KorniaAugmentations

__all__ = ['BaselinePersonReid']


class BaselinePersonReid(pl.LightningModule, OnnxFreezable, ModuleBaseMixin):
    def __init__(self, **kwargs):
        super().__init__()
        OnnxFreezable.__init__(self)
        ModuleBaseMixin.__init__(self)
        self.save_hyperparameters(kwargs)
        self._build()

    def _build(self):
        self.backbone, self.head = ModuleBuilders.build_models(
            self.hparams.backbone_cfg, self.hparams.head_cfg
        )

        self.miner = ModuleBuilders.build_miner(
            self.hparams.miner_distance_cfg,
            self.hparams.miner_cfg
        )

        self.losses = [
            ModuleBuilders.build_loss(
                self.hparams.loss_cfg.pop('name'), self.hparams.loss_distance_cfg,
                self.hparams.loss_regularizer_cfg, self.hparams.loss_reducer_cfg,
                self.hparams.loss_cfg
            )
        ]
        self._register_module_list(self.losses)

        self.metrics = [
            ModuleBuilders.build_metric(dict(type='EmbeddingsDistances')),
            ModuleBuilders.build_metric(dict(type='ConfusionMetrics'))
        ]
        self._register_module_list(self.metrics)

        self.optimizer, self.scheduler_info = ModuleBuilders.build_optimizers(
            list(self.backbone.parameters()) + list(self.head.parameters()) +
            [p for m in self.losses for p in m.parameters()],
            self.hparams.optimizer_cfg, self.hparams.scheduler_cfg,
            self.hparams.scheduler_update_params
        )

        self.train_transforms = KorniaAugmentations()
        self.val_transforms = KorniaAugmentations()

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        images, gt_labels = batch['image'], batch['class_idx']
        embeddings = self.forward(self.train_transforms(images))

        loss_values = []
        for loss in self.losses:
            if self.miner is not None and 'softmax' not in type(loss).__name__:
                loss_values.append(loss(embeddings, gt_labels, self.miner(embeddings, gt_labels)))
            else:
                loss_values.append(loss(embeddings, gt_labels))
            self.log(f'loss/{loss.name}', loss_values[-1], prog_bar=True, on_epoch=False, on_step=True, logger=True)

        self.log("lr", self.optimizers().optimizer.param_groups[0]['lr'], prog_bar=True, on_step=True, logger=False)
        return torch.stack(loss_values).sum()

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        images, gt_labels, is_query = batch['image'], batch['class_idx'], batch['is_query']
        embeddings = self.forward(self.val_transforms(images))

        for metric in self.metrics:
            metric(embeddings, gt_labels, is_query=is_query)

    def validation_epoch_end(self, outputs):
        from person_reid.metrics.utils import is_numeric_metric

        for metric in self.metrics:
            output = metric.compute()
            metric.reset()
            for name, value in output.items():
                if is_numeric_metric(value):
                    self.log(name, value, prog_bar=True, on_epoch=True, logger=True)
                else:
                    wandb.log({'global_step': self.global_step, name: value})

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler_info]

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
