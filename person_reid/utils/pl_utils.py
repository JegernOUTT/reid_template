import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from person_reid.utils.builders import build_callbacks
from person_reid.utils.common import get_checkpoint_callback

__all__ = ['retrieve_data_module_info', 'build_params_for_trainer', 'save_best_checkpoint']


def retrieve_data_module_info(data_module):
    data_module.setup(stage="fit")
    if rank_zero_only.rank == 0:
        data_module.resample()
    print(f'Train person categories count: {data_module.person_categories_count("train")}')
    print(f'Val person categories count: {data_module.person_categories_count("val")}')
    print(f'Train camera categories count: {data_module.cameras_categories_count("train")}')
    print(f'Val camera categories count: {data_module.cameras_categories_count("val")}\n')

    return dict(
        train_num_classes=data_module.person_categories_count("train"),
        train_dataset_len=data_module.len('train'),
        val_num_classes=data_module.person_categories_count('val'),
        val_dataset_len=data_module.len('val')
    )


def build_params_for_trainer(args, trainer_cfg, lightning_module):
    if 'callbacks' in trainer_cfg:
        trainer_cfg['callbacks'] = [
            build_callbacks(config)
            for config in trainer_cfg['callbacks']
        ]
    logger = WandbLogger(**trainer_cfg.pop('wandb_logger'))
    logger.watch(lightning_module)
    if rank_zero_only.rank == 0:
        wandb.save(str(args.config))
    return dict(logger=[logger], **trainer_cfg)


def save_best_checkpoint(trainer_cfg):
    if rank_zero_only.rank == 0:
        maybe_cp_callback = get_checkpoint_callback(trainer_cfg['callbacks'])
        if maybe_cp_callback is not None:
            wandb.save(maybe_cp_callback.best_model_path)
