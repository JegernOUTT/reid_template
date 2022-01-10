from pathlib import Path

from dssl_dl_utils import Size2D

seed = 42
gpus = [0]
batch_size = 256
epochs = 300
image_size = Size2D(96, 192)
num_workers = 16 // len(gpus)
backbone_max_stride = 16
reid_features_number = 512
data_base_path = Path('/media/svakhreev/fast/person_reid')


def datamodule_cfg():
    return dict(
        type='DataModule',
        train_paths=[p for p in (data_base_path / 'train').iterdir() if p.suffix == '.tar'],
        val_paths=[p for p in (data_base_path / 'test').iterdir() if p.suffix == '.tar'],
        sampler=dict(type='PersonSampler', output_path=data_base_path / 'train_shards'),
        full_resample=False,
        image_size=image_size,
        with_keypoints_and_masks=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=8
    )


def trainer_cfg(train_num_classes, **kwargs):
    return dict(
        gpus=gpus,
        max_epochs=epochs,
        callbacks=[
            dict(type='LearningRateMonitor', logging_interval='step'),
            dict(type='ModelCheckpoint', save_top_k=5, verbose=True, mode='max',
                 monitor='accuracy_1_min', dirpath='./checkpoints/', filename='{epoch:02d}_{accuracy_1_min:.2f}')
        ],
        benchmark=True,
        deterministic=True,
        terminate_on_nan=True,
        precision=16,
        sync_batchnorm=True,
        strategy='ddp_find_unused_parameters_false',
        check_val_every_n_epoch=5,
        wandb_logger=dict(
            name=f'{Path(__file__).stem}_{image_size.width}x{image_size.height}'
                 f'_bs{batch_size}_ep{epochs}_cls{train_num_classes}',
            project='person_reid'
        )
    )


def mainmodule_cfg(train_num_classes, train_dataset_len, **kwargs):
    return dict(
        type='BaselinePersonReid',
        # Model agnostic parameters
        backbone_cfg=dict(type='OSNET_AIN_x0_75'),
        head_cfg=dict(
            type='GDC',
            reid_features_number=reid_features_number,
            input_conv_kernel_size=(image_size.height // backbone_max_stride,
                                    image_size.width // backbone_max_stride)
        ),
        # Batch data miner agnostic parameters
        miner_distance_cfg=None,
        miner_cfg=None,

        # Loss stuff agnostic parameters
        loss_distance_cfg=None,
        loss_regularizer_cfg=None,
        loss_reducer_cfg=None,
        loss_cfg=dict(type='SphereProduct2', name='sphere2',
                      in_features=reid_features_number, out_features=train_num_classes),

        # Optimization stuff agnostic parameters
        optimizer_cfg=dict(
            type='AdamW',
            lr=1e-7 * len(gpus),
            betas=(0.9, 0.999),
            weight_decay=0.05
        ),
        scheduler_cfg=dict(
            type='CyclicLR',
            base_lr=1e-5 * len(gpus),
            max_lr=1e-3 * len(gpus),
            step_size_up=int(train_dataset_len // batch_size * (epochs * 0.1)),
            mode='triangular2',
            cycle_momentum=False,
        ),
        scheduler_update_params=dict(
            interval='step',
            frequency=1
        )
    )
