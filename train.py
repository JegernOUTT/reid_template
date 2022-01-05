from argparse import ArgumentParser
from pathlib import Path

import torch
from detector_utils.utils.other import load_module
from pytorch_lightning import Trainer

from person_reid.utils.builders import build_main_module, build_data_module
from person_reid.utils.common import seed_everything_deterministic, load_pretrained_weights
from person_reid.utils.pl_utils import (build_params_for_trainer, save_best_checkpoint,
                                        retrieve_data_module_info)


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"

    config = load_module(args.config)
    data_module = build_data_module(config.datamodule_cfg())
    data_module_info = retrieve_data_module_info(data_module)
    trainer_cfg = config.trainer_cfg(**data_module_info)
    seed_everything_deterministic(config.seed)
    if hasattr(config, 'mp_start_method'):
        torch.multiprocessing.set_start_method(config.mp_start_method)

    main_module = build_main_module(config.mainmodule_cfg(**data_module_info))
    load_pretrained_weights(main_module, config.__dict__.get('pretrained', None))
    trainer = Trainer(**build_params_for_trainer(args, trainer_cfg, main_module))
    trainer.fit(main_module, datamodule=data_module)
    save_best_checkpoint(trainer_cfg)


if __name__ == '__main__':
    main()
