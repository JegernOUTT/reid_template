from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from detector_utils.utils.other import load_module
from pytorch_lightning import Trainer

from person_reid.utils.builders import build_data_module, build_main_module
from person_reid.utils.common import seed_everything_deterministic
from person_reid.utils.pl_utils import retrieve_data_module_info, build_params_for_trainer


def get_checkpoint_path(path: Path):
    if path.is_dir():
        print('There was given a folder, so we are trying to find the latest checkpoint in it')
        files = [(datetime.fromtimestamp(name.stat().st_mtime), name)
                 for name in path.iterdir() if name.suffix == '.ckpt']
        most_recent_file = max(files, key=lambda x: x[0])[1]
        print(f'The most recent checkpoint which was found is {most_recent_file}')
        return most_recent_file
    else:
        return path


def main():
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('path', type=Path, help='Checkpoint path')
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.path.exists(), f"Checkpoint is not found: {args.path}"

    checkpoint_path = get_checkpoint_path(args.path)

    config = load_module(args.config)
    data_module = build_data_module(config.datamodule_cfg())
    data_module_info = retrieve_data_module_info(data_module)
    trainer_cfg = config.trainer_cfg(**data_module_info)
    seed_everything_deterministic(config.seed)
    main_module = build_main_module(config.mainmodule_cfg(**data_module_info))
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=True)
    main_module.eval()

    trainer = Trainer(**build_params_for_trainer(args, trainer_cfg, main_module))
    trainer.test(main_module, datamodule=data_module)


if __name__ == '__main__':
    main()
