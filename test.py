from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from detector_utils.utils.other import load_module
from pytorch_lightning import Trainer

from person_reid.utils.builders import build_lightning_module, build_callbacks


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

    lightning_module = build_lightning_module(config.module_cfg)
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    lightning_module.load_state_dict(state_dict, strict=True)
    lightning_module.eval()

    if 'callbacks' in config.trainer_cfg:
        config.trainer_cfg['callbacks'] = [
            build_callbacks(config)
            for config in config.trainer_cfg['callbacks']
        ]

    trainer = Trainer(**config.trainer_cfg)
    trainer.test(lightning_module)


if __name__ == '__main__':
    main()
