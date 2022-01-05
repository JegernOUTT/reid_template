import argparse
from pathlib import Path

import torch
from dssl_dl_utils.utils.other import load_module
from torch.onnx import export

from test import get_checkpoint_path
from person_reid.utils.builders import build_lightning_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    args = parser.parse_args()
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('path', type=Path, help='Checkpoint path')
    parser.add_argument('out_path', type=Path, help='Output path')
    args = parser.parse_args()

    assert args.config.exists(), f"Config is not found: {args.config}"
    assert args.path.exists(), f"Checkpoint is not found: {args.path}"

    checkpoint_path = get_checkpoint_path(args.path)

    config = load_module(args.config)

    lightning_module = build_lightning_module(config.module_cfg)
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    lightning_module.load_state_dict(state_dict, strict=True)
    lightning_module.eval()
    output_names = lightning_module.output_names
    lightning_module.forward = lightning_module.forward_postprocess

    with torch.no_grad():
        export(lightning_module, torch.randn((1, 3, config.img_side_size, config.img_side_size), dtype=torch.float32),
               args.out_path,
               opset_version=11,
               export_params=True,
               input_names=['input'],
               output_names=output_names,
               dynamic_axes={'input': [0], **{name: [0] for name in output_names}},
               do_constant_folding=True)


if __name__ == '__main__':
    main()
