import argparse
from pathlib import Path

import torch
from dssl_dl_utils.utils.other import load_module
from torch.onnx import export

from person_reid.utils.pl_utils import retrieve_data_module_info
from test import get_checkpoint_path
from person_reid.utils.builders import build_data_module, build_main_module


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
    data_module_info = retrieve_data_module_info(build_data_module(config.datamodule_cfg()))
    main_module = build_main_module(config.mainmodule_cfg(**data_module_info))
    state_dict = torch.load(str(checkpoint_path), map_location='cpu')['state_dict']
    main_module.load_state_dict(state_dict, strict=True)
    main_module.eval()
    main_module.forward = main_module.forward_postprocess

    with torch.no_grad():
        export(main_module, torch.randn((1, 3, config.image_size.height, config.image_size.width),
                                        dtype=torch.float32),
               args.out_path,
               opset_version=11,
               export_params=True,
               input_names=['input'],
               output_names=main_module.output_names,
               dynamic_axes={'input': [0], **{name: [0] for name in main_module.output_names}},
               do_constant_folding=True)


if __name__ == '__main__':
    main()
