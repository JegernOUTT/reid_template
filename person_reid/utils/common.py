from functools import wraps
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

__all__ = ['IGNORE_VAL', 'seed_everything_deterministic', 'get_checkpoint_callback',
           'load_pretrained_weights', 'filter_ignore_values']

IGNORE_VAL = int(-1e4)


def filter_ignore_values(loss_func):
    @wraps(loss_func)
    def _impl(self, pred, gt, **method_kwargs):
        pos_mask = gt != IGNORE_VAL
        return loss_func(self, pred[pos_mask], gt[pos_mask], **method_kwargs)

    return _impl


def seed_everything_deterministic(seed):
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_checkpoint_callback(callbacks) -> Optional[ModelCheckpoint]:
    try:
        return next((c for c in callbacks if type(c) == ModelCheckpoint))
    except StopIteration:
        return None


def load_pretrained_weights(module, checkpoint_path):
    if checkpoint_path is None:
        return module

    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f'Check checkpoint path: {checkpoint_path}'
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")['state_dict']
    return module.load_state_dict(state_dict, strict=False)
