from functools import wraps
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

__all__ = ['IGNORE_VAL', 'plot', 'seed_everything_deterministic', 'get_checkpoint_callback',
           'load_pretrained_weights', 'filter_ignore_values']

IGNORE_VAL = int(-1e4)


def filter_ignore_values(loss_func):
    @wraps(loss_func)
    def _impl(self, pred, gt, **method_kwargs):
        pos_mask = gt != IGNORE_VAL
        return loss_func(self, pred[pos_mask], gt[pos_mask], **method_kwargs)

    return _impl


def plot(embeds, labels, fig_path='./example.pdf'):
    print('Inside plot, saving path=', fig_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    #    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)


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
