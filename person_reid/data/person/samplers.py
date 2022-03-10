import io
import os
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import List

import numpy as np
import webdataset
from tqdm import tqdm

__all__ = ['PersonClassSampler']


def default_intra_class_sampler(images, images_per_class=12, **kwargs):
    if len(images) < images_per_class:
        selected_images = images
    else:
        selected_images = random.sample(images, k=images_per_class)
    return selected_images


class PersonClassSampler:
    def __init__(self,
                 output_path: Path,
                 shards_max_count=500,
                 images_ext=('.jpg', '.jpeg', '.png'),
                 intra_person_sampler=default_intra_class_sampler):
        self.output_path = output_path
        self.shards_max_count = shards_max_count
        self.images_ext = set(images_ext)
        self.intra_class_sampler = intra_person_sampler

    def _grouped_output_filename(self, class_idx: int):
        folder_idx = class_idx // self.shards_max_count
        output_path = self.output_path / f'{folder_idx}/{class_idx}.tar'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def resample_dataset(self, dataset) -> List[Path]:
        shutil.rmtree(self.output_path, ignore_errors=True)
        self.output_path.mkdir(parents=True)

        last_idx = None
        data_cache = []
        for item in tqdm(dataset, desc='Sampling dataset'):
            if last_idx is None:
                last_idx = item["_class_idx"]

            current_idx = item["_class_idx"]
            if last_idx == current_idx:
                data_cache.append(item)
            else:
                filename = self._grouped_output_filename(last_idx)
                with tarfile.open(filename, mode='a:') as f:
                    for cache_item in data_cache:
                        key = cache_item['__key__']
                        for k in [k for k in cache_item.keys()
                                  if k.startswith('_') or k.startswith('__')]:
                            cache_item.pop(k)
                        for k in sorted(cache_item.keys()):
                            v = cache_item[k]
                            if isinstance(v, str):
                                v = v.encode("utf-8")
                            info = tarfile.TarInfo(key + "." + k)
                            info.size = len(v)
                            info.mtime = time.time()
                            info.mode = 0o0444
                            info.uname = 'combined_ds'
                            info.gname = 'combined_ds'
                            stream = io.BytesIO(v)
                            f.addfile(info, stream)
                last_idx = current_idx
                data_cache = [item]

        out_filename = self.output_path / 'combined_ds_%09d.tar'
        with webdataset.ShardWriter(
                str(out_filename), encoder=False, maxcount=self.shards_max_count) as sink:

            paths = [p for p in self.output_path.iterdir() if p.is_dir()]
            np.random.shuffle(paths)

            for path in tqdm(paths, desc='Sharding dataset'):
                for file in path.iterdir():
                    with open(file, 'rb') as stream:
                        data = stream.read()
                    sink.write({
                        "__key__": file.stem,
                        "tar": data,
                    })
                    file.unlink()
                shutil.rmtree(path)

        return list(self.output_path.iterdir())

    def select_k_random(self, data, read_metadata):
        def is_img(name):
            return Path(name).suffix.lower() in self.images_ext

        assert 'tar' in data, f'{data} is invalid'
        with tarfile.open(fileobj=io.BytesIO(data['tar'])) as tar:
            tar_list = list(tar)
            pickles = {Path(item.name).name: item for item in tar_list}
            images = [item for item in tar_list if is_img(item.name)]

            selected_images = self.intra_class_sampler(images)
            if read_metadata:
                selected_pickles = [pickles[f'{Path(item.name).stem}.pickle'] for item in selected_images]
            else:
                selected_pickles = [None for _ in range(len(selected_images))]

            return [
                {
                    '__key__': f'{os.path.splitext(img_info.name)[0]}',
                    f'{os.path.splitext(img_info.name)[1][1:]}': tar.extractfile(img_info).read(),
                    'pickle': object() if pickle_info is None else tar.extractfile(pickle_info).read()
                }
                for img_info, pickle_info in zip(selected_images, selected_pickles)
            ]

    @property
    def shard_paths(self) -> List[Path]:
        return [p for p in self.output_path.iterdir() if p.suffix == '.tar']
