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

__all__ = ['PersonSampler']


def default_intra_person_sampler(images, images_per_person=12, **kwargs):
    from person_reid.data.ds_info_extractors import extract_ds_metadata, IGNORE_VALUE

    images_metadata = [extract_ds_metadata(os.path.splitext(img.name)[0]) for img in images]
    images = [img for img, metadata in zip(images, images_metadata)
              if metadata['_clothes_idx'] in {IGNORE_VALUE, 0}]

    if len(images) < images_per_person:
        selected_images = images
    else:
        selected_images = random.sample(images, k=images_per_person)
    return selected_images


class PersonSampler:
    def __init__(self,
                 output_path: Path,
                 shards_max_count=500,
                 images_ext=('.jpg', '.jpeg', '.png'),
                 intra_person_sampler=default_intra_person_sampler):
        self.output_path = output_path
        self.shards_max_count = shards_max_count
        self.images_ext = set(images_ext)
        self.intra_person_sampler = intra_person_sampler

    def resample_dataset(self, dataset) -> List[Path]:
        shutil.rmtree(self.output_path, ignore_errors=True)
        self.output_path.mkdir(parents=True)

        last_idx = None
        data_cache = []
        for item in tqdm(dataset, desc='Sampling dataset'):
            if last_idx is None:
                last_idx = item["_person_idx"]

            current_idx = item["_person_idx"]
            if last_idx == current_idx:
                data_cache.append(item)
            else:
                filename = self.output_path / f'{last_idx}.tar'
                with tarfile.open(filename, mode='a:') as f:
                    for cache_item in data_cache:
                        key = cache_item['__key__']
                        for k in {'_dataset_idx', '_person_idx', '_cam_idx', '_clothes_idx',
                                  *[k for k in cache_item.keys() if k.startswith('__')]}:
                            cache_item.pop(k)
                        for k in sorted(cache_item.keys()):
                            v = cache_item[k]
                            if isinstance(v, str):
                                v = v.encode("utf-8")
                            info = tarfile.TarInfo(key + "." + k)
                            info.size = len(v)
                            info.mtime = time.time()
                            info.mode = 0o0444
                            info.uname = 'person_reid_ds'
                            info.gname = 'person_reid_ds'
                            stream = io.BytesIO(v)
                            f.addfile(info, stream)
                last_idx = current_idx
                data_cache = [item]

        out_filename = self.output_path / 'reid_ds_%06d.tar'
        with webdataset.ShardWriter(
                str(out_filename), encoder=False, maxcount=self.shards_max_count) as sink:

            filenames = list(self.output_path.iterdir())
            np.random.shuffle(filenames)

            for path in tqdm(filenames, desc='Sharding dataset'):
                if path.name.startswith('reid_ds'):
                    continue

                with open(path, 'rb') as stream:
                    data = stream.read()
                sink.write({
                    "__key__": path.stem,
                    "tar": data,
                })
                path.unlink()

        return list(self.output_path.iterdir())

    def select_k_random(self, data):
        def is_img(name):
            return Path(name).suffix.lower() in self.images_ext

        assert 'tar' in data, f'{data} is invalid'
        with tarfile.open(fileobj=io.BytesIO(data['tar'])) as tar:
            tar_list = list(tar)
            pickles = {Path(item.name).name: item for item in tar_list}
            images = [item for item in tar_list if is_img(item.name)]

            selected_images = self.intra_person_sampler(images)
            selected_pickles = [pickles[f'{Path(item.name).stem}.pickle'] for item in selected_images]

            return [
                {
                    '__key__': f'{os.path.splitext(img_info.name)[0]}',
                    f'{os.path.splitext(img_info.name)[1][1:]}': tar.extractfile(img_info).read(),
                    'pickle': tar.extractfile(pickle_info).read()
                }
                for img_info, pickle_info in zip(selected_images, selected_pickles)
            ]

    @property
    def shard_paths(self) -> List[Path]:
        return [p for p in self.output_path.iterdir() if p.suffix == '.tar']
