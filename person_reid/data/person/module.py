import json
import pickle
from functools import partial
from pathlib import Path
from typing import List, Union, Optional, Any

import numpy as np
import pytorch_lightning as pl
import webdataset
from dssl_dl_utils import Size2D
from torch.utils.data import DataLoader
from tqdm import tqdm

from person_reid.data.base_transforms import make_crop_and_resize, jpg_decoder, jpeg_decoder, png_decoder
from person_reid.data.person.ds_info_extractors import DATASETS_EXTRACTORS, IGNORE_VALUE, extract_ds_metadata

__all__ = ['PersonDataModule']


class PersonDataModule(pl.LightningDataModule):
    CACHE_FILENAME = 'cache.json'

    def __init__(self,
                 train_paths: List[Path],
                 val_paths: List[Path],
                 image_size: Size2D,
                 sampler: Any,
                 full_resample: bool,
                 batch_shuffle: bool = True,
                 with_keypoints_and_masks: bool = True,
                 batch_size: int = 128,
                 **loader_kwargs):
        super().__init__()
        from person_reid.utils.builders import build_sampler

        assert all(p.exists() for p in train_paths)
        assert all(p.exists() for p in val_paths)
        self._train_paths = list(map(str, train_paths))
        self._val_paths = list(map(str, val_paths))
        self._image_size = image_size
        self._sampler = build_sampler(sampler)
        self._full_resample = full_resample
        self._batch_shuffle = batch_shuffle
        self._with_keypoints_and_masks = with_keypoints_and_masks
        self._batch_size = batch_size
        self._loader_kwargs = loader_kwargs

        self._train_dataset = None
        self._train_overall_length = None
        self._train_ds_indexes = None
        self._train_indexes_offsets = None

        self._val_dataset = None
        self._val_overall_length = None
        self._val_ds_indexes = None
        self._val_indexes_offsets = None

    def setup(self, stage: Optional[str] = None):
        def _retrieve_ds_info(dataset):
            count = 0
            ds_indexes = set()
            for data in tqdm(dataset, desc=f'Initial {stage} dataset loading'):
                ds_indexes.add(data['_dataset_idx'])
                count += 1
            return count, ds_indexes

        def _calc_indexes_offsets(ds_indexes):
            indexes_offsets = {}
            class_idx_buff, cam_idx_buff = 0, 0
            for idx in sorted(list(ds_indexes)):
                indexes_offsets[idx] = {
                    '_class_idx': class_idx_buff,
                    '_cam_idx': cam_idx_buff
                }
                class_idx_buff += DATASETS_EXTRACTORS[idx].class_count()
                cam_idx_buff += DATASETS_EXTRACTORS[idx].cameras_count()
            return indexes_offsets

        if not self._full_resample:
            self._train_paths = list(map(str, self._sampler.shard_paths))

        if self._try_to_load_cache():
            return

        print('Loading train datasets')
        _dry_train_dataset = self._build_ds(self._train_paths, True, not self._full_resample)
        self._train_overall_length, self._train_ds_indexes = _retrieve_ds_info(_dry_train_dataset)
        self._train_indexes_offsets = _calc_indexes_offsets(self._train_ds_indexes)
        print(f'Train datasets overall size: {self._train_overall_length}')

        print('\nLoading validation datasets')
        _dry_val_dataset = self._build_ds(self._val_paths, True, False)
        self._val_overall_length, self._val_ds_indexes = _retrieve_ds_info(_dry_val_dataset)
        self._val_indexes_offsets = _calc_indexes_offsets(self._val_ds_indexes)
        print(f'Validation datasets overall size: {self._val_overall_length}')

        self._save_to_cache()

    def resample(self, *args, **kwargs):
        if not self._full_resample:
            print('Skipping full resampling. Be sure that you have correct and fresh training shards')
            return

        new_paths = self._sampler.resample_dataset((
            self._build_ds(self._train_paths, True, False)
                .map(partial(self._recalculate_indexes, 'train')))
        )
        print(f"Full resampling completed")

        self._train_paths = list(map(str, new_paths))

    def _build_ds(self, paths, add_extra_filters=False, add_sampler_preprocessing=False,
                  **dataset_kwargs):
        def _filter(dataset):
            if add_extra_filters:
                return (dataset
                        .select(lambda x: 'jpg' in x or 'png' in x or 'jpeg' in x)
                        .select(lambda x: pickle.loads(x['pickle']).bbox is not None)
                        .select(lambda x: pickle.loads(x['pickle']).keypoint_graph is not None))
            else:
                return dataset

        if add_sampler_preprocessing:
            return (_filter(webdataset.WebDataset(paths, **dataset_kwargs)
                            .map(self._sampler.select_k_random)
                            .unlisted())
                    .map(self._expose_metadata))
        else:
            return (_filter(webdataset.WebDataset(paths, **dataset_kwargs))
                    .map(self._expose_metadata))

    def _expose_metadata(self, data):
        data.update(extract_ds_metadata(data['__key__']))
        return data

    def _recalculate_indexes(self, mode, data):
        offsets = self._train_indexes_offsets if mode == 'train' else self._val_indexes_offsets
        data['_class_idx'] += offsets[data['_dataset_idx']]['_class_idx']
        if data['_cam_idx'] != IGNORE_VALUE:
            data['_cam_idx'] += offsets[data['_dataset_idx']]['_cam_idx']
        return data

    def _format_output(self, mode, data):
        output = {'image': data['image'],
                  'dataset_idx': np.array(data['_dataset_idx'], dtype=np.long),
                  'class_idx': np.array(data['_class_idx'], dtype=np.long),
                  'cam_idx': np.array(data['_cam_idx'], dtype=np.long),
                  'clothes_idx': np.array(data['_appearance_idx'], dtype=np.long)}
        if self._with_keypoints_and_masks:
            output['mask'] = np.array(data['mask'], dtype=np.uint8)
            output['keypoints'] = np.array(data['keypoints'], dtype=np.float32)

        if mode != 'train':
            output['is_query'] = np.array(data['_is_query'], dtype=np.bool)
        return output

    def _make_dataloader(self, mode):
        if mode == 'train':
            overall_length = self._train_overall_length
            dataset = (self._build_ds(self._train_paths, True, True, shardshuffle=True)
                       .shuffle(self._batch_size if self._batch_shuffle else 0)
                       .decode(webdataset.handle_extension("jpg", jpg_decoder),
                               webdataset.handle_extension("jpeg", jpeg_decoder),
                               webdataset.handle_extension("png", png_decoder))
                       .map(partial(self._recalculate_indexes, mode))
                       .map(make_crop_and_resize(self._image_size, self._with_keypoints_and_masks))
                       .select(lambda x: x['valid'])
                       .map(partial(self._format_output, mode)))
            self._train_dataset = dataset
        else:
            overall_length = self._val_overall_length
            dataset = (self._build_ds(self._val_paths, True, False, shardshuffle=False)
                       .decode(webdataset.handle_extension("jpg", jpg_decoder),
                               webdataset.handle_extension("jpeg", jpeg_decoder),
                               webdataset.handle_extension("png", png_decoder))
                       .map(partial(self._recalculate_indexes, mode))
                       .map(make_crop_and_resize(self._image_size, self._with_keypoints_and_masks))
                       .select(lambda x: x['valid'])
                       .map(partial(self._format_output, mode)))
            self._val_dataset = dataset

        loader = webdataset.WebLoader(
            dataset,
            batch_size=self._batch_size,
            **self._loader_kwargs
        )
        loader.length = overall_length // self._batch_size
        if mode == 'train':
            loader = loader.ddp_equalize(overall_length // self._batch_size)
        return loader

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._make_dataloader('train')

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self._make_dataloader('val')

    def test_dataloader(self):
        return self.val_dataloader()

    def class_categories_count(self, mode):
        ds_indexes = self._train_ds_indexes if mode == 'train' else self._val_ds_indexes
        return sum(DATASETS_EXTRACTORS[idx].class_count() for idx in ds_indexes)

    def cameras_categories_count(self, mode):
        ds_indexes = self._train_ds_indexes if mode == 'train' else self._val_ds_indexes
        return sum(DATASETS_EXTRACTORS[idx].cameras_count() for idx in ds_indexes)

    def len(self, mode):
        return self._train_overall_length if mode == 'train' else self._val_overall_length

    def _try_to_load_cache(self):
        assert len(self._train_paths) > 0
        cache_path = Path(self._train_paths[0]).parent / PersonDataModule.CACHE_FILENAME
        if not cache_path.exists():
            return False

        with open(cache_path, 'r') as f:
            cache = json.load(f)

        print(f'Loading train datasets from cache: {cache_path}')
        self._train_overall_length = cache['train_overall_length']
        self._train_ds_indexes = set(cache['train_ds_indexes'])
        self._train_indexes_offsets = {int(k): v for k, v in cache['train_indexes_offsets'].items()}
        print(f'Train datasets overall size: {self._train_overall_length}')

        print(f'\nLoading validation datasets from cache: {cache_path}')
        self._val_overall_length = cache['val_overall_length']
        self._val_ds_indexes = set(cache['val_ds_indexes'])
        self._val_indexes_offsets = {int(k): v for k, v in cache['val_indexes_offsets'].items()}
        print(f'Validation datasets overall size: {self._val_overall_length}')

        return True

    def _save_to_cache(self):
        cache = {
            'train_overall_length': self._train_overall_length,
            'train_ds_indexes': list(self._train_ds_indexes),
            'train_indexes_offsets': self._train_indexes_offsets,
            'val_overall_length': self._val_overall_length,
            'val_ds_indexes': list(self._val_ds_indexes),
            'val_indexes_offsets': self._val_indexes_offsets,
        }

        assert len(self._train_paths) > 0
        cache_path = Path(self._train_paths[0]).parent / PersonDataModule.CACHE_FILENAME
        assert not cache_path.exists()

        with open(cache_path, 'w') as f:
            json.dump(cache, f)


if __name__ == '__main__':
    import cv2

    base_path = Path('/media/svakhreev/slow1/webface/')
    batch_size = 256
    train_paths = [p for p in base_path.iterdir() if p.suffix == '.tar']
    data = PersonDataModule(train_paths=train_paths,
                            val_paths=[],
                            sampler=dict(type='ClassSampler', output_path=base_path / 'train_shards'),
                            full_resample=False,
                            batch_shuffle=False,
                            image_size=Size2D(112, 112),
                            with_keypoints_and_masks=False,
                            batch_size=batch_size,
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=8)
    data.setup()
    data.resample()

    print(f'Train class categories count: {data.class_categories_count("train")}')
    print(f'Val class categories count: {data.class_categories_count("val")}')
    print(f'Train categories count: {data.cameras_categories_count("train")}')
    print(f'Val camera categories count: {data.cameras_categories_count("val")}\n')

    print(f'Sampling dataset')
    train_data = data.train_dataloader()
    cv2.namedWindow('image', flags=cv2.WINDOW_KEEPRATIO)
    all_categories = set(range(data.class_categories_count("train")))
    for item in tqdm(train_data):
        for i in range(batch_size):
            image = cv2.cvtColor(item['image'][i].cpu().numpy(), cv2.COLOR_RGB2BGR)
            for x, y in item['keypoints'][i].cpu().numpy():
                cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), 3)

            mask = item['mask'][i].cpu().numpy()[..., None].repeat(3, 2)
            image = np.hstack([image, mask])

            cv2.imshow('image', image)
            print(f'Dataset index: {DATASETS_EXTRACTORS[item["dataset_idx"][i].item()].__name__}')
            print(f'Class index: {item["class_idx"][i].item()}')
            print(f'Camera index: {item["cam_idx"][i].item()}')
            print(f'Clothes index: {item["clothes_idx"][i].item()}\n\n')
            cv2.waitKey(1)
