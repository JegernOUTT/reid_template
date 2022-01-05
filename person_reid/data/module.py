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

from person_reid.data.base_transforms import make_crop_and_resize
from person_reid.data.ds_info_extractors import DATASETS_EXTRACTORS, IGNORE_VALUE

__all__ = ['DataModule']


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_paths: List[Path],
                 val_paths: List[Path],
                 image_size: Size2D,
                 sampler: Any,
                 full_resample: True,
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
        self._with_keypoints_and_masks = with_keypoints_and_masks
        self._batch_size = batch_size
        self._loader_kwargs = loader_kwargs
        self._ds_extractors = {
            extractor.name(): extractor
            for extractor in DATASETS_EXTRACTORS.values()
        }

        self._dry_train_dataset = None
        self._train_dataset = None
        self._train_overall_length = None
        self._train_ds_indexes = None
        self._train_indexes_offsets = None

        self._dry_val_dataset = None
        self._val_dataset = None
        self._val_overall_length = None
        self._val_ds_indexes = None
        self._val_indexes_offsets = None

    def prepare_data(self, *args, **kwargs):
        if not self._full_resample:
            print('Skipping full resampling. Be sure that you have correct and fresh training shards')
            return

        new_paths = self._sampler.resample_dataset((
            self._build_ds(self._train_dataset, True, False)
                .map(partial(data._recalculate_indexes, 'train')))
        )
        new_paths = list(map(str, new_paths))
        print(f"Full resampling completed, use these paths in the future if you're not gonna change "
              f"sampling params:\n{new_paths}")
        self._train_paths = new_paths

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
            person_idx_buff, cam_idx_buff = 0, 0
            for idx in sorted(list(ds_indexes)):
                indexes_offsets[idx] = {
                    '_person_idx': person_idx_buff,
                    '_cam_idx': cam_idx_buff
                }
                person_idx_buff += DATASETS_EXTRACTORS[idx].persons_count()
                cam_idx_buff += DATASETS_EXTRACTORS[idx].cameras_count()
            return indexes_offsets

        print('Loading train datasets')
        self._dry_train_dataset = self._build_ds(self._train_paths, False, True)
        self._train_overall_length, self._train_ds_indexes = _retrieve_ds_info(self._dry_train_dataset)
        self._train_indexes_offsets = _calc_indexes_offsets(self._train_ds_indexes)
        print(f'Train datasets overall size: {self._train_overall_length}')

        print('\nLoading validation datasets')
        self._dry_val_dataset = self._build_ds(self._val_paths, True, False)
        self._val_overall_length, self._val_ds_indexes = _retrieve_ds_info(self._dry_val_dataset)
        self._val_indexes_offsets = _calc_indexes_offsets(self._val_ds_indexes)
        print(f'Validation datasets overall size: {self._val_overall_length}')

    def _build_ds(self, paths, add_extra_filters=False, add_sampler_preprocessing=False):
        def _filter(dataset):
            if add_extra_filters:
                return (dataset
                        .select(lambda x: 'jpg' in x or 'png' in x or 'jpeg' in x)
                        .select(lambda x: pickle.loads(x['pickle']).bbox is not None))
            else:
                return dataset

        if add_sampler_preprocessing:
            return (_filter(webdataset.WebDataset(paths)
                            .map(self._sampler.select_k_random)
                            .unlisted())
                    .map(self._expose_metadata))
        else:
            return (_filter(webdataset.WebDataset(paths))
                    .map(self._expose_metadata))

    def _expose_metadata(self, data):
        ds_name = data['__key__'].split('/')[0]
        assert ds_name in self._ds_extractors, f'Unknown dataset: {ds_name}'
        extractor = self._ds_extractors[ds_name]
        extracted = extractor.extract(data)
        assert extracted['_person_idx'] < extractor.persons_count(), f'Check {extractor.__name__} extractor validity'
        assert extracted['_person_idx'] >= 0, f'Check {extractor.__name__} extractor validity'

        assert extracted['_cam_idx'] == IGNORE_VALUE or extracted['_cam_idx'] < extractor.cameras_count(), \
            f'Check {extractor.__name__} extractor validity'
        assert extracted['_cam_idx'] >= 0 or extracted['_cam_idx'] < extractor.cameras_count(), \
            f'Check {extractor.__name__} extractor validity'
        return extracted

    def _recalculate_indexes(self, mode, data):
        offsets = self._train_indexes_offsets if mode == 'train' else self._val_indexes_offsets
        data['_person_idx'] += offsets[data['_dataset_idx']]['_person_idx']
        if data['_cam_idx'] != IGNORE_VALUE:
            data['_cam_idx'] += offsets[data['_dataset_idx']]['_cam_idx']
        return data

    def _format_output(self, mode, data):
        output = {'image': data['image'],
                  'dataset_idx': np.array(data['_dataset_idx'], dtype=np.long),
                  'person_idx': np.array(data['_person_idx'], dtype=np.long),
                  'cam_idx': np.array(data['_cam_idx'], dtype=np.long),
                  'clothes_idx': np.array(data['_clothes_idx'], dtype=np.long)}
        if self._with_keypoints_and_masks:
            output['mask'] = np.array(data['mask'], dtype=np.uint8)
            output['keypoints'] = np.array(data['keypoints'], dtype=np.float32)

        if mode != 'train':
            output['is_query'] = np.array(data['_is_query'], dtype=np.bool)
        return output

    def _make_dataloader(self, mode):
        if mode == 'train':
            overall_length = self._train_overall_length
            dataset = (self._build_ds(self._train_paths, False, True)
                       .shuffle(self._batch_size)
                       .decode("rgb8")
                       .map(partial(self._recalculate_indexes, mode))
                       .map(make_crop_and_resize(self._image_size, self._with_keypoints_and_masks))
                       .map(partial(self._format_output, mode)))
            self._train_dataset = dataset
        else:
            overall_length = self._val_overall_length
            dataset = (self._build_ds(self._val_paths, True, False)
                       .decode("rgb8")
                       .map(self._expose_metadata)
                       .map(partial(self._recalculate_indexes, mode))
                       .map(make_crop_and_resize(self._image_size, self._with_keypoints_and_masks))
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

    def person_categories_count(self, mode):
        ds_indexes = self._train_ds_indexes if mode == 'train' else self._val_ds_indexes
        return sum(DATASETS_EXTRACTORS[idx].persons_count() for idx in ds_indexes)

    def cameras_categories_count(self, mode):
        ds_indexes = self._train_ds_indexes if mode == 'train' else self._val_ds_indexes
        return sum(DATASETS_EXTRACTORS[idx].cameras_count() for idx in ds_indexes)

    def len(self, mode):
        return self._train_overall_length if mode == 'train' else self._val_overall_length


if __name__ == '__main__':
    import cv2

    base_path = Path('/media/svakhreev/fast/person_reid/')
    batch_size = 16
    train_paths = [p for p in (base_path / 'train_shards').iterdir() if p.suffix == '.tar']
    data = DataModule(train_paths=train_paths,
                      val_paths=[base_path / 'TEST/last_test.tar'],
                      sampler=dict(type='PersonSampler', output_path=base_path / 'train_shards'),
                      full_resample=False,
                      image_size=Size2D(128, 256),
                      with_keypoints_and_masks=True,
                      batch_size=batch_size)
    data.prepare_data()
    data.setup()

    print(f'Train person categories count: {data.person_categories_count("train")}')
    print(f'Val person categories count: {data.person_categories_count("val")}')
    print(f'Train camera categories count: {data.cameras_categories_count("train")}')
    print(f'Val camera categories count: {data.cameras_categories_count("val")}\n')

    print(f'Sampling dataset')
    train_data = data.train_dataloader()
    cv2.namedWindow('image', flags=cv2.WINDOW_KEEPRATIO)
    for item in train_data:
        for i in range(batch_size):
            image = cv2.cvtColor(item['image'][i].cpu().numpy(), cv2.COLOR_RGB2BGR)
            for x, y in item['keypoints'][i].cpu().numpy():
                cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), 3)

            mask = item['mask'][i].cpu().numpy()[..., None].repeat(3, 2)
            image = np.hstack([image, mask])

            cv2.imshow('image', image)
            print(f'Dataset index: {DATASETS_EXTRACTORS[item["dataset_idx"][i].item()].__name__}')
            print(f'Person index: {item["person_idx"][i].item()}')
            print(f'Camera index: {item["cam_idx"][i].item()}')
            print(f'Clothes index: {item["clothes_idx"][i].item()}\n\n')
            cv2.waitKey(0)
