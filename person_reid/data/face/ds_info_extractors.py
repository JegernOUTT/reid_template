from pathlib import Path
from typing import Union

__all__ = ['IGNORE_VALUE', 'extract_ds_metadata']

IGNORE_VALUE = int(-1e2)


class WebFace42M:
    IDX = 0
    FOLDER_OFFSETS = {
        0: 0,
        1: 205990,
        2: 205990 * 2,
        3: 205990 * 3,
        4: 205990 * 4,
        5: 205990 * 5,
        6: 205990 * 6,
        7: 205990 * 7,
        8: 205990 * 8,
        9: 205990 * 9
    }

    @staticmethod
    def name():
        return 'webface42m'

    @staticmethod
    def class_count():
        return 205990 * 9 + 205996

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-2]
        folder_idx, _, class_idx = map(int, key.split('_'))
        class_idx += WebFace42M.FOLDER_OFFSETS[folder_idx]
        with_mask = 'masked' in path

        return {
            '_dataset_idx': WebFace42M.IDX,
            '_class_idx': class_idx,
            '_cam_idx': IGNORE_VALUE,
            '_with_mask': with_mask,
        }


class Glint360K:
    IDX = 1

    @staticmethod
    def name():
        return 'glint360k'

    @staticmethod
    def class_count():
        return 360232

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        keys = path.split('/')
        class_idx = int(keys[-2].replace('id_', ''))
        with_mask = 'masked' in keys[-1]
        is_query = int(keys[-1].split('_')[1]) % 2 == 0

        return {
            '_dataset_idx': Glint360K.IDX,
            '_class_idx': class_idx,
            '_cam_idx': IGNORE_VALUE,
            '_is_query': is_query,
            '_with_mask': with_mask
        }


DATASETS_EXTRACTORS = {
    WebFace42M.IDX: WebFace42M,
    Glint360K.IDX: Glint360K,
}

DB_EXTRACTORS_PER_DS_NAME = {
    extractor.name(): extractor
    for extractor in DATASETS_EXTRACTORS.values()
}


def extract_ds_metadata(path: Union[Path, str]):
    path = Path(path)
    ds_name = path.parts[0]
    assert ds_name in DB_EXTRACTORS_PER_DS_NAME, f'Unknown dataset: {ds_name}'
    extractor = DB_EXTRACTORS_PER_DS_NAME[ds_name]
    extracted = extractor.extract(str(path))
    assert extracted['_class_idx'] < extractor.class_count(), f'Check {extractor.__name__} extractor validity'
    assert extracted['_class_idx'] >= 0, f'Check {extractor.__name__} extractor validity'
    assert extracted['_cam_idx'] == IGNORE_VALUE or extracted['_cam_idx'] < extractor.cameras_count(), \
        f'Check {extractor.__name__} extractor validity'
    assert extracted['_cam_idx'] >= 0 or extracted['_cam_idx'] < extractor.cameras_count(), \
        f'Check {extractor.__name__} extractor validity'
    return extracted


if __name__ == '__main__':
    def _expose_metadata(data):
        data.update(extract_ds_metadata(data['__key__']))
        return data


    import webdataset
    from tqdm import tqdm

    paths = [str(p) for p in Path('/media/svakhreev/slow1/webface/test/glint360k/').iterdir()]
    dataset = (webdataset.WebDataset(paths)
               .select(lambda x: 'jpg' in x or 'png' in x or 'jpeg' in x)
               .map(_expose_metadata))

    all_categories = set(range(Glint360K.class_count()))
    gathered_categories = set()
    for item in tqdm(dataset):
        gathered_categories.add(item['_class_idx'])

    print(max(gathered_categories))
    print(all_categories.difference(gathered_categories))
