from pathlib import Path
from typing import Union

__all__ = ['IGNORE_VALUE', 'extract_ds_metadata']

IGNORE_VALUE = int(-1e2)


class Market1501:
    IDX = 0

    @staticmethod
    def name():
        return 'market_1501'

    @staticmethod
    def persons_count():
        return 1501

    @staticmethod
    def cameras_count():
        return 6

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx = int(key[0]) - 1
        cam_idx = int(key[1][1])
        return {
            '_dataset_idx': Market1501.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class Last:
    IDX = 1

    @staticmethod
    def name():
        return 'last'

    @staticmethod
    def persons_count():
        return 5000

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].replace('__', '_').split('_')
        person_idx, image_idx, video_idx, frame_idx, bbox_idx, clothes_idx = map(int, key)
        return {
            '_dataset_idx': Last.IDX,
            '_person_idx': person_idx,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': clothes_idx - 1
        }


class LastTest:
    IDX = 2

    @staticmethod
    def name():
        return 'last_test'

    @staticmethod
    def persons_count():
        return 5808

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].replace('__', '_').split('_')
        is_query = 'query' in path
        person_idx, image_idx, video_idx, frame_idx, bbox_idx, clothes_idx = map(int, key)
        return {
            '_dataset_idx': LastTest.IDX,
            '_is_query': is_query,
            '_person_idx': person_idx,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': clothes_idx
        }


class DDDPes:
    IDX = 3

    @staticmethod
    def name():
        return '3dpes'

    @staticmethod
    def persons_count():
        return 204

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        return {
            '_dataset_idx': DDDPes.IDX,
            '_person_idx': int(key[0]) - 1,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': IGNORE_VALUE
        }


class Caviar4Reid:
    IDX = 4

    @staticmethod
    def name():
        return 'caviar4reid'

    @staticmethod
    def persons_count():
        return 72

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1]
        return {
            '_dataset_idx': Caviar4Reid.IDX,
            '_person_idx': int(key[:4]) - 1,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': IGNORE_VALUE
        }


class CUHK03:
    IDX = 5
    CAM_OFFSETS = {
        1: 0,
        2: 843,
        3: 843 + 440,
        4: 843 + 440 + 77,
        5: 843 + 440 + 77 + 58,
    }

    @staticmethod
    def name():
        return 'cuhk_03'

    @staticmethod
    def persons_count():
        return 1467

    @staticmethod
    def cameras_count():
        return 5

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        cam_idx, person_idx = map(int, key[:2])
        person_idx += CUHK03.CAM_OFFSETS[cam_idx]

        return {
            '_dataset_idx': CUHK03.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class DukeMTMC:
    IDX = 6

    @staticmethod
    def name():
        return 'duke_mtmc'

    @staticmethod
    def persons_count():
        return 7140

    @staticmethod
    def cameras_count():
        return 8

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx, cam_idx = int(key[0]), int(key[1][1:])
        return {
            '_dataset_idx': DukeMTMC.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class ETH:
    IDX = 7
    CAM_OFFSETS = {
        1: 0,
        2: 83,
        3: 35,
    }

    @staticmethod
    def name():
        return 'eth'

    @staticmethod
    def persons_count():
        return 146

    @staticmethod
    def cameras_count():
        return 3

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        cam_idx = int(key[-3][3:])
        person_idx = int(key[-2][1:])
        person_idx += ETH.CAM_OFFSETS[cam_idx]
        return {
            '_dataset_idx': ETH.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class FSBReid:
    IDX = 8

    @staticmethod
    def name():
        return 'fsb_reid'

    @staticmethod
    def persons_count():
        return 164

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        return {
            '_dataset_idx': FSBReid.IDX,
            '_person_idx': int(key[-2]),
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': IGNORE_VALUE
        }


class ILIDSVID:
    IDX = 9
    CAM_OFFSETS = {
        1: 0,
        2: 300
    }

    @staticmethod
    def name():
        return 'ilids_vid'

    @staticmethod
    def persons_count():
        return 600

    @staticmethod
    def cameras_count():
        return 2

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        cam_idx = int(key[0][3:])
        person_idx = int(key[1][6:])
        person_idx += ILIDSVID.CAM_OFFSETS[cam_idx]
        return {
            '_dataset_idx': ILIDSVID.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class MSMT17:
    IDX = 10

    @staticmethod
    def name():
        return 'msmt_17'

    @staticmethod
    def persons_count():
        return 4101

    @staticmethod
    def cameras_count():
        return 15

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx, _, cam_idx = map(int, key[:3])
        return {
            '_dataset_idx': MSMT17.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class PKUReid:
    IDX = 11

    @staticmethod
    def name():
        return 'pku_reid'

    @staticmethod
    def persons_count():
        return 114

    @staticmethod
    def cameras_count():
        return 2

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx, cam_idx, _ = map(int, key[:3])

        return {
            '_dataset_idx': PKUReid.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class PRID2011:
    IDX = 12

    @staticmethod
    def name():
        return 'prid_2011'

    @staticmethod
    def persons_count():
        return 749

    @staticmethod
    def cameras_count():
        return 2

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        if key[-3] == 'single_shot':
            person_idx = int(key[-1][7:])
            cam = key[-2]
        else:
            person_idx = int(key[-2][7:])
            cam = key[-3]

        return {
            '_dataset_idx': PRID2011.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': 0 if cam == 'cam_a' else 1,
            '_clothes_idx': IGNORE_VALUE
        }


class QMULILIDS:
    IDX = 13

    @staticmethod
    def name():
        return 'qmul_ilids'

    @staticmethod
    def persons_count():
        return 119

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-1][:4])
        return {
            '_dataset_idx': QMULILIDS.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': IGNORE_VALUE
        }


class Real28:
    IDX = 14

    @staticmethod
    def name():
        return 'Real28'

    @staticmethod
    def persons_count():
        return 28

    @staticmethod
    def cameras_count():
        return 4

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx, cam_idx, clothes_idx, _ = map(int, key)
        return {
            '_dataset_idx': Real28.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': clothes_idx - 1
        }


class VCClothes:
    IDX = 15

    @staticmethod
    def name():
        return 'VC-Clothes'

    @staticmethod
    def persons_count():
        return 512

    @staticmethod
    def cameras_count():
        return 4

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('-')
        person_idx, cam_idx, clothes_idx, _ = map(int, key)
        return {
            '_dataset_idx': VCClothes.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': clothes_idx - 1
        }


class Viper:
    IDX = 16

    @staticmethod
    def name():
        return 'viper'

    @staticmethod
    def persons_count():
        return 632

    @staticmethod
    def cameras_count():
        return 2

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-1].split('_')[0])
        cam_idx = 0 if key[-2] == 'cam_a' else 1

        return {
            '_dataset_idx': Viper.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx,
            '_clothes_idx': IGNORE_VALUE
        }


class Ward:
    IDX = 17

    @staticmethod
    def name():
        return 'ward'

    @staticmethod
    def persons_count():
        return 70

    @staticmethod
    def cameras_count():
        return 3

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-1][:4])
        cam_idx = int(key[-1][4:8])

        return {
            '_dataset_idx': Ward.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class CUHK02:
    IDX = 18
    SCENE_OFFSETS = {
        'P1': 0,
        'P2': 973,
        'P3': 973 + 306,
        'P4': 973 + 306 + 107,
        'P5': 973 + 306 + 107 + 193,
    }

    @staticmethod
    def name():
        return 'cuhk_02'

    @staticmethod
    def persons_count():
        return 1818

    @staticmethod
    def cameras_count():
        return 2

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-1].split('_')[0])
        cam_idx = int(key[-2][3:])
        person_idx += CUHK02.SCENE_OFFSETS[key[-3]]
        if key[-3] in {'P4', 'P5'}:
            person_idx += 1

        return {
            '_dataset_idx': CUHK02.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class RAiD:
    IDX = 19

    @staticmethod
    def name():
        return 'RAiD'

    @staticmethod
    def persons_count():
        return 43

    @staticmethod
    def cameras_count():
        return 4

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx, cam_idx, _ = map(int, key)
        return {
            '_dataset_idx': VCClothes.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class MARS:
    IDX = 20

    @staticmethod
    def name():
        return 'mars'

    @staticmethod
    def persons_count():
        return 1500

    @staticmethod
    def cameras_count():
        return 6

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1]
        person_idx = int(key[:4])
        cam_idx = int(key[5])
        return {
            '_dataset_idx': MARS.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class UnrealPerson:
    IDX = 21
    MODEL_TYPE_OFFSETS = {
        1: 0,
        2: 2800,
        3: 2000
    }
    SCENE_TYPE_OFFSETS = {
        1: 0,  # [1 - 6]
        2: 6,  # [1 - 16]
        3: 16,  # [23 - 28]
        4: 6  # [1 - 6]
    }

    @staticmethod
    def name():
        return 'unreal_person'

    @staticmethod
    def persons_count():
        return 6800

    @staticmethod
    def cameras_count():
        return 34

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        model_type_idx = int(key[-3][10])
        scene_type_idx = int(key[-3][8])
        name_splat = key[-1].split('_')
        person_idx = int(name_splat[0])
        cam_idx = int(name_splat[1][1:])
        person_idx += UnrealPerson.MODEL_TYPE_OFFSETS[model_type_idx]
        cam_idx += UnrealPerson.SCENE_TYPE_OFFSETS[scene_type_idx]
        if scene_type_idx == 3:
            cam_idx -= 23

        return {
            '_dataset_idx': UnrealPerson.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class CUHKSYSU:
    IDX = 22

    @staticmethod
    def name():
        return 'cuhksysu'

    @staticmethod
    def persons_count():
        return 11934

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx = int(key[0][1:])
        return {
            '_dataset_idx': CUHKSYSU.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': IGNORE_VALUE
        }


class GRID:
    IDX = 23

    @staticmethod
    def name():
        return 'grid'

    @staticmethod
    def persons_count():
        return 256

    @staticmethod
    def cameras_count():
        return 2

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx = int(key[0])
        cam_idx = 0 if 'gallery' in path else 1
        return {
            '_dataset_idx': GRID.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx,
            '_clothes_idx': IGNORE_VALUE
        }


class PRAI1581:
    IDX = 24

    @staticmethod
    def name():
        return 'PRAI1581'

    @staticmethod
    def persons_count():
        return 1581

    @staticmethod
    def cameras_count():
        return 0

    @staticmethod
    def extract(path: str):
        key = path.split('/')[-1].split('_')
        person_idx = int(key[0])
        return {
            '_dataset_idx': PRAI1581.IDX,
            '_person_idx': person_idx,
            '_cam_idx': IGNORE_VALUE,
            '_clothes_idx': IGNORE_VALUE
        }


class WildtrackDataset:
    IDX = 25

    @staticmethod
    def name():
        return 'wildtrack_dataset'

    @staticmethod
    def persons_count():
        return 313

    @staticmethod
    def cameras_count():
        return 7

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-2])
        cam_idx = int(key[-1].split('_')[0])
        return {
            '_dataset_idx': WildtrackDataset.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx,
            '_clothes_idx': IGNORE_VALUE
        }


class RPIField:
    IDX = 26

    @staticmethod
    def name():
        return 'RPIfield'

    @staticmethod
    def persons_count():
        return 15107

    @staticmethod
    def cameras_count():
        return 12

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-2].split('_')[0])
        cam_idx = int(key[-3][4:])
        return {
            '_dataset_idx': RPIField.IDX,
            '_person_idx': person_idx - 1,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


class LPW:
    IDX = 27
    PERSON_SCENE_OFFSETS = {
        0: 0,
        1: 756,
        2: 1751
    }
    CAMERA_SCENE_OFFSETS = {
        0: 0,
        1: 3,
        2: 4
    }

    @staticmethod
    def name():
        return 'lpw'

    @staticmethod
    def persons_count():
        return 2731

    @staticmethod
    def cameras_count():
        return 11

    @staticmethod
    def extract(path: str):
        key = path.split('/')
        person_idx = int(key[-2])
        cam_idx = int(key[-3][4:])
        scene_idx = int(key[-4][4:]) - 1

        person_idx += LPW.PERSON_SCENE_OFFSETS[scene_idx]
        cam_idx += LPW.CAMERA_SCENE_OFFSETS[scene_idx]

        return {
            '_dataset_idx': LPW.IDX,
            '_person_idx': person_idx,
            '_cam_idx': cam_idx - 1,
            '_clothes_idx': IGNORE_VALUE
        }


DATASETS_EXTRACTORS = {
    Market1501.IDX: Market1501,
    Last.IDX: Last,
    LastTest.IDX: LastTest,
    DDDPes.IDX: DDDPes,
    Caviar4Reid.IDX: Caviar4Reid,
    CUHK03.IDX: CUHK03,
    DukeMTMC.IDX: DukeMTMC,
    ETH.IDX: ETH,
    FSBReid.IDX: FSBReid,
    ILIDSVID.IDX: ILIDSVID,
    MSMT17.IDX: MSMT17,
    PKUReid.IDX: PKUReid,
    PRID2011.IDX: PRID2011,
    QMULILIDS.IDX: QMULILIDS,
    Real28.IDX: Real28,
    VCClothes.IDX: VCClothes,
    Viper.IDX: Viper,
    Ward.IDX: Ward,
    CUHK02.IDX: CUHK02,
    RAiD.IDX: RAiD,
    MARS.IDX: MARS,
    UnrealPerson.IDX: UnrealPerson,
    CUHKSYSU.IDX: CUHKSYSU,
    GRID.IDX: GRID,
    PRAI1581.IDX: PRAI1581,
    WildtrackDataset.IDX: WildtrackDataset,
    RPIField.IDX: RPIField,
    LPW.IDX: LPW,
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
    assert extracted['_person_idx'] < extractor.persons_count(), f'Check {extractor.__name__} extractor validity'
    assert extracted['_person_idx'] >= 0, f'Check {extractor.__name__} extractor validity'
    assert extracted['_cam_idx'] == IGNORE_VALUE or extracted['_cam_idx'] < extractor.cameras_count(), \
        f'Check {extractor.__name__} extractor validity'
    assert extracted['_cam_idx'] >= 0 or extracted['_cam_idx'] < extractor.cameras_count(), \
        f'Check {extractor.__name__} extractor validity'
    return extracted
