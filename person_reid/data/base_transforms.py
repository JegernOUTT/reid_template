import random
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
from dssl_dl_utils import Size2D

__all__ = ['make_crop_and_resize']


class LetterPackResize(A.DualTransform):
    def __init__(self,
                 width: int, height: int,
                 interpolations: Tuple[int] = (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR,
                                               cv2.INTER_NEAREST, cv2.INTER_LANCZOS4),
                 always_apply: bool = False,
                 p: float = 1):
        super(LetterPackResize, self).__init__(always_apply, p)
        self.interpolations = interpolations
        self.width, self.height = width, height

    def _calc_resize_size(self, params):
        new_h, new_w = params["rows"], params["cols"]

        if (self.width / new_w) < (self.height / new_h):
            return (new_h * self.width) // new_w, self.width
        else:
            return self.height, (new_w * self.height) // new_h

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        new_h, new_w = self._calc_resize_size(params)
        return A.resize(img, new_h, new_w, random.choice(self.interpolations))

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        new_h, _ = self._calc_resize_size(params)

        scale = new_h / params["rows"]
        return A.keypoint_scale(keypoint, scale, scale)

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def get_transform_init_args_names(self):
        return "width", "height", "interpolation"


def jpg_decoder(data):
    buf = np.ndarray(shape=(1, len(data)), dtype=np.uint8, buffer=data)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def jpeg_decoder(data):
    buf = np.ndarray(shape=(1, len(data)), dtype=np.uint8, buffer=data)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def png_decoder(data):
    buf = np.ndarray(shape=(1, len(data)), dtype=np.uint8, buffer=data)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def make_crop_and_resize(image_size: Size2D, with_keypoints_and_masks: bool = False):
    transform_pipeline = A.Compose(
        [
            A.RandomCropNearBBox(max_part_shift=0, cropping_box_key='cropping_bbox', p=1),
            LetterPackResize(width=image_size.width, height=image_size.height),
            A.PadIfNeeded(min_height=image_size.height, min_width=image_size.width,
                          border_mode=0, value=(127, 127, 127), p=1)
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )

    def expose_metadata(data):
        data['image'] = next((data[suf] for suf in ('jpg', 'jpeg', 'png') if suf in data))
        img_size = Size2D(data['image'].shape[1], data['image'].shape[0])
        if with_keypoints_and_masks:
            data['bbox'] = data['pickle'].bbox.xyxy(img_size)
            data['mask'] = data['pickle'].bitmap_mask.mask
            data['keypoints'] = data['pickle'].keypoint_graph.to_numpy(img_size)
        return data

    def crop_and_resize(data):
        data = expose_metadata(data)

        extra_args = {}
        if with_keypoints_and_masks:
            extra_args['cropping_bbox'] = data.pop('bbox')
            extra_args['mask'] = data.pop('mask')
            extra_args['keypoints'] = data.pop('keypoints')

        image = data.pop('image')
        try:
            augmented = transform_pipeline(image=image, **extra_args)
            augmented['valid'] = True
        except (cv2.error, ZeroDivisionError):
            print(f'Invalid image: {data["__key__"]}, skipping it')
            augmented = data
            augmented.update({'valid': False, 'image': image})
            if with_keypoints_and_masks:
                augmented.update({'mask': extra_args['mask'], 'keypoints': extra_args['keypoints']})

        output = {'image': augmented['image'], 'valid': augmented['valid']}
        if with_keypoints_and_masks:
            output.update({'mask': augmented['mask'], 'keypoints': augmented['keypoints']})
        output.update(data)

        return output

    return crop_and_resize


def make_resize(image_size: Size2D):
    transform_pipeline = A.Compose(
        [
            LetterPackResize(width=image_size.width, height=image_size.height),
            A.PadIfNeeded(min_height=image_size.height, min_width=image_size.width,
                          border_mode=0, value=(127, 127, 127), p=1)
        ]
    )

    def expose_metadata(data):
        data['image'] = next((data[suf] for suf in ('jpg', 'jpeg', 'png') if suf in data))
        return data

    def crop_and_resize(data):
        data = expose_metadata(data)
        image = data.pop('image')
        try:
            augmented = transform_pipeline(image=image)
            augmented['valid'] = True
        except (cv2.error, ZeroDivisionError):
            print(f'Invalid image: {data["__key__"]}, skipping it')
            augmented = data
            augmented.update({'valid': False, 'image': image})

        data['image'] = augmented['image']
        data['valid'] = augmented['valid']
        return data

    return crop_and_resize
