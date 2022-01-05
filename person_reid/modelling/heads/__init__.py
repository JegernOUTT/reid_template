from person_reid.modelling.heads.conv_head import ConvHead
from person_reid.modelling.heads.embedding_extra_target import EmbeddingExtraTargetHead
from person_reid.modelling.heads.gdc import GDC
from person_reid.modelling.heads.gnap import GNAP
from person_reid.modelling.heads.keypoints_hm_head import KeypointsHmHead
from person_reid.modelling.heads.pool_head import PoolHead, PoolHead as AdaptiveAvgPoolHead
from person_reid.modelling.heads.vargfacenet_head import VarGFaceNetHead

__all__ = ['AdaptiveAvgPoolHead', 'ConvHead', 'EmbeddingExtraTargetHead',
           'GDC', 'GNAP', 'KeypointsHmHead', 'PoolHead', 'VarGFaceNetHead']
