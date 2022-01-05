from person_reid.modelling.backbones.ghostnet import GhostNet
from person_reid.modelling.backbones.hardnet import (HarDNet39Det, HarDNet39sDet, HarDNet68Det,
                                                     HarDNet68sDet, HarDNet85Det, HarDNet85sDet)
from person_reid.modelling.backbones.mobileface import MobileFaceNet
from person_reid.modelling.backbones.proxylessnas import (ProxylessNASCPU, ProxylessNASGPU, ProxylessNASMobile,
                                                          ProxylessNASMobile14)
from person_reid.modelling.backbones.resattnet import (AttentionNetIR56, AttentionNetIRSE56, AttentionNetIR92,
                                                       AttentionNetIRSE92)
from person_reid.modelling.backbones.vargfacenet import VarGFaceNet

__all__ = ['AttentionNetIR56', 'AttentionNetIRSE56', 'AttentionNetIR92', 'AttentionNetIRSE92',
           'GhostNet', 'MobileFaceNet',
           'ProxylessNASCPU', 'ProxylessNASGPU', 'ProxylessNASMobile', 'ProxylessNASMobile14',
           'VarGFaceNet',  'HarDNet39Det', 'HarDNet39sDet', 'HarDNet68Det', 'HarDNet68sDet',
           'HarDNet85Det', 'HarDNet85sDet']
