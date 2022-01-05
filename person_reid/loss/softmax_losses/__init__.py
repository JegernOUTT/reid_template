from person_reid.loss.softmax_losses.adacos import AdaCos
from person_reid.loss.softmax_losses.airface import AirFace
from person_reid.loss.softmax_losses.am_softmax import AmSoftmax
from person_reid.loss.softmax_losses.arcface import ArcFace
from person_reid.loss.softmax_losses.arcface_easymargin import ArcFaceEasyMargin
from person_reid.loss.softmax_losses.arcnegface import ArcNegFace
from person_reid.loss.softmax_losses.circle import Circle
from person_reid.loss.softmax_losses.cosface import CosFace
from person_reid.loss.softmax_losses.curricular_face import CurricularFace
from person_reid.loss.softmax_losses.margin_softmax import MarginSoftmax
from person_reid.loss.softmax_losses.qam_face import QAMFace
from person_reid.loss.softmax_losses.sphereface import SphereFace, SphereProduct2
from person_reid.loss.softmax_losses.svx_softmax import SVXSoftmax

__all__ = ['AdaCos', 'AirFace', 'AmSoftmax', 'ArcFace', 'ArcFaceEasyMargin', 'ArcNegFace',
           'Circle', 'CosFace', 'CurricularFace', 'QAMFace', 'SphereFace', 'SphereProduct2',
           'SVXSoftmax', 'MarginSoftmax']
