from .sam import SAM
from .ig import IG
from .cam import CAM
from .grad_cam import GradCAM
from .grad_cam_plus import GradCAMPlus
from .rise import RISE
from .sg import SG
from .lrp import LRP
from .deeplift import DeepLIFT
from .lime import LIME
from .xrai import XRAI
from .ig_blur import BlurIG
from .ig_guided import GuidedIG
from .sam_naive import SAMNaive

from .traceback import TraceBack
from .traceback_topk import TraceTopk

__method_factory__ = {
    "sam": SAM,
    "sam_naive": SAMNaive,
    "ig": IG,
    "cam": CAM,
    "gradcam": GradCAM,
    "gradcamplus": GradCAMPlus,
    "rise": RISE,
    "sg": SG,
    "lrp": LRP,
    "deeplift": DeepLIFT,
    "lime": LIME,
    "xrai": XRAI,
    "blurig": BlurIG,
    "guidedig": GuidedIG,
    # "traceback": TraceBack,     # TODO: add Bort method
    # "tracetopk": TraceTopk      # TODO: add Bort method
}

def give_method(config, **kwargs):
    xai_name = config.xai_name
    method = __method_factory__[xai_name](**kwargs)
    return method