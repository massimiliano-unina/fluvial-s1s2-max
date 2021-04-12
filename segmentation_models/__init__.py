name = "segmentation_models"

from .__version__ import __version__

from .unet import Unet
from .fpn import FPN
from .linknet import Linknet
from .pspnet import PSPNet
from .shallow_CNN import shallow_CNN#, shallow_CNN2
from .nestnet import Nestnet#, shallow_CNN2
from .xnet import Xnet#, shallow_CNN2
# from .resnext import ResNeXt50#, shallow_CNN2
from . import metrics
from . import losses