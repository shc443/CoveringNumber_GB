"""Model architectures for compositionality research."""

from .base_net import BaseNet
from .accordion_net import AccordionNet
from .deep_net import DeepNet
from .shallow_net import ShallowNet

__all__ = ['BaseNet', 'AccordionNet', 'DeepNet', 'ShallowNet']