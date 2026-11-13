"""Method registry for the TOPMAX delivery package."""

from .base import BaseMethod, AVAILABLE_METHODS, get_method, list_methods, register_method
from .topmax_cpu import TOPMAX
from .topmax_gpu import TOPMAXGPU

__all__ = [
    "BaseMethod",
    "AVAILABLE_METHODS",
    "get_method",
    "list_methods",
    "register_method",
    "TOPMAX",
    "TOPMAXGPU",
]
