"""
MACE Model Deviation Calculator

A standalone package for calculating ensemble uncertainty from MACE models.
Follows DeepMD methodology and integrates seamlessly with AI2Kit workflows.
"""

__version__ = "0.1.0"
__author__ = "Xiliang LIAN"

from .core import calculate_mace_model_deviation
from .utils import load_mace_models, read_trajectory, write_model_deviation

__all__ = [
    "calculate_mace_model_deviation",
    "load_mace_models", 
    "read_trajectory",
    "write_model_deviation",
]