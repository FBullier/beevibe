"""
BeeVibe: Feel the vibe of building smarter models.

A Python package for training and inference of language models on thematic datasets.
"""

# Expose key modules and classes for easier imports
from .core.trainer import EarlyStopping
from .core.trainer import MultiClassTrainer
from .core.datasets import TextDatasetMC
from .core.datasets import TextDatasetML
from .core.models import HFTokenizer
from .core.models import HFModelForClassification
from .core.models import SimpleModel
from .core.models import CustomModel
from .core.schedulers import AnnealOnPlateau

#from .ai.openai_interface import OpenAIInterface
#from .utils.logger import setup_logger
#from .utils.config import config

# Package metadata
__version__ = "0.1.0"
__author__ = "BeeVibe Authors"
__license__ = "MIT"
__all__ = [
    "EarlyStopping",
    "MultiClassTrainer",
    "TextDatasetMC",
    "TextDatasetML",
    "HFTokenizer",
    "HFModelForClassification",
    "SimpleModel",
    "CustomModel",
    "AnnealOnPlateau",
]
