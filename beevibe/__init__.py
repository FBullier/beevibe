"""
BeeVibe: Feel the vibe of building smarter models.

A Python package for training and inference of language models on thematic datasets.
"""

# Expose key modules and classes for easier imports
from .core.trainer import EarlyStopping
from .core.trainer import BeeTrainer
from .core.datasets import TextDatasetMC
from .core.datasets import TextDatasetML
from .core.tokenizers import HFTokenizer
from .core.models import HFMLMClassifier
from .core.models import BeeMLMClassifier
from .core.schedulers import AnnealOnPlateau

#from .ai.openai_interface import AIHelper
#from .ai.openai_interface import OpenAIInterface
#from .utils.logger import setup_logger
#from .utils.config import config

# Package metadata
__version__ = "0.1.0"
__author__ = "François Bullier"
__license__ = "MIT"
__all__ = [
    "EarlyStopping",
    "BeeTrainer",
    "TextDatasetMC",
    "TextDatasetML",
    "HFTokenizer",
    "HFMLMClassifier",
    "BeeMLMClassifier",
    "AnnealOnPlateau"
]