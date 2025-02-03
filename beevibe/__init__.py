"""
BeeVibe: Feel the vibe of building smarter models.

A Python package for training and inference of language models on thematic datasets.
"""

# Expose key modules and classes for easier imports
from .core.trainer import BeeEarlyStopping
from .core.trainer import BeeTrainer
from .core.datasets import BeeTextDataset
from .core.tokenizers import HFTokenizer
from .core.models import HFMLMClassifier
from .core.models import BeeMLMClassifier
from .core.schedulers import BeeAnnealOnPlateau
from .utils.huggingface import HuggingFaceHub

# Package metadata
__version__ = "0.1.0"
__author__ = "François Bullier"
__license__ = "MIT"
__all__ = [
    "BeeEarlyStopping",
    "BeeTrainer",
    "BeeTextDataset",
    "HFTokenizer",
    "HFMLMClassifier",
    "BeeMLMClassifier",
    "BeeAnnealOnPlateau",
    "HuggingFaceHub"
]
