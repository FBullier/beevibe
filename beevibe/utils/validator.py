import os
import torch
from numpy import ndarray
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from typing import List, Any, Optional, Union

class ValidateParams(BaseModel):

    texts: Optional[List[str]] = None
    labels: Optional[List[Any]] = None
    val_texts: Optional[List[str]] = None
    val_labels: Optional[List[Any]] = None

    raw_texts: Optional[List[str]] = None
    return_probabilities: Optional[bool] = False
    num_workers: Optional[int] = None
    threshold: Optional[float] = 0.0

    labels_names: Optional[Union[List[str], ndarray]] = None
    lr: Optional[float] = 1e-5
    multilabel: Optional[bool] = False
    device: Optional[str] = "cpu"
    verbose: Optional[bool] = False

    model_name: Optional[str] = None
    path: Optional[str] = None
    save_path: Optional[str] = None

    num_labels: Optional[int] = None
    seed: Optional[int] = 1811
    num_epochs: Optional[int] = 20
    batch_size: Optional[int] = 4
    patience: Optional[int] = 3
    max_len: Optional[int] = 128
    n_splits: Optional[int] = 5
    loss_threshold: Optional[float] = 0.0

    train_size: Optional[float] = 1.0
    val_size: Optional[float] = 0.2
    min_delta: Optional[float] = 0.001

    balanced: Optional[bool] = False

    use_lora: Optional[bool] = False
    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 32
    lora_dropout: Optional[float] = 0.1

    quantization_type: Optional[str] = None
    compute_dtype: Optional[torch.dtype] = torch.float16
    quant_type: Optional[str] = "nf4"
    enable_dynamic: Optional[bool] = False
    use_double_quant: Optional[bool] = False

    @field_validator("labels", "val_labels", mode="before")
    def validate_labels(cls, value):
        """Validate the list of integers or list of integers."""
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("The field must be a list.")
            for item in value:
                if isinstance(item, int):
                    continue
                elif isinstance(item, list):
                    if not all(isinstance(sub_item, int) for sub_item in item):
                        raise TypeError("All elements in the nested list must be integers.")
                else:
                    raise TypeError("Each element must be an integer or a list of integers.")
        return value


    @field_validator("texts", "val_texts", "raw_texts", mode="before")
    def validate_texts(cls, value, info):
        """Validate the list of texts field."""
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"{info.field_name} must be a list.")
            if not all(isinstance(item, str) for item in value):
                raise TypeError(f"All elements in {info.field_name} must be strings.")
        return value

    @field_validator("labels_names", mode="before")
    def validate_labels_names(cls, value, info):
        """Validate the list or array of class names."""
        if value is not None:
            if not isinstance(value, list) and not isinstance(value, ndarray):
                raise TypeError(f"{info.field_name} must be a list or a numpy array.")
            if not all(isinstance(item, str) for item in value):
                raise TypeError(f"All elements in {info.field_name} must be strings.")
        return value

    @field_validator("seed", "num_labels", "num_epochs", "batch_size", "patience", "max_len", "n_splits", "num_workers", mode="before")
    def validate_positive_integers(cls, value, info):
        """Validate integer fields to ensure they are positive and >= 0."""
        if value is not None:
            if not isinstance(value, int):
                raise TypeError(f"{info.field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{info.field_name} cannot be less than 0.")
        return value

    @field_validator("train_size", "val_size", "min_delta", "lr", "loss_threshold", "threshold", mode="before")
    def validate_floats(cls, value, info):
        """Validate float fields."""
        if value is not None:
            if not isinstance(value, float):
                raise TypeError(f"{info.field_name} must be a float.")
            if info.field_name in {"train_size", "val_size"} and not (0.0 < value <= 1.0):
                raise ValueError(f"{info.field_name} must be between 0 and 1.0.")
            if info.field_name == "min_delta" and not (0.0 < value <= 0.1):
                raise ValueError(f"{info.field_name} must be between 0 and 0.1.")
            if info.field_name == "lr" and not (0.0 < value):
                raise ValueError(f"{info.field_name} must be upper than 0.")
            if info.field_name == "loss_threshold" and not (0.0 < value):
                raise ValueError(f"{info.field_name} must be upper than 0.")
            if info.field_name == "threshold" and not (0.0 < value <= 0.1):
                raise ValueError(f"{info.field_name} must be between 0 and 0.1.")

        return value

    @field_validator("balanced", "multilabel", "verbose", "return_probabilities", mode="before")
    def validate_boolean(cls, value, info):
        """Validate the boolean fields."""
        if value is  None :
            raise TypeError(f"{info.field_name} is a boolean and must not be None.")
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"{info.field_name} must be a boolean.")
        return value

    @field_validator("path", mode="before")
    def validate_path(cls, value):
        """Validate the path field."""
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("path must be a string.")
            if not os.path.exists(value):
                raise ValueError(f"The path '{value}' does not exist.")
            if not (os.path.isfile(value) or os.path.isdir(value)):
                raise ValueError(f"The path '{value}' is neither a valid file nor a directory.")
        return value

    @field_validator("save_path", mode="before")
    def validate_save_path(cls, value):
        """Validate the save_path field."""
        if value is not None:
            if not isinstance(value, str):
                raise TypeError("path must be a string.")
        else:
            raise TypeError("path must be defined.")
        return value

    @field_validator("device", mode="before")
    def validate_device(cls, value):
        if value not in ["cuda", "cpu", None]:
            raise ValueError(f"The device must be 'cpu' or 'cuda'. Invalid value: {value}")
        return value

    @model_validator(mode="after")
    def validate_lora_params(cls, values):
        """Ensure LoRA parameters are correctly defined based on `use_lora`."""
        if values.use_lora is None:
            raise TypeError("use_lora is a boolean and must not be None.")
        if values.use_lora:
            if values.lora_r is None or values.lora_r <= 0:
                raise ValueError("lora_r must be a positive integer when use_lora is True.")
            if values.lora_alpha is None or values.lora_alpha <= 0:
                raise ValueError("lora_alpha must be a positive integer when use_lora is True.")
            if values.lora_dropout is None or not (0.0 <= values.lora_dropout <= 1.0):
                raise ValueError("lora_dropout must be between 0.0 and 1.0 when use_lora is True.")
        return values

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
