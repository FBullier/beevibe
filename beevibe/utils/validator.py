from pydantic import BaseModel, field_validator
from typing import List, Union, Any, Optional


class DatasetConfig(BaseModel):

    texts: Optional[List[str]]
    labels: Optional[List[Any]]

    model_name: Optional[str]
    path: Optional[str]

    num_classes: Optional[int]
    seed: Optional[int] = 1811
    num_epochs:Optional[int] = 20
    batch_size: Optional[int] = 4
    patience: Optional[int] = 3
    max_len: Optional[int] = 128
    n_splits: Optional[int] = 5

    train_size: Optional[float] = 1.0
    val_size: Optional[float] = 0.2
    min_delta: Optional[float] = 0.001

    balanced: Optional[bool] = False

    @field_validator("texts", "labels", mode="before")
    def validate_non_empty_list(cls, value, field):
        if value is not None and not isinstance(value, list):
            raise ValueError(f"{field.name} must be a list.")
        return value

    @field_validator("num_classes", "max_len", "seed", "num_epochs", "batch_size", "patience", "n_splits")
    def validate_int(cls, value, field):
        if value is not None and not value >= 1):
            raise ValueError(f"{field.name} must be upper than 0.")
        return value

    @field_validator("train_size", "val_size")
    def validate_size(cls, value, field):
        if value is not None and not (0.0 < value <= 1.0):
            raise ValueError(f"{field.name} must be upper 0 and lower 1.0")
        return value

    @field_validator("min_delta")
    def validate_delta(cls, value, field):
        if value is not None and not (0.0 < value <= 0.1) :
            raise ValueError(f"{field.name} must be upper 0 and lower 0.1")
        return value
    