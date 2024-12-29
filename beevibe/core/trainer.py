import random
import pandas as pd
import numpy as np
import time
import gc

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

from beevibe.core.datasets import TextDatasetML, TextDatasetMC
from beevibe.core.models import HFTokenizer, HFModelForClassification
from beevibe.core.earlystopping import EarlyStopping
from beevibe.utils.logger import setup_logger
from beevibe.utils.validator import DatasetConfig

from typing import List, Optional, Tuple, Callable, Any, Dict

from peft import LoraConfig, TaskType
from peft import get_peft_model

class MultiClassTrainer:
    """
    A class for training multi-class or multi-label classification models with customizable
    model, optimizer, and scheduler creators. Provides functionality to manage training
    parameters, logging, and random seeds.
    """
    def __init__(
        self,
        num_classes: int = 3,
        classes_names: List[str] = [],
        model_name: str = "camembert-base",
        max_len: int = 128,
        lr: float = 1e-5,
        multilabel: bool = False,
        device: Optional[str] = None,
        model_creator: Optional[Callable[[str, int], nn.Module]] = None,
        optimizer_creator: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        scheduler_creator: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        scheduler_needs_loss: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the MultiClassTrainer with the specified parameters.

        Args:
            num_classes (int): Number of classes for classification.
            classes_names (List[str]): Names of the classes. If empty, generates default names.
            model_name (str): Name of the model to use (e.g., Hugging Face model name).
            max_len (int): Maximum token length for the tokenizer.
            lr (float): Learning rate for the optimizer.
            multilabel (bool): Whether the task is multi-label classification.
            device (Optional[str]): Device to use (e.g., 'cuda' or 'cpu'). If None, defaults to CUDA if available.
            model_creator (Optional[Callable]): Custom function to create the model.
            optimizer_creator (Optional[Callable]): Custom function to create the optimizer.
            optimizer_params (Optional[Dict[str, Any]]): Parameters for the optimizer.
            scheduler_creator (Optional[Callable]): Custom function to create the scheduler.
            scheduler_params (Optional[Dict[str, Any]]): Parameters for the scheduler.
            scheduler_needs_loss (bool): Whether the scheduler needs loss information.
            verbose (bool): If True, enables verbose logging.
        """

        _ = DatasetConfig(num_classes=num_classes, 
                          classes_names=classes_names,
                          model_name=model_name,
                          max_len=max_len,
                          lr=lr,
                          multilabel=multilabel,
                          device=device,
                          verbose=verbose,
                          use_lora=use_lora,
                          lora_r=lora_r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout
                          )

        self.model_name = model_name
        self.num_classes = num_classes
        self.max_len = max_len
        self.lr = lr
        self.multilabel = multilabel

        self.use_lora = use_lora

        if not self.use_lora:
            self.lora_r = None
            self.lora_alpha = None
            self.lora_dropout = None

        self.verbose = verbose

        if len(classes_names) == 0:
            self.classes_names = [f"Class {i}" for i in range(num_classes)]
        else:
            self.classes_names = classes_names
            self.num_classes = len(classes_names)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_creator = (
            model_creator if model_creator is not None else self.default_model_creator
        )
        self.optimizer_creator = (
            optimizer_creator
            if optimizer_creator is not None
            else self.default_optimizer_creator
        )
        self.scheduler_creator = (
            scheduler_creator
            if scheduler_creator is not None
            else self.default_scheduler_creator
        )
        self.scheduler_params = scheduler_params if scheduler_params else {}
        self.optimizer_params = optimizer_params if optimizer_params else {"lr": lr}

        self.scheduler_needs_loss = scheduler_needs_loss

        self.logger = self.__init_logger()

        self.logger_info(f"Device : {self.device}")

    def logger_info(self, message: str) -> None:
        """
        Log a message if verbose logging is enabled.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            self.logger.info(message)

    def default_model_creator(self, model_name: str, num_classes: int) -> nn.Module:
        """
        Default function to create a classification model using a pre-trained Hugging Face model.

        Args:
            model_name (str): Name of the pre-trained model.
            num_classes (int): Number of output classes.

        Returns:
            nn.Module: A classification model.
        """
        return HFModelForClassification().from_pretrained(
            model_name, num_labels=num_classes
        )

    def default_optimizer_creator(self, model: nn.Module, **params: Any) -> torch.optim.Optimizer:
        """
        Default function to create an Adam optimizer.

        Args:
            model (nn.Module): The model for which to create the optimizer.
            **params: Additional parameters for the optimizer.

        Returns:
            torch.optim.Optimizer: An Adam optimizer.
        """
        self.logger_info("Call default optimizer :")
        self.logger_info(f" - lr:{params.get('lr', 1e-5)}")

        return Adam(model.parameters(), lr=params.get("lr", 1e-5))

    def default_scheduler_creator(self, optimizer: torch.optim.Optimizer, **params: Any) -> Any:
        """
        Default function to create a learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to create the scheduler.
            **params: Additional parameters for the scheduler.

        Returns:
            Any: A ReduceLROnPlateau scheduler.
        """
        self.logger_info("Call default scheduler :")
        self.logger_info(f" - mode:{params.get('mode', 'min')}")
        self.logger_info(f" - factor:{params.get('factor', 0.8)}")
        self.logger_info(f" - patience:{params.get('patience', 2)}")

        return ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.8),
            patience=params.get("patience", 2),
        )

    def __find_target_modules(self, model):
        target_modules = []
        for name, module in model.named_modules():
            if any(keyword in name for keyword in ["query", "key", "dense"]):
                target_modules.append(name)
        return target_modules

    def __print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        formatted_string = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        self.logger_info(formatted_string)

    def __init_model(self) -> None:
        """
        Initialize the model, optimizer, and scheduler for training.
        """
        self.model = self.model_creator(self.model_name, self.num_classes)
        self.model = self.model.to(self.device)

        if self.use_lora:

            self.logger_info("Using Lora")

            # Get Lora targets from current model
            target_modules = self.__find_target_modules(self.model)
            if len(target_modules) == 0:
                self.release_model()
                raise ValueError("Lora : no Target modules found in current model")
            else:    
                self.logger_info("Target modules : ", target_modules)

            # Set Lora configuration
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, 
                r=self.lora_r,                        # Rank of low-rank decomposition
                lora_alpha=self.lora_alpha,              # Scaling factor
                target_modules=target_modules,  # Target modules for LoRA
                lora_dropout=self.lora_dropout,           # Dropout for regularization
                bias="none"                 # Do not fine-tune biases
            )

            # Get Peft model from Lora configuration
            self.model = get_peft_model(self.model, lora_config)

            # Add classifier head to train
            for name, param in self.model.named_parameters():
                if "classifier" in name:  # Matches all layers under the classifier head
                    param.requires_grad = True

            # Display trainable parameters counts
            self.__print_trainable_parameters(self.model)

        # Plug Optimizer function
        self.optimizer = self.optimizer_creator(self.model, **self.optimizer_params)

        # Plug Scheduler function
        if self.scheduler_creator:
            self.logger_info("Use scheduler")
            self.scheduler = self.scheduler_creator(
                self.optimizer, **self.scheduler_params
            )
        else:
            self.logger_info("Don't use scheduler")
            self.scheduler = None

    def __init_logger(self) -> Any:
        """
        Initialize the logger for the trainer.

        Returns:
            Any: The initialized logger.
        """
        return setup_logger()

    def __init_tokenizer(self) -> None:
        """
        Initialize the tokenizer using a pre-trained Hugging Face tokenizer.
        """
        self.tokenizer = HFTokenizer().from_pretrained(
            self.model_name, clean_up_tokenization_spaces=True
        )

    def set_seed(self, seed: int = 1811) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """

        _ = DatasetConfig(
            seed=seed
            )
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int = 20,
        class_weights: Optional[torch.Tensor] = None,
        patience: int = 3,
        min_delta: float = 0.001,
        seed: int = 1811,
    ) -> Dict[str, Any]:
        """
        Train the model using the provided training and validation data loaders.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (Optional[DataLoader]): DataLoader for the validation data. Can be None.
            num_epochs (int): Number of epochs to train the model.
            class_weights (Optional[torch.Tensor]): Class weights for loss computation.
            patience (int): Early stopping patience.
            min_delta (float): Minimum improvement in validation loss to reset patience.
            seed (int): Random seed for reproducibility.

        Returns:
            Dict[str, Any]: A dictionary containing training and validation metrics, losses, and other details.
        """
        start_training_time = time.time()

        self.set_seed(seed)

        self.__init_model()

        if val_loader is not None:
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        val_losses = []
        train_losses = []

        val_metrics = []
        # train_metrics = []
        epochs_duration = []
        val_preds = []
        val_labels = []

        lr_values = []

        if self.multilabel:
            loss_fn = (
                nn.BCEWithLogitsLoss(weight=class_weights)
                if class_weights is not None
                else nn.BCEWithLogitsLoss()
            )
        else:
            loss_fn = (
                nn.CrossEntropyLoss(weight=class_weights)
                if class_weights is not None
                else nn.CrossEntropyLoss()
            )

        for epoch in range(num_epochs):
            start_epoch_time = time.time()
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                self.optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                if self.multilabel:
                    loss = loss_fn(
                        outputs.logits, labels.float()
                    )  # Labels must be float for BCEWithLogitsLoss
                else:
                    loss = loss_fn(outputs.logits, labels)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_values.append(current_lr)

            # Validation
            # val_accuracy = 0
            if val_loader is not None:
                self.model.eval()

                val_loss, res_metrics, res_preds, res_labels = self.__validation(
                    val_loader
                )

                val_losses.append(val_loss)
                train_losses.append(total_loss / len(train_loader))
                val_metrics.append(res_metrics)
                val_preds.append(res_preds)
                val_labels.append(res_labels)

                if self.multilabel:
                    self.logger_info(
                        f"Epoch {epoch}/{num_epochs-1}, Train Loss: {total_loss / len(train_loader):.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val F1-Micro: {res_metrics.get('f1_micro', 'N/A'):.4f}, lr: {current_lr:.3e}"
                    )
                else:
                    self.logger_info(
                        f"Epoch {epoch}/{num_epochs-1}, Train Loss: {total_loss / len(train_loader):.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val MCC: {res_metrics.get('mcc', 'N/A'):.4f}, lr: {current_lr:.3e}"
                    )

                if early_stopping.should_stop(val_loss):
                    best_epoch = epoch - early_stopping.counter
                    best_loss = early_stopping.best_loss
                    break
                else:
                    best_epoch = epoch - early_stopping.counter
                    best_loss = early_stopping.best_loss

            else:
                best_epoch = -1
                best_loss = -1
                train_losses.append(total_loss / len(train_loader))
                self.logger_info(
                    f"Epoch {epoch}/{num_epochs-1}, Training Loss: {total_loss/len(train_loader):.4f}"
                )

            if self.scheduler_needs_loss:
                if best_epoch > -1:
                    self.scheduler.step(best_loss)
                else:
                    self.logger_info(
                        "ReduceLROnPlateau scheduler can't be used without validation losses"
                    )
            else:
                self.scheduler.step()

            end_epoch_time = time.time()
            epochs_duration.append(end_epoch_time - start_epoch_time)

        if best_epoch > -1:
            self.logger_info(f"Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")

        end_training_time = time.time()

        ret = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "max_len": self.max_len,
            "lr": self.lr,
            "device": self.device,
            "num_epochs": num_epochs,
            "class_weights": class_weights,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "seed": seed,
            "scheduler_params": self.scheduler_params,
            "optimizer_params": self.optimizer_params,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
            "val_preds": val_preds,
            "val_labels": val_labels,
            "lr_values": lr_values,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "training_duration": end_training_time - start_training_time,
            "epochs_duration": epochs_duration,
        }

        return ret

    def __calculate_metrics(self, y_true: List[Any], y_pred: List[Any]) -> Dict[str, Any]:
        """
        Calculate metrics for the provided true and predicted labels.

        Args:
            y_true (List[Any]): Ground truth labels.
            y_pred (List[Any]): Predicted labels.

        Returns:
            Dict[str, Any]: A dictionary containing computed metrics such as precision, recall, F1 score, accuracy, and confusion matrix.
        """
        res = {}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if self.multilabel:
            # For multi-label: Compute per-class metrics
            prfs = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            res["class_precision"] = prfs[0]
            res["class_recall"] = prfs[1]
            res["class_f1"] = prfs[2]
            res["class_support"] = prfs[3]

            # Global metrics for multi-label
            res["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
            res["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
            res["f1_weighted"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # Example-based accuracy (All labels must match for a correct prediction)
            accuracy_per_example = np.mean(np.all(y_true == y_pred, axis=1))
            res["accuracy"] = accuracy_per_example

            # Confusion matrix per class (each label treated as binary classification)
            res["confusion_matrix"] = [
                confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
                for i in range(self.num_classes)
            ]

        else:
            # For multi-class: Compute standard metrics
            prfs = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            res["class_precision"] = prfs[0]
            res["class_recall"] = prfs[1]
            res["class_f1"] = prfs[2]
            res["class_support"] = prfs[3]

            # Global metrics for multi-class
            res["f1_macro"] = f1_score(y_true, y_pred, average="macro")
            res["f1_micro"] = f1_score(y_true, y_pred, average="micro")
            res["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
            res["accuracy"] = accuracy_score(y_true, y_pred)
            res["mcc"] = matthews_corrcoef(y_true, y_pred)

            # Standard confusion matrix
            res["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        return res

    def __print_metrics(self, ret: Dict[str, Any]) -> None:
        """
        Print global and per-class metrics, including confusion matrices.

        Args:
            ret (Dict[str, Any]): Dictionary containing validation metrics and other training results.
        """
        best_epoch = ret.get("best_epoch")
        if best_epoch is not None:
            current_dict = ret["val_metrics"][best_epoch]
        else:
            current_dict = ret["val_metrics"]

        self.logger_info("** Global metrics :\n")

        lib_glob_metrics = ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]
        if not self.multilabel:
            lib_glob_metrics.append("mcc")  # Only relevant for multi-class

        for metric in lib_glob_metrics:
            self.logger_info(f" - {metric}: {current_dict[metric]:.4f}")

        self.logger_info("\n")

        self.logger_info("** Per-Classes metrics :\n")
        keys_to_extract = [
            "class_precision",
            "class_recall",
            "class_f1",
            "class_support",
        ]
        extracted_dict = {key: current_dict[key] for key in keys_to_extract}

        class_labels = self.classes_names
        metrics = ["precision", "recall", "f1", "support"]

        # Définir les largeurs de colonnes fixes
        class_width = (
            max(len(c) for c in class_labels) + 5
        )  # Ajuster en fonction du nom de classe le plus long
        metric_width = 12

        # Display header
        header = f"{'Class':<{class_width}}" + "".join(
            [f"{metric.capitalize():>{metric_width}}" for metric in metrics]
        )
        self.logger_info(header)

        for i, class_label in enumerate(class_labels):
            row_values = [
                f"{extracted_dict[f'class_{metric}'][i]:.4f}"
                if metric != "support"
                else f"{extracted_dict[f'class_{metric}'][i]:.0f}"
                for metric in metrics
            ]
            row = f"{class_label:<{class_width}}" + "".join(
                [f"{value:>{metric_width}}" for value in row_values]
            )
            self.logger_info(row)

        self.logger_info("\n")

        self.logger_info("** Confusion Matrix (FN/Row - FP/Col):\n")

        if self.multilabel:
            # For multi-label, print confusion matrices per class
            for i, class_label in enumerate(class_labels):
                self.logger_info(f"Confusion Matrix for {class_label}:")
                conf_matrix = current_dict["confusion_matrix"][i]
                self.logger_info(f"  {conf_matrix[0]}")  # Row for negatives
                self.logger_info(f"  {conf_matrix[1]}")  # Row for positives
                self.logger_info("")
        else:
            # For multi-class, print a single confusion matrix

            # Calculer la largeur des colonnes
            conf_matrix = current_dict["confusion_matrix"]
            max_class_name_length = max(len(name) for name in class_labels)
            num_digits = max(len(str(elem)) for row in conf_matrix for elem in row)
            col_width = (
                max(max_class_name_length, num_digits) + 2
            )  # +2 pour l'espacement

            # Créer une ligne d'en-tête pour les colonnes
            header = " " * (col_width + 2) + "".join(
                f"{name:^{col_width}}" for name in class_labels
            )
            self.logger_info(header)

            # Afficher chaque ligne de la matrice
            for i, row in enumerate(conf_matrix):
                row_str = f"{class_labels[i]:<{col_width}}"  # Alignement à gauche pour les noms de classes
                row_str += "".join(
                    f"{elem:^{col_width}}" for elem in row
                )  # Alignement centré pour les valeurs
                self.logger_info(row_str)

    def display_losses(self, ret: Dict[str, Any]) -> None:
        """
        Plot training and validation losses over epochs.

        Args:
            ret (Dict[str, Any]): Dictionary containing training and validation losses.
        """

        train_losses = ret.get("train_losses")
        val_losses = ret.get("val_losses")

        epochs = list(range(0, len(train_losses)))

        data = pd.DataFrame(
            {
                "Epoch": epochs * 2,
                "Loss": train_losses + val_losses,
                "Type": ["Train"] * len(train_losses)
                + ["Validation"] * len(val_losses),
            }
        )

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        sns.lineplot(
            x="Epoch", y="Loss", hue="Type", data=data, marker="o", linewidth=2.5
        )

        plt.title("Training and Validation Loss per Epoch", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)

        plt.legend(title="Loss Type", fontsize=12)
        plt.xticks(epochs)

        plt.show()

    def __validation(self, val_loader: DataLoader) -> Tuple[float, Dict[str, Any], List[Any], List[Any]]:
        """
        Perform validation for the provided data loader.

        Args:
            val_loader (DataLoader): DataLoader for the validation data.

        Returns:
            Tuple[float, Dict[str, Any], List[Any], List[Any]]: Validation loss, metrics, predictions, and true labels.
        """
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        if self.multilabel:
            loss_fn = nn.BCEWithLogitsLoss()  # For multi-label classification
        else:
            loss_fn = nn.CrossEntropyLoss()  # For multi-class classification

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                if self.multilabel:
                    loss = loss_fn(
                        outputs.logits, labels.float()
                    )  # Labels need to be float for BCEWithLogitsLoss
                else:
                    loss = loss_fn(outputs.logits, labels)

                val_loss += loss.item()

                if self.multilabel:
                    preds = (
                        torch.sigmoid(outputs.logits).cpu().numpy()
                    )  # Apply sigmoid for multi-label
                    preds = (preds > 0.5).astype(
                        int
                    )  # Apply a threshold to make binary predictions
                else:
                    preds = (
                        torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    )  # For multi-class, use argmax

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        res_metrics = self.__calculate_metrics(all_labels, all_preds)

        return val_loss, res_metrics, all_preds, all_labels

    def compute_class_weights(self, labels: List[Any], num_classes: int) -> torch.Tensor:
        """
        Compute class weights for balancing the loss function.

        Args:
            train_labels (List[Any]): Training labels.
            num_classes (int): Number of classes.

        Returns:
            torch.Tensor: Tensor containing class weights.
        """
        _ = DatasetConfig(
            labels=labels,
            num_classes=num_classes
            )

        if not self.multilabel:
            # Multi-class case
            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.arange(num_classes), y=labels
            )
            return torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Multi-label case
        train_labels = np.array(labels)
        num_samples = train_labels.shape[0]
        class_weights = []

        for label_idx in range(num_classes):
            num_positives = np.sum(train_labels[:, label_idx])
            if num_positives > 0:
                class_weight = num_samples / (2.0 * num_positives)
            else:
                class_weight = 0.0
            class_weights.append(class_weight)

        self.logger_info(f"class_weights: {class_weights}")

        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

    def train(
        self,
        texts: List[str],
        labels: List[Any],
        train_size: float = 1.0,
        num_epochs: int = 20,
        batch_size: int = 4,
        balanced: bool = False,
        patience: int = 3,
        min_delta: float = 0.001,
        seed: int = 1811,
    ) -> Dict[str, Any]:
        """
        Train the model using the provided texts and labels.

        Args:
            texts (List[str]): List of input text samples.
            labels (List[Any]): Corresponding labels for the input texts.
            train_size (float): Proportion of data to use for training (default: 1.0).
            num_epochs (int): Number of training epochs (default: 20).
            batch_size (int): Batch size for training (default: 4).
            balanced (bool): Whether to compute class weights for balancing (default: False).
            patience (int): Early stopping patience (default: 3).
            min_delta (float): Minimum improvement in validation loss to reset patience (default: 0.001).
            seed (int): Random seed for reproducibility (default: 1811).

        Returns:
            Dict[str, Any]: Dictionary containing training results, including losses and metrics.
        """

        _ = DatasetConfig(
            texts=texts, 
            labels=labels,
            train_size=train_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            balanced=balanced,
            patience=patience,
            min_delta=min_delta,
            seed=seed
            )

        self.__init_tokenizer()

        if train_size == 1.0:
            train_texts = texts
            train_labels = labels
        else:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts,
                labels,
                test_size=(1.0 - train_size),
                shuffle=True,
                random_state=seed,
            )

        if balanced:
            class_weights = self.compute_class_weights(train_labels, self.num_classes)
        else:
            class_weights = None

        if self.multilabel:
            train_dataset = TextDatasetML(
                train_texts, train_labels, self.tokenizer, self.max_len
            )
        else:
            train_dataset = TextDatasetMC(
                train_texts, train_labels, self.tokenizer, self.max_len
            )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

        ret = self.__train(
            train_loader=train_loader,
            val_loader=None,
            num_epochs=num_epochs,
            patience=patience,
            min_delta=min_delta,
            class_weights=class_weights,
            seed=seed,
        )

        return ret

    def save(self, path: str) -> None:
        """
        Save the trained model and tokenizer to the specified path.

        Args:
            path (str): Directory path where the model and tokenizer will be saved.
        """

        _ = DatasetConfig(
            path=path
            )

        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        """
        Load a pre-trained model and tokenizer from the specified path.

        Args:
            path (str): Directory path where the model and tokenizer are saved.
        """
        _ = DatasetConfig(
            path=path
            )

        self.tokenizer = HFTokenizer().from_pretrained(path)
        self.model = HFModelForClassification().from_pretrained(path)

    def __preprocess(self, raw_reviews: List[str], max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess input text samples for the model.

        Args:
            raw_reviews (List[str]): List of raw text inputs.
            max_len (int): Maximum sequence length for tokenization (default: 128).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tokenized input IDs and attention masks.
        """

        encoded_batch = self.tokenizer(
            raw_reviews,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return encoded_batch["input_ids"], encoded_batch["attention_mask"]

    def predict(self, texts: List[str], max_len: int = 128) -> List[Any]:
        """
        Perform inference on a list of text inputs.

        Args:
            texts (List[str]): List of input text samples.
            max_len (int): Maximum sequence length for tokenization (default: 128).

        Returns:
            List[Any]: Predicted labels or probabilities.
        """

        _ = DatasetConfig(
            texts=texts, 
            max_len=max_len
            )

        with torch.no_grad():
            self.model.eval()
            input_ids, attention_mask = self.__preprocess(raw_reviews=texts,max_len=max_len)
            outputs = self.model(input_ids, attention_mask=attention_mask)

            if self.multilabel:
                logits = outputs.logits
                probs = torch.sigmoid(
                    logits
                )  # Apply sigmoid for multi-label classification
                preds = (
                    (probs > 0.5).int().cpu().numpy().tolist()
                )  # Use threshold to make binary predictions
            else:
                preds = outputs.logits.argmax(dim=1).tolist()

            return preds

    def holdout(
        self,
        texts: List[str],
        labels: List[Any],
        val_size: float = 0.2,
        num_epochs: int = 20,
        batch_size: int = 4,
        balanced: bool = False,
        patience: int = 3,
        min_delta: float = 0.001,
        seed: int = 1811,
    ) -> Dict[str, Any]:
        """
        Perform holdout validation on the provided dataset.

        Args:
            texts (List[str]): List of input text samples.
            labels (List[Any]): Corresponding labels for the input texts.
            val_size (float): Proportion of data to use for validation (default: 0.2).
            num_epochs (int): Number of training epochs (default: 20).
            batch_size (int): Batch size for training (default: 4).
            balanced (bool): Whether to compute class weights for balancing (default: False).
            patience (int): Early stopping patience (default: 3).
            min_delta (float): Minimum improvement in validation loss to reset patience (default: 0.001).
            seed (int): Random seed for reproducibility (default: 1811).

        Returns:
            Dict[str, Any]: Dictionary containing holdout validation results.
        """

        _ = DatasetConfig(
            texts=texts, 
            labels=labels,
            val_size=val_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            balanced=balanced,
            patience=patience,
            min_delta=min_delta,
            seed=seed
            )

        self.__init_tokenizer()

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=val_size, shuffle=True, random_state=seed
        )

        if balanced:
            class_weights = self.compute_class_weights(train_labels, self.num_classes)
        else:
            class_weights = None

        if self.multilabel:
            train_dataset = TextDatasetML(
                train_texts, train_labels, self.tokenizer, self.max_len
            )
            val_dataset = TextDatasetML(
                val_texts, val_labels, self.tokenizer, self.max_len
            )
        else:
            train_dataset = TextDatasetMC(
                train_texts, train_labels, self.tokenizer, self.max_len
            )
            val_dataset = TextDatasetMC(
                val_texts, val_labels, self.tokenizer, self.max_len
            )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        ret = self.__train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            patience=patience,
            min_delta=min_delta,
            class_weights=class_weights,
            seed=seed,
        )

        self.logger_info("\n")
        self.__print_metrics(ret)

        return ret

    def cross_validation(
        self,
        texts: List[str],
        labels: List[Any],
        n_splits: int = 5,
        num_epochs: int = 20,
        batch_size: int = 4,
        balanced: bool = False,
        patience: int = 3,
        min_delta: float = 0.001,
        seed: int = 1811,
    ) -> List[Dict[str, Any]]:
        """
        Perform cross-validation on the provided dataset.

        Args:
            texts (List[str]): List of input text samples.
            labels (List[Any]): Corresponding labels for the input texts.
            n_splits (int): Number of folds for cross-validation (default: 5).
            num_epochs (int): Number of training epochs (default: 20).
            batch_size (int): Batch size for training (default: 4).
            balanced (bool): Whether to compute class weights for balancing (default: False).
            patience (int): Early stopping patience (default: 3).
            min_delta (float): Minimum improvement in validation loss to reset patience (default: 0.001).
            seed (int): Random seed for reproducibility (default: 1811).

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing cross-validation results for each fold.
        """

        _ = DatasetConfig(
            texts=texts, 
            labels=labels,
            n_splits=n_splits,
            num_epochs=num_epochs,
            batch_size=batch_size,
            balanced=balanced,
            patience=patience,
            min_delta=min_delta,
            seed=seed
            )

        self.__init_tokenizer()

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_accuracies = []
        fold_mcc = []
        fold_f1_micro = []
        fold_best_epoch = []
        rets = []

        all_labels = np.zeros(len(labels))
        all_preds = np.zeros(len(labels))

        for fold, (train_idx, val_idx) in enumerate(kf.split(texts, labels)):
            self.logger_info("\n")
            self.logger_info(f"> Fold {fold + 1}")

            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(train_idx))
            shuffled_train_idx = [train_idx[i] for i in shuffled_indices]
            train_texts = [texts[i] for i in shuffled_train_idx]
            train_labels = [labels[i] for i in shuffled_train_idx]

            np.random.seed(seed)
            shuffled_indices = np.random.permutation(len(val_idx))
            shuffled_val_idx = [val_idx[i] for i in shuffled_indices]
            val_texts = [texts[i] for i in shuffled_val_idx]
            val_labels = [labels[i] for i in shuffled_val_idx]

            if balanced:
                class_weights = self.compute_class_weights(
                    train_labels, self.num_classes
                )
            else:
                class_weights = None

            if self.multilabel:
                train_dataset = TextDatasetML(
                    train_texts, train_labels, self.tokenizer, self.max_len
                )
                val_dataset = TextDatasetML(
                    val_texts, val_labels, self.tokenizer, self.max_len
                )
            else:
                train_dataset = TextDatasetMC(
                    train_texts, train_labels, self.tokenizer, self.max_len
                )
                val_dataset = TextDatasetMC(
                    val_texts, val_labels, self.tokenizer, self.max_len
                )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, drop_last=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            ret = self.__train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                patience=patience,
                min_delta=min_delta,
                class_weights=class_weights,
                seed=seed,
            )

            rets.append(ret)

            best_epoch = ret.get("best_epoch")

            current_dict = ret["val_metrics"][best_epoch]
            val_accuracy = current_dict.get("accuracy")
            val_mcc = current_dict.get("mcc")
            val_f1_micro = current_dict.get("f1_micro")

            val_preds = ret["val_preds"][best_epoch]
            val_labels = ret["val_labels"][best_epoch]

            all_labels[shuffled_val_idx] = val_labels
            all_preds[shuffled_val_idx] = val_preds

            fold_best_epoch.append(best_epoch)
            fold_accuracies.append(val_accuracy)
            fold_mcc.append(val_mcc)
            fold_f1_micro.append(val_f1_micro)

        self.logger_info("\n")
        self.logger_info(">> Folds metrics:")
        self.logger_info(
            f" - Best epoch: [{np.min(fold_best_epoch):.4f} <-> {np.max(fold_best_epoch):.4f}]"
        )
        self.logger_info(
            f" - Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}"
        )
        self.logger_info(f" - MCC: {np.mean(fold_mcc):.4f} ± {np.std(fold_mcc):.4f}")
        self.logger_info(
            f" - F1-micro: {np.mean(fold_f1_micro):.4f} ± {np.std(fold_f1_micro):.4f}"
        )

        ret = self.__calculate_metrics(all_labels.tolist(), all_preds.tolist())

        self.logger_info("\n")
        self.__print_metrics({"val_metrics": ret})

        return rets

    def release_model(self) -> None:
        """
        Release the model from memory and clear GPU cache.
        """
        self.model = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
