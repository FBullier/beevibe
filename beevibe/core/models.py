import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from safetensors.torch import save_file, load_file
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import transformers

from beevibe.core.tokenizers import HFTokenizer

from typing import Optional, List

from huggingface_hub.utils import disable_progress_bars as hfhub_disable_progress_bar
hfhub_disable_progress_bar()


class HFModelForClassification(AutoModelForSequenceClassification):
    """
    A wrapper for Hugging Face models for sequence classification tasks.
    """

    def __init__(self):
        """
        Initializes the HFModelForClassification class.
        """
        super(HFModelForClassification, self).__init__()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """
        Load a model for sequence classification from a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            **kwargs: Additional keyword arguments to pass to the
                      AutoModelForSequenceClassification.from_pretrained method.

        Returns:
            HFModelForClassification: A model instance.
        """
        # Call the ancestor class's `from_pretrained` method
        return super(AutoModelForSequenceClassification, cls).from_pretrained(
            pretrained_model_name_or_path=model_name,
            **kwargs
        )


class BeeBaseModel(nn.Module):

    def __init__(self):
        super(BeeBaseModel, self).__init__()

    def from_pretrained(self, quantization_config: Optional[BitsAndBytesConfig] = None ):
        return


class BeeSimpleMaskModelForClassification(BeeBaseModel):
    """
    A simple model for sequence classification with a linear stack on top of a pretrained transformer.
    """

    def __init__(self, model_name: str, num_labels: int):
        """
        Initializes the SimpleModel class.

        Args:
            model_name (str): The name of the pretrained model.
            num_labels (int): The number of labels for classification.
        """
        super(BeeSimpleMaskModelForClassification, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.classes_names = []
        self.multilabel = None

        self.hftokenizer = None
        self.model_directory = ""

    def from_pretrained(self, quantization_config: Optional[BitsAndBytesConfig] = None ):

        self.base_model = AutoModel.from_pretrained(self.model_name, quantization_config=quantization_config)
        self.config = self.base_model.config

        self.classifier = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: torch.Tensor = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for classification. Default is None.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Output containing logits.
        """

        # Validate attention_mask
        if attention_mask is None:
            raise ValueError("attention_mask is required for the forward pass.")

        # Explicitly validate batch size and sequence length
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"input_ids shape {input_ids.shape} and attention_mask shape {attention_mask.shape} must match.")


        # Extract features from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][:, 0]  # Extract the CLS token representation
        logits = self.classifier(embeddings)

        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits)


    def predict(self, raw_texts: List[str], hftokenizer: Optional[HFTokenizer] = None, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5):
        """
        Perform predictions on raw text inputs using the model.

        This function tokenizes the input text data, processes it through the model in batches, and returns either
        class probabilities or predictions. Supports multi-class and multi-label classification based on the model configuration.

        Args:
            raw_texts (List[str]): A list of raw text strings to be classified.
            hftokenizer (Optional[HFTokenizer]): An optional tokenizer instance. If not provided, it attempts to use
                `self.hftokenizer` or loads a tokenizer based on `self.model_name`.
            return_probabilities (bool): If True, the function returns class probabilities. Otherwise, it returns
                predicted class labels (for multi-class) or binary predictions (for multi-label).
            batch_size (int): Number of samples to process per batch during prediction. Defaults to 32.
            num_workers (int): Number of parallel worker processes for batch processing (CPU only). Defaults to
                `None`, which uses all available CPU cores minus one.
            threshold (float): The threshold for binary predictions in multi-label classification. Defaults to 0.5.

        Returns:
            List: The predictions for the input texts. The output format depends on `return_probabilities` and
            the model type:
                - Multi-class: Returns a list of predicted class indices or probabilities.
                - Multi-label: Returns a list of binary predictions (per label) or probabilities.

        Raises:
            AssertionError: If no tokenizer is found and cannot be loaded for the given model name.

        Example:
            >>> model = BeeSimpleMaskModelForClassification(model_name="bert-base-uncased", num_labels=3)
            >>> raw_texts = ["This is a positive example.", "This is a negative example."]
            >>> predictions = model.predict(raw_texts, batch_size=16)
            >>> print(predictions)
            [0, 2]

            >>> probabilities = model.predict(raw_texts, return_probabilities=True, batch_size=16)
            >>> print(probabilities)
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
        """

        # Get/Load the tokenizer
        if hftokenizer is None:
            if self.hftokenizer is None:
                hftokenizer = HFTokenizer().from_pretrained(self.model_name, model_directory=self.model_directory)
                if hftokenizer.tokenizer is None:
                    assert f"No tokenizer found for the model name : {self.model_name}"
                else:
                    self.hftokenizer = hftokenizer
        else:
            self.hftokenizer = hftokenizer

        # Encode row texts list
        input_ids, attention_mask = self.hftokenizer.encode(raw_texts)

        # Get model raw predictions
        ret = self._raw_predict(input_ids, attention_mask, return_probabilities, batch_size, num_workers, threshold)

        return ret

    def process_batch(self, batch_input_ids, batch_attention_mask):
        """Process a single batch and return predictions or probabilities."""
        with torch.no_grad():  # Disable gradient computation
            outputs = self.forward(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits

            if self.multilabel:
                # Multi-label classification
                probs = torch.sigmoid(logits)  # Apply sigmoid for probabilities
                if self.return_probabilities:
                    return probs.cpu()  # Return probabilities
                else:
                    # Apply threshold to generate binary predictions
                    return (probs > self.threshold).int().cpu()
            else:
                # Multi-class classification
                if self.return_probabilities:
                    # Convert logits to probabilities using softmax
                    return softmax(logits, dim=-1).cpu()
                else:
                    # Convert logits to class labels using argmax
                    return torch.argmax(logits, dim=-1).cpu()

    def _raw_predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5):
        """
        Perform batched prediction on input data, supporting multi-label and multi-class classification.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask.
            return_probabilities (bool): If True, return probabilities instead of class labels.
            batch_size (int): Number of samples to process per batch.
            num_workers (int): Number of parallel processes to use.
            threshold (float): Threshold for binary predictions in multi-label classification.

        Returns:
            list: Predicted class labels, probabilities, or binary predictions for multi-label classification.
        """
        self.eval()  # Set model to evaluation mode

        self.multilabel = False
        self.threshold = threshold
        self.return_probabilities = return_probabilities

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        # Create batches
        batches = [
            (input_ids[i:i + batch_size], attention_mask[i:i + batch_size])
            for i in range(0, len(input_ids), batch_size)
        ]

        # Parallel processing with joblib
        results = Parallel(n_jobs=num_workers)(
            delayed(self.process_batch)(batch_input_ids, batch_attention_mask)
            for batch_input_ids, batch_attention_mask in batches
        )

        # Concatenate results from all batches
        return torch.cat(results, dim=0).tolist()


    def save_model_safetensors(self, save_directory: str):
        """
        Saves the model using safetensors format.

        Args:
            save_directory (str): The directory where the model will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Collect weights
        weights = {}
        weights.update(self.base_model.state_dict())  # Add base model weights
        weights.update({f"classifier.{k}": v for k, v in self.classifier.state_dict().items()})  # Add classifier weights

        # Save weights using safetensors
        save_file(weights, os.path.join(save_directory, "model.safetensors"))

        # Manage JSON serialization
        if isinstance(self.classes_names, np.ndarray):
            classes_names = self.classes_names.tolist()
        else:
            classes_names = self.classes_names

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes_names": classes_names,
            "multilabel":self.multilabel
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load_model_safetensors(cls, save_directory: str):
        """
        Loads the model from safetensors format.

        Args:
            save_directory (str): The directory where the model is saved.

        Returns:
            BeeSimpleMaskModelForClassification: The loaded model.
        """
        # Load the configuration
        #config = torch.load(os.path.join(save_directory, "config.pth"))
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config = json.load(f)

        model_name = config["model_name"]
        num_labels = config["num_labels"]
        multilabel = config["multilabel"]
        classes_names = config["classes_names"]

        # Manage JSON serialization
        if isinstance(classes_names, list):
            classes_names = np.array(classes_names)

        # Initialize the model
        model = cls(model_name=model_name, num_labels=num_labels)

        # Create model
        model.model_directory = save_directory
        model.multilabel = multilabel
        model.classes_names = classes_names
        model.from_pretrained()  # Load the base model

        # Load the weights from safetensors
        weights = load_file(os.path.join(save_directory, "model.safetensors"))
        base_model_weights = {k: v for k, v in weights.items() if not k.startswith("classifier.")}
        classifier_weights = {k[len("classifier."):]: v for k, v in weights.items() if k.startswith("classifier.")}

        # Load the weights into the respective components
        model.base_model.load_state_dict(base_model_weights, strict=True)
        model.classifier.load_state_dict(classifier_weights, strict=True)

        return model


class BeeCustomMaskModelForClassification(BeeBaseModel):
    """
    A custom model for sequence classification with a flexible linear stack on top of a pretrained transformer.
    """

    def __init__(self, model_name: str, num_labels: int, layer_configs: list[dict]):
        """
        Initializes the CustomModel class.

        Args:
            model_name (str): The name of the pretrained model.
            num_labels (int): The number of labels for classification.
            layer_configs (list of dict): Configuration for custom layers.
        """
        super(BeeCustomMaskModelForClassification, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.classes_names = []
        self.multilabel = None
        self.layer_configs = layer_configs

        self.hftokenizer = None
        self.model_directory = ""

    def from_pretrained(self, quantization_config: Optional[BitsAndBytesConfig] = None ):

        self.base_model = AutoModel.from_pretrained(self.model_name, quantization_config=quantization_config)
        self.config = self.base_model.config

        self.classifier = self._build_custom_stack(self.layer_configs)

    def _build_custom_stack(self, layer_configs: list[dict]) -> nn.Sequential:
        """
        Builds a custom stack of layers based on the given configurations.

        Args:
            layer_configs (list of dict): Configuration for each layer.

        Returns:
            nn.Sequential: A stack of custom layers.
        """
        layers = []
        previous_size = layer_configs[0]["input_size"]  # This should be 768 for camembert base

        for config in layer_configs:
            output_size = config["output_size"]
            activation = config.get("activation")
            dropout_rate = config.get("dropout_rate")
            batch_norm = config.get("batch_norm", False)
            layer_norm = config.get("layer_norm", False)
            residual = config.get("residual", False)

            # Add linear layer
            linear_layer = nn.Linear(previous_size, output_size)
            layers.append(linear_layer)

            # Apply batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))

            # Apply layer normalization
            if layer_norm:
                layers.append(nn.LayerNorm(output_size))

            # Apply activation (instantiate the activation function if provided)
            if activation:
                layers.append(activation())  # Instantiate activation function

            # Apply dropout
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))

            # Handle residual connections (to be implemented)
            if residual:
                pass

            previous_size = output_size  # Update input size for the next layer

        return nn.Sequential(*layers)


    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: torch.Tensor = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for classification. Default is None.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Output containing logits.
        """

        # Validate attention_mask
        if attention_mask is None:
            raise ValueError("attention_mask is required for the forward pass.")

        # Explicitly validate batch size and sequence length
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"input_ids shape {input_ids.shape} and attention_mask shape {attention_mask.shape} must match.")


        # Extract features from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][:, 0]  # Extract the CLS token representation
        logits = self.classifier(embeddings)

        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits)


    def predict(self, raw_texts: List[str], hftokenizer: Optional[HFTokenizer] = None, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5):
        """
        Perform predictions on raw text inputs using the model.

        This function tokenizes the input text data, processes it through the model in batches, and returns either
        class probabilities or predictions. Supports multi-class and multi-label classification based on the model configuration.

        Args:
            raw_texts (List[str]): A list of raw text strings to be classified.
            hftokenizer (Optional[HFTokenizer]): An optional tokenizer instance. If not provided, it attempts to use
                `self.hftokenizer` or loads a tokenizer based on `self.model_name`.
            return_probabilities (bool): If True, the function returns class probabilities. Otherwise, it returns
                predicted class labels (for multi-class) or binary predictions (for multi-label).
            batch_size (int): Number of samples to process per batch during prediction. Defaults to 32.
            num_workers (int): Number of parallel worker processes for batch processing (CPU only). Defaults to
                `None`, which uses all available CPU cores minus one.
            threshold (float): The threshold for binary predictions in multi-label classification. Defaults to 0.5.

        Returns:
            List: The predictions for the input texts. The output format depends on `return_probabilities` and
            the model type:
                - Multi-class: Returns a list of predicted class indices or probabilities.
                - Multi-label: Returns a list of binary predictions (per label) or probabilities.

        Raises:
            AssertionError: If no tokenizer is found and cannot be loaded for the given model name.

        Example:
            >>> model = BeeSimpleMaskModelForClassification(model_name="bert-base-uncased", num_labels=3)
            >>> raw_texts = ["This is a positive example.", "This is a negative example."]
            >>> predictions = model.predict(raw_texts, batch_size=16)
            >>> print(predictions)
            [0, 2]

            >>> probabilities = model.predict(raw_texts, return_probabilities=True, batch_size=16)
            >>> print(probabilities)
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
        """

        # Get/Load the tokenizer
        if hftokenizer is None:
            if self.hftokenizer is None:
                hftokenizer = HFTokenizer().from_pretrained(self.model_name, model_directory=self.model_directory)
                if hftokenizer.tokenizer is None:
                    assert f"No tokenizer found for the model name : {self.model_name}"
                else:
                    self.hftokenizer = hftokenizer
        else:
            self.hftokenizer = hftokenizer

        # Encode row texts list
        input_ids, attention_mask = self.hftokenizer.encode(raw_texts)

        # Get model raw predictions
        ret = self._raw_predict(input_ids, attention_mask, return_probabilities, batch_size, num_workers, threshold)

        return ret


    def process_batch(self, batch_input_ids, batch_attention_mask):
        """Process a single batch and return predictions or probabilities."""
        with torch.no_grad():  # Disable gradient computation
            outputs = self.forward(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits

            if self.multilabel:
                # Multi-label classification
                probs = torch.sigmoid(logits)  # Apply sigmoid for probabilities
                if self.return_probabilities:
                    return probs.cpu()  # Return probabilities
                else:
                    # Apply threshold to generate binary predictions
                    return (probs > self.threshold).int().cpu()
            else:
                # Multi-class classification
                if self.return_probabilities:
                    # Convert logits to probabilities using softmax
                    return softmax(logits, dim=-1).cpu()
                else:
                    # Convert logits to class labels using argmax
                    return torch.argmax(logits, dim=-1).cpu()


    def _raw_predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5):
        """
        Perform batched prediction on input data, supporting multi-label and multi-class classification.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask.
            return_probabilities (bool): If True, return probabilities instead of class labels.
            batch_size (int): Number of samples to process per batch.
            num_workers (int): Number of parallel processes to use.
            threshold (float): Threshold for binary predictions in multi-label classification.

        Returns:
            list: Predicted class labels, probabilities, or binary predictions for multi-label classification.
        """
        self.eval()  # Set model to evaluation mode

        self.multilabel = False
        self.threshold = threshold
        self.return_probabilities = return_probabilities

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        # Create batches
        batches = [
            (input_ids[i:i + batch_size], attention_mask[i:i + batch_size])
            for i in range(0, len(input_ids), batch_size)
        ]

        # Parallel processing with joblib
        results = Parallel(n_jobs=num_workers)(
            delayed(self.process_batch)(batch_input_ids, batch_attention_mask)
            for batch_input_ids, batch_attention_mask in batches
        )

        # Concatenate results from all batches
        return torch.cat(results, dim=0).tolist()


    def save_model_safetensors(self, save_directory: str):
        """
        Saves the model using safetensors format.

        Args:
            save_directory (str): The directory where the model will be saved.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Collect weights
        weights = {}
        weights.update(self.base_model.state_dict())  # Add base model weights
        weights.update({f"classifier.{k}": v for k, v in self.classifier.state_dict().items()})  # Add classifier weights

        # Save weights using safetensors
        save_file(weights, os.path.join(save_directory, "model.safetensors"))

        # Manage JSON serialization
        if isinstance(self.classes_names, np.ndarray):
            classes_names = self.classes_names.tolist()
        else:
            classes_names = self.classes_names

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes_names": classes_names,
            "multilabel":self.multilabel,
            "layer_configs": self.layer_configs,
        }

        # torch.save(config, os.path.join(save_directory, "config.pth"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)


    @classmethod
    def load_model_safetensors(cls, save_directory: str):
        """
        Loads the model from safetensors format.

        Args:
            save_directory (str): The directory where the model is saved.

        Returns:
            BeeCustomMaskModelForClassification: The loaded model.
        """
        # Load the configuration
        #config = torch.load(os.path.join(save_directory, "config.pth"), weights_only=True)
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config = json.load(f)

        model_name = config["model_name"]
        num_labels = config["num_labels"]
        multilabel = config["multilabel"]
        classes_names = config["classes_names"]
        layer_configs = config["layer_configs"]

        # Manage JSON serialization
        if isinstance(classes_names, list):
            classes_names = np.array(classes_names)

        # Initialize the model
        model = cls(model_name=model_name, num_labels=num_labels, layer_configs=layer_configs)

        # Create model
        model.model_directory = save_directory
        model.multilabel = multilabel
        model.classes_names = classes_names
        model.from_pretrained()  # Load the base model

        # Load the weights from safetensors
        weights = load_file(os.path.join(save_directory, "model.safetensors"))
        base_model_weights = {k: v for k, v in weights.items() if not k.startswith("classifier.")}
        classifier_weights = {k[len("classifier."):]: v for k, v in weights.items() if k.startswith("classifier.")}

        # Load the weights into the respective components
        model.base_model.load_state_dict(base_model_weights, strict=True)
        model.classifier.load_state_dict(classifier_weights, strict=True)

        return model
