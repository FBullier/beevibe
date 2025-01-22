import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file, load_file
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from multiprocessing import cpu_count
import transformers

from beevibe.core.tokenizers import HFTokenizer

from typing import Optional, List

from huggingface_hub.utils import disable_progress_bars as hfhub_disable_progress_bar
hfhub_disable_progress_bar()


class HFMLMClassifier(AutoModelForSequenceClassification):
    """
    A wrapper for Hugging Face models for sequence classification tasks.
    """

    def __init__(self):
        """
        Initializes the HFMLMClassifier class.
        """
        super(HFMLMClassifier, self).__init__()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """
        Load a model for sequence classification from a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            **kwargs: Additional keyword arguments to pass to the
                      AutoModelForSequenceClassification.from_pretrained method.

        Returns:
            HFMLMClassifier: A model instance.
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


class BeeMLMClassifier(BeeBaseModel):
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
        super(BeeMLMClassifier, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.classes_names = []
        self.multilabel = None
        self.layer_configs = layer_configs
        self.return_probabilities = False

        self.hftokenizer = None
        self.model_directory = ""

        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def from_pretrained(self, quantization_config: Optional[BitsAndBytesConfig] = None ):

        self.base_model = AutoModel.from_pretrained(self.model_name, quantization_config=quantization_config)
        self.config = self.base_model.config

        self.verify_layer_configs()
        self.classifier = self._build_custom_stack(self.layer_configs)

    def verify_layer_configs(self):
        """
        Verifies the content of self.layer_configs to ensure it is valid before building the custom stack.

        Raises:
            ValueError: If any issue is found in the layer configurations.
        """
        if not isinstance(self.layer_configs, list):
            raise ValueError("layer_configs must be a list of dictionaries.")

        # Define allowed keys for the layer configurations
        allowed_keys = {
            "input_size",
            "output_size",
            "activation",
            "dropout_rate",
            "batch_norm",
            "layer_norm",
            "residual",
        }

        for i, config in enumerate(self.layer_configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer configuration at index {i} is not a dictionary.")

            # Check for invalid keys
            invalid_keys = set(config.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(f"Invalid keys {invalid_keys} found in layer configuration at index {i}.")

            # Validate required keys
            required_keys = ["input_size", "output_size"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required key '{key}' in layer configuration at index {i}.")
                if not isinstance(config[key], int) or config[key] <= 0:
                    raise ValueError(f"'{key}' in layer configuration at index {i} must be a positive integer.")

            # Validate optional keys
            if "activation" in config:
                activation = config["activation"]
                if activation is not None and not callable(activation):
                    raise ValueError(f"'activation' in layer configuration at index {i} must be callable or None.")

            if "dropout_rate" in config:
                dropout_rate = config["dropout_rate"]
                if not (isinstance(dropout_rate, (int, float)) and 0 <= dropout_rate <= 1):
                    raise ValueError(f"'dropout_rate' in layer configuration at index {i} must be between 0 and 1.")

            if "batch_norm" in config:
                if not isinstance(config["batch_norm"], bool):
                    raise ValueError(f"'batch_norm' in layer configuration at index {i} must be a boolean.")

            if "layer_norm" in config:
                if not isinstance(config["layer_norm"], bool):
                    raise ValueError(f"'layer_norm' in layer configuration at index {i} must be a boolean.")

            if "residual" in config:
                if not isinstance(config["residual"], bool):
                    raise ValueError(f"'residual' in layer configuration at index {i} must be a boolean.")

        # Ensure that the input size of the first layer matches the base model's hidden size
        if self.layer_configs:
            if hasattr(self.config, "hidden_size"):
                first_input_size = self.layer_configs[0]["input_size"]
                if self.config.hidden_size != first_input_size:
                    raise ValueError(
                        f"Input size of the first layer ({first_input_size}) does not match the base model's hidden size ({self.config.hidden_size})."
                    )
            else:
                raise AttributeError("Base model configuration does not have a 'hidden_size' attribute.")


    def _build_custom_stack(self, layer_configs: list[dict]) -> nn.Sequential:
        """
        Builds a custom stack of layers based on the given configurations.

        Args:
            layer_configs (list of dict): Configuration for each layer.

        Returns:
            nn.Sequential: A stack of custom layers.
        """
        layers = []

        # Create a default classification head
        if not layer_configs:
            if hasattr(self.config, "hidden_size"):
                layer_configs = [{"input_size": self.config.hidden_size, "output_size": self.num_labels, "activation": None}]
            else:
                raise AttributeError("Can't create a default classification head beacause the base model configuration does not have a 'hidden_size' attribute.")
        else:
            # Check input size of classification head
            if hasattr(self.config, "hidden_size"):
                if self.config.hidden_size != layer_configs[0]["input_size"]:
                    raise AttributeError(f"Input size of classification head does not match with the base model configuration 'hidden_size' attribute. {self.config.hidden_size} != {layer_configs[0]['input_size']}")

        # Get previous layer size, this should be 768 for common MLM
        previous_size = layer_configs[0]["input_size"]

        # Create a simple classifier if there is no layer_configs

        for config in layer_configs:
            output_size = config["output_size"]
            activation = config.get("activation")
            dropout_rate = config.get("dropout_rate")
            batch_norm = config.get("batch_norm", False)
            layer_norm = config.get("layer_norm", False)
            residual = config.get("residual", False)

            if output_size <= 0:
                raise ValueError(f"Invalid output_size {output_size} in layer configuration.")

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


    def predict(self, raw_texts: List[str], hftokenizer: Optional[HFTokenizer] = None, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5, device=None):
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
        ret = self._raw_predict(input_ids, attention_mask, return_probabilities, batch_size, num_workers, threshold, device)

        return ret

    def process_batch(self, batch_input_ids, batch_attention_mask):
        """Process a single batch and return predictions or probabilities."""
        with torch.no_grad():  # Disable gradient computation
            # Ensure inputs are on the same device as the model
            batch_input_ids = batch_input_ids.to(next(self.parameters()).device)
            batch_attention_mask = batch_attention_mask.to(next(self.parameters()).device)

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

    def _raw_predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5, device=None):
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

        self.threshold = threshold
        self.return_probabilities = return_probabilities

        # Ensure `device` is a torch.device object
        if isinstance(device, str):
            device = torch.device(device)
        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)  # Move the model to the device

        # Set the number of workers
        if device.type != "cpu":
            num_workers = 0
        else:
            if num_workers is None:
                num_workers = max(1, cpu_count() - 1)

        # Move inputs to the correct device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Create a DataLoader for efficient batching
        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        # Process batches in DataLoader
        results = []
        for batch_input_ids, batch_attention_mask in dataloader:
            batch_input_ids = batch_input_ids.to(device)  # Ensure the batch is on the correct device
            batch_attention_mask = batch_attention_mask.to(device)  # Ensure the batch is on the correct device
            results.append(self.process_batch(batch_input_ids, batch_attention_mask))

        # Concatenate all results
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

        # jsonify layer_config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "classes_names": classes_names,
            "multilabel":self.multilabel,
            "head_layer_config": [
                {k: (v.__name__ if k == "activation" and v else v) for k, v in layer.items() if not (k == "activation" and v is None)}
                for layer in self.layer_configs
            ],
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
        classes_names = config["classes_names"]
        multilabel = config["multilabel"]

        #layer_configs = config["layer_configs"]
        layer_configs = [
           {k: (getattr(nn, v) if k == "activation" and v else v) for k, v in layer.items() if not (k == "activation" and v is None)}
            for layer in config.get("head_layer_config")
           #{**layer, "activation": getattr(nn, layer.get("activation")) if layer.get("activation") else None}
           #for layer in config.get("head_layer_config")
           ]


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
