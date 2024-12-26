import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import transformers

from huggingface_hub.utils import disable_progress_bars as hfhub_disable_progress_bar
hfhub_disable_progress_bar()

class HFTokenizer:
    """
    A wrapper for Hugging Face Tokenizer to handle tokenization with custom configurations.
    """

    def __init__(self):
        """
        Initializes the HFTokenizer class.
        """
        super(HFTokenizer, self).__init__()

    def from_pretrained(self, model_name: str, clean_up_tokenization_spaces: bool = None):
        """
        Load a tokenizer from a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            clean_up_tokenization_spaces (bool, optional): Whether to clean up tokenization spaces. Default is None.

        Returns:
            transformers.PreTrainedTokenizer: A tokenizer instance.
        """
        if clean_up_tokenization_spaces is None:
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
        else:
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

class HFModelForClassification:
    """
    A wrapper for Hugging Face models for sequence classification tasks.
    """

    def __init__(self):
        """
        Initializes the HFModelForClassification class.
        """
        super(HFModelForClassification, self).__init__()

    def from_pretrained(self, model_name: str, num_labels: int = None):
        """
        Load a model for sequence classification from a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            num_labels (int, optional): Number of labels for classification. Default is None.

        Returns:
            transformers.AutoModelForSequenceClassification: A model instance.
        """
        if num_labels is None:
            return AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
            )
        else:
            return AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=num_labels
            )

class SimpleModel(nn.Module):
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
        super(SimpleModel, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for classification. Default is None.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Output containing logits.
        """
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][:, 0]  # Extract the CLS token representation
        logits = self.linear_relu_stack(embeddings)

        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits)

class CustomModel(nn.Module):
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
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(self.model_name)

        # Build custom linear stack
        self.linear_relu_stack = self._build_custom_stack(layer_configs)

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for classification. Default is None.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: Output containing logits.
        """
        # Extract features from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][
            :, 0
        ]  # CLS token representation (shape: [batch_size, 768] for Camembert)

        # Pass embeddings through custom stack
        logits = self.linear_relu_stack(embeddings)

        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits)
