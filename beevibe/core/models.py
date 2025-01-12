import os
import json
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from safetensors.torch import save_file, load_file
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from multiprocessing import Pool, cpu_count
import transformers
from typing import Optional, List


from huggingface_hub.utils import disable_progress_bars as hfhub_disable_progress_bar
hfhub_disable_progress_bar()


class HFTokenizer:
    """
    A wrapper for Hugging Face Tokenizer to handle tokenization with custom configurations.
    """

    def __init__(self, preprocessing_config: Optional[dict] = None):
        """
        Initializes the HFTokenizer class.
        """
        super(HFTokenizer, self).__init__()

        self.tokenizer =  None
        self.preprocessing_config = preprocessing_config


    def from_pretrained(self, model_name: str, **kwargs):
        """
        Load a tokenizer from a pretrained model.

        Args:
            model_name (str): The name of the pretrained model.
            **kwargs: Additional keyword arguments to pass to the
                      AutoTokenizer.from_pretrained method.

        Returns:
            transformers.PreTrainedTokenizer: A tokenizer instance.
        """

        # Load preprocessing config if needed from model path
        if self.preprocessing_config is None:
            self.preprocessing_config = self.load_config(model_name)
            if self.preprocessing_config is None:
                assert "Tokenizer preprocessing configuration is not define"

        # Load tokenizer from model name or path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name,
                **kwargs
            )
        except Exception:
            self.tokenizer = None

        return self

    def  save_pretrained(self, path):
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        else:
            assert "Tokenizer is not define, call from_pretrained()"

    def save_config(self, path):
        if self.tokenizer:
            if self.preprocessing_config:
                with open(f"{path}/preprocessing_config.json", "w") as f:
                    json.dump(self.preprocessing_config, f)
            else:
                assert "Tokenizer preprocessing configuration is not define"
        else:
            assert "Tokenizer is not define, call from_pretrained()"

    def load_config(self, path):
        try:
            with open(f"{path}/preprocessing_config.json", "r") as f:
                self.preprocessing_config = json.load(f)
        except FileNotFoundError:
            self.preprocessing_config = None

    def encode(self, raw_texts: str, ):
        if self.tokenizer:
            if self.preprocessing_config:
                encoded_batch = self.tokenizer(
                            raw_texts,
                            **self.preprocessing_config
                    )
            else:
                assert "Tokenizer preprocessing configuration must be defined, call from_pretrained()"
        else:
            assert "Tokenizer is not define, call from_pretrained()"
        return encoded_batch["input_ids"], encoded_batch["attention_mask"]

    def encode_plus(self, raw_texts: str, ):
        
        if self.tokenizer:
            if self.preprocessing_config:
                encoded_batch = self.tokenizer.encode_plus(
                            raw_texts,
                            **self.preprocessing_config
                    )
            else:
                assert "Tokenizer preprocessing configuration must be defined, call from_pretrained()"
        else:
            assert "Tokenizer is not define, call from_pretrained()"

        return encoded_batch


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
        self.hftokenizer = None

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
                hftokenizer = HFTokenizer().from_pretrained(self.model_name)
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


    def _raw_predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_probabilities: bool = False, batch_size: int = 32, num_workers: int = None, threshold: float = 0.5):
        """
        Perform batched prediction on input data, supporting multi-label and multi-class classification.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask.
            return_probabilities (bool): If True, return probabilities instead of class labels.
            batch_size (int): Number of samples to process per batch.
            num_workers (int): Number of parallel processes to use (defaults to number of CPU cores minus one).
            threshold (float): Threshold for binary predictions in multi-label classification.

        Returns:
            list: Predicted class labels, probabilities, or binary predictions for multi-label classification.
        """
        self.eval()  # Set model to evaluation mode

        # Default to the number of available CPU cores minus one
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        def process_batch(batch_input_ids, batch_attention_mask):
            """Process a single batch and return predictions or probabilities."""
            with torch.no_grad():  # Disable gradient computation
                outputs = self.forward(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits

                if self.multilabel:
                    # Multi-label classification
                    probs = torch.sigmoid(logits)  # Apply sigmoid for probabilities
                    if return_probabilities:
                        return probs.cpu()  # Return probabilities
                    else:
                        # Apply threshold to generate binary predictions
                        return (probs > threshold).int().cpu()
                else:
                    # Multi-class classification
                    if return_probabilities:
                        # Convert logits to probabilities using softmax
                        return softmax(logits, dim=-1).cpu()
                    else:
                        # Convert logits to class labels using argmax
                        return torch.argmax(logits, dim=-1).cpu()

        # Create batches
        batches = [
            (input_ids[i:i + batch_size], attention_mask[i:i + batch_size])
            for i in range(0, len(input_ids), batch_size)
        ]

        if torch.cuda.is_available():
            # Process batches sequentially for GPU (CUDA tensors cannot be shared across processes)
            results = []
            for batch_input_ids, batch_attention_mask in batches:
                results.append(process_batch(batch_input_ids, batch_attention_mask))
        else:
            # Use multiprocessing for CPU
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(process_batch, batches)

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

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
        }

        #torch.save(config, os.path.join(save_directory, "config.pth"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

        print(f"Model saved in Safetensors format to {save_directory}")

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

        # Initialize the model
        model = cls(model_name=model_name, num_labels=num_labels)
        model.from_pretrained()  # Load the base model

        # Load the weights from safetensors
        weights = load_file(os.path.join(save_directory, "model.safetensors"))
        base_model_weights = {k: v for k, v in weights.items() if not k.startswith("classifier.")}
        classifier_weights = {k[len("classifier."):]: v for k, v in weights.items() if k.startswith("classifier.")}

        # Load the weights into the respective components
        model.base_model.load_state_dict(base_model_weights, strict=True)
        model.classifier.load_state_dict(classifier_weights, strict=True)

        print(f"Model loaded from Safetensors format in {save_directory}")
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
        self.layer_configs = layer_configs

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

        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
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
        layer_configs = config["layer_configs"]

        # Initialize the model
        model = cls(model_name=model_name, num_labels=num_labels, layer_configs=layer_configs)
        model.from_pretrained()  # Load the base model

        # Load the weights from safetensors
        weights = load_file(os.path.join(save_directory, "model.safetensors"))
        base_model_weights = {k: v for k, v in weights.items() if not k.startswith("classifier.")}
        classifier_weights = {k[len("classifier."):]: v for k, v in weights.items() if k.startswith("classifier.")}

        # Load the weights into the respective components
        model.base_model.load_state_dict(base_model_weights, strict=True)
        model.classifier.load_state_dict(classifier_weights, strict=True)

        return model
