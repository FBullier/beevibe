import torch.nn as nn
from transformers import AutoModel
import transformers


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][:, 0]  # Extract the CLS token representation
        logits = self.linear_relu_stack(embeddings)

        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits)


class CustomModelParams(nn.Module):
    def __init__(self, model_name, num_labels, layer_configs):
        super(CustomModelParams, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(self.model_name)

        # Build custom linear stack
        self.linear_relu_stack = self._build_custom_stack(layer_configs)

    def _build_custom_stack(self, layer_configs):
        layers = []
        previous_size = layer_configs[0][
            "input_size"
        ]  # This should be 768 for camembert base

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

    def forward(self, input_ids, attention_mask, labels=None):
        # Extract features from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][
            :, 0
        ]  # CLS token representation (shape: [batch_size, 768])

        # Pass embeddings through custom stack
        logits = self.linear_relu_stack(embeddings)

        return transformers.modeling_outputs.SequenceClassifierOutput(logits=logits)
