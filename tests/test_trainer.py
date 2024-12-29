import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from beevibe import HFTokenizer, HFModelForClassification, BeeSimpleMaskModelForClassification, BeeCustomMaskModelForClassification

@pytest.fixture
def tokenizer():
    return HFTokenizer().from_pretrained("bert-base-uncased")

@pytest.fixture
def simple_model():
    return BeeSimpleMaskModelForClassification("bert-base-uncased", num_labels=2)

@pytest.fixture
def custom_model():
    layer_configs = [
        {"input_size": 768, "output_size": 128, "activation": torch.nn.ReLU, "dropout_rate": 0.1},
        {"input_size": 128, "output_size": 2},
    ]
    return BeeCustomMaskModelForClassification("bert-base-uncased", num_labels=2, layer_configs=layer_configs)

def test_hf_tokenizer(tokenizer):
    text = "Hello, how are you?"
    tokens = tokenizer(text, return_tensors="pt")
    assert "input_ids" in tokens
    assert "attention_mask" in tokens

def test_hf_model_for_classification():
    model = HFModelForClassification().from_pretrained("bert-base-uncased")
    assert isinstance(model, torch.nn.Module)

def test_simple_model_forward(simple_model, tokenizer):
    text = ["This is a test sentence.", "Another example sentence."]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor([0, 1])

    outputs = simple_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs.logits.shape == (2, 2)  # Batch size x num_labels

def test_custom_model_forward(custom_model, tokenizer):
    text = ["This is a test sentence.", "Another example sentence."]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor([0, 1])

    outputs = custom_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs.logits.shape == (2, 2)  # Batch size x num_labels

def test_custom_model_layer_config():
    layer_configs = [
        {"input_size": 768, "output_size": 256, "activation": torch.nn.ReLU, "dropout_rate": 0.1},
        {"input_size": 256, "output_size": 128, "activation": torch.nn.ReLU, "dropout_rate": 0.1},
        {"input_size": 128, "output_size": 2},
    ]
    model = BeeCustomMaskModelForClassification("bert-base-uncased", num_labels=2, layer_configs=layer_configs)
    
    # Count expected layers (linear + optional components like activation/dropout)
    expected_layers = sum(1 + bool(config.get("activation")) + bool(config.get("dropout_rate")) for config in layer_configs)

    assert len(model.classifier) == expected_layers

def test_tokenizer_special_tokens(tokenizer):
    text = "Hello, how are you?"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    assert tokens["input_ids"].shape[1] >= len(text.split())  # Ensures special tokens are added

def test_simple_model_output_consistency(simple_model, tokenizer):
    text = ["Test sentence one.", "Test sentence two."]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Set model to evaluation mode to disable dropout
    simple_model.eval()
    outputs1 = simple_model(input_ids=input_ids, attention_mask=attention_mask)
    outputs2 = simple_model(input_ids=input_ids, attention_mask=attention_mask)

    # Ensure deterministic outputs for the same inputs
    assert torch.allclose(outputs1.logits, outputs2.logits, atol=1e-5)

def test_custom_model_output_shape(custom_model, tokenizer):
    text = ["This is a test.", "Another example."]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = custom_model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape == (2, 2)  # Batch size x num_labels
    assert outputs.logits.dtype == torch.float32  # Ensure logits are of type float32

def test_custom_model_with_incorrect_input(custom_model):
    with pytest.raises(ValueError, match="attention_mask is required for the forward pass"):
        input_ids = torch.tensor([[101, 2000, 102]])  # Missing attention mask
        custom_model(input_ids=input_ids)

def test_simple_model_gradients(simple_model, tokenizer):
    text = ["Gradient test sentence."]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor([0])

    outputs = simple_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
    loss.backward()

    # Ensure gradients are computed for model parameters
    has_grads = any(param.grad is not None for param in simple_model.parameters())
    assert has_grads