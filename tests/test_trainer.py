import pytest
import torch
from torch import nn
from unittest.mock import MagicMock, patch
from beevibe import BeeMLMClassifier


@pytest.fixture
def model():
    model_name = "bert-base-uncased"
    num_labels = 3
    layer_configs = [
        {"input_size": 768, "output_size": 512, "activation": nn.ReLU, "dropout_rate": 0.1},
        {"input_size": 512, "output_size": num_labels, "activation": None, "dropout_rate": None},
    ]
    return BeeMLMClassifier(model_name=model_name, num_labels=num_labels, layer_configs=layer_configs)

def test_init(model):
    assert model.model_name == "bert-base-uncased"
    assert model.num_labels == 3
    assert model.layer_configs is not None
    assert model.classes_names == []

def test_from_pretrained(model):
    with patch("transformers.AutoModel.from_pretrained", return_value=MagicMock()) as mock_from_pretrained:
        model.from_pretrained()
        mock_from_pretrained.assert_called_once_with(model.model_name, quantization_config=None)
        assert hasattr(model, "base_model")
        assert isinstance(model.classifier, nn.Sequential)

def test_build_custom_stack(model):
    stack = model._build_custom_stack(model.layer_configs)
    assert isinstance(stack, nn.Sequential)
    assert len(stack) > 0

def test_forward_attention_mask_validation(model):
    input_ids = torch.randint(0, 100, (4, 16))
    with pytest.raises(ValueError, match="attention_mask is required for the forward pass."):
        model.forward(input_ids=input_ids)

def test_predict(model):
    model.hftokenizer = MagicMock()
    model.hftokenizer.encode = MagicMock(return_value=(torch.randint(0, 100, (4, 16)), torch.ones((4, 16))))
    model._raw_predict = MagicMock(return_value=[[0.8, 0.2], [0.4, 0.6]])

    raw_texts = ["Sample text 1", "Sample text 2"]
    predictions = model.predict(raw_texts)
    assert len(predictions) == len(raw_texts)

def test_forward_shape_mismatch(model):
    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 15))  # Mismatched shape
    with pytest.raises(ValueError, match="input_ids shape .* and attention_mask shape .* must match."):
        model.forward(input_ids=input_ids, attention_mask=attention_mask)

def test_forward_pass(model):
    model.base_model = MagicMock()
    model.base_model.return_value = (torch.rand((4, 16, 768)),)  # CLS token representations

    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 16))
    model.classifier = nn.Sequential(nn.Linear(768, model.num_labels))

    output = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    assert hasattr(output, "logits")
    assert output.logits.shape == (4, model.num_labels)

    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 16))
    model.classifier = nn.Sequential(nn.Linear(768, model.num_labels))

    output = model.forward(input_ids=input_ids, attention_mask=attention_mask)
    assert hasattr(output, "logits")
    assert output.logits.shape == (4, model.num_labels)

def test_predict_multilabel(model):
    model.hftokenizer = MagicMock()
    model.hftokenizer.encode = MagicMock(return_value=(torch.randint(0, 100, (4, 16)), torch.ones((4, 16))))
    model._raw_predict = MagicMock(return_value=torch.tensor([[0.8, 0.2], [0.4, 0.6]]))

    model.multilabel = True
    raw_texts = ["Sample text 1", "Sample text 2"]
    predictions = model.predict(raw_texts, return_probabilities=False, threshold=0.5)

    assert len(predictions) == len(raw_texts)
    assert all(isinstance(pred, torch.Tensor) for pred in predictions)

def test_save_and_load_model(tmp_path, model):
    # Mock base model
    mock_base_model = MagicMock()
    mock_base_model.state_dict.return_value = {"dummy_key": torch.rand(1)}
    mock_base_model.load_state_dict = MagicMock()  # Skip strict loading

    model.base_model = mock_base_model

    # Mock classifier with state_dict matching the Sequential structure
    mock_classifier = MagicMock()
    mock_classifier.state_dict.return_value = {
        "0.weight": torch.rand(512, 768),
        "0.bias": torch.rand(512),
        "2.weight": torch.rand(3, 512),
        "2.bias": torch.rand(3),
    }
    model.classifier = mock_classifier

    # Mock valid layer_configs with serializable activations
    model.layer_configs = [
        {"input_size": 768, "output_size": 512, "activation": nn.ReLU},
        {"input_size": 512, "output_size": 3, "activation": None},
    ]

    save_directory = tmp_path / "model_dir"
    model.save_model_safetensors(save_directory)

    # Assert saved files exist
    assert (save_directory / "model.safetensors").exists()
    assert (save_directory / "config.json").exists()

    # Load model
    with patch("transformers.BertModel.load_state_dict", MagicMock()) as mocked_load:
        loaded_model = BeeMLMClassifier.load_model_safetensors(save_directory)
        mocked_load.assert_called_once()

    # Validate loaded model attributes
    assert loaded_model.model_name == model.model_name
    assert loaded_model.num_labels == model.num_labels
    assert loaded_model.layer_configs[0]["activation"] == nn.ReLU

def test_raw_predict(model):
    model.eval = MagicMock()
    model.to = MagicMock()
    model.device = "cpu"
    model.process_batch = MagicMock(
        side_effect=[
            torch.tensor([[0.8, 0.2], [0.4, 0.6]]),
            torch.tensor([[0.7, 0.3], [0.5, 0.5]])
        ]
    )

    input_ids = torch.randint(0, 100, (4, 16))
    attention_mask = torch.ones((4, 16))

    predictions = model._raw_predict(input_ids, attention_mask, batch_size=2)
    assert len(predictions) == 4  # Total predictions match input size

def test_process_batch(model):
    # Mock parameters with a valid generator of Parameters
    model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.empty(1)) for _ in range(2)]))

    # Mock forward method to return valid logits
    model.forward = MagicMock(return_value=MagicMock(logits=torch.rand(4, model.num_labels)))

    # Set required attributes
    model.return_probabilities = True  # Ensure probabilities are returned
    model.threshold = 0.5  # Required for multi-label classification

    batch_input_ids = torch.randint(0, 100, (4, 16))
    batch_attention_mask = torch.ones((4, 16))

    # Process batch
    predictions = model.process_batch(batch_input_ids, batch_attention_mask)
    assert predictions.shape == (4, model.num_labels)  # Check for the expected shape


