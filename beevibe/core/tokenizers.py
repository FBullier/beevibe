import json
from transformers import AutoTokenizer
from typing import Optional

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
            model_directory = kwargs.get("model_directory")
            self.load_config(model_directory)
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
        """
            Save the tokenizer to the specified path.

            Args:
                path (str): The directory where the tokenizer will be saved.

            Returns:
                None: This method does not return any value, but saves the tokenizer to the specified path.
            """        
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        else:
            assert "Tokenizer is not define, call from_pretrained()"


    def save_config(self, path):
        """
            Save the tokenizer's preprocessing configuration to the specified path.

            Args:
                path (str): The directory where the preprocessing configuration will be saved.

            Returns:
                None: This method does not return any value, but saves the configuration to the specified path.
            """

        if self.tokenizer:
            if self.preprocessing_config:
                with open(f"{path}/preprocessing_config.json", "w") as f:
                    json.dump(self.preprocessing_config, f)
            else:
                assert "Tokenizer preprocessing configuration is not define"
        else:
            assert "Tokenizer is not define, call from_pretrained()"


    def load_config(self, path):
        """
        Load the tokenizer's preprocessing configuration from the specified path.

        Args:
            path (str): The directory from which the preprocessing configuration will be loaded.

        Returns:
            None: This method does not return any value, but loads the configuration into `self.preprocessing_config`.
        """
        try:
            with open(f"{path}/preprocessing_config.json", "r") as f:
                self.preprocessing_config = json.load(f)
        except FileNotFoundError:
            self.preprocessing_config = None


    def encode(self, raw_texts: str):
        """
        Encode a batch of raw texts using the tokenizer.

        Args:
            raw_texts (str): A string of raw text to be tokenized.

        Returns:
            tuple: A tuple containing:
                - input_ids (torch.Tensor): The tokenized input IDs.
                - attention_mask (torch.Tensor): The attention mask for the tokenized input.
        """

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
        """
        Encode a batch of raw texts using the tokenizer with `encode_plus`.

        Args:
            raw_texts (str): A string of raw text to be tokenized.

        Returns:
            dict: A dictionary containing the encoded information, including 'input_ids', 'attention_mask', and possibly
                other tokenizer-specific outputs.
        """

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
