import torch
from torch.utils.data import Dataset

class TextDatasetMC(Dataset):
    """
    A Dataset class for multi-class text classification.

    Attributes:
        texts (list of str): The list of text samples.
        labels (list of int): The list of corresponding labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the text samples.
        max_len (int): Maximum sequence length for tokenization.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        """
        Initializes the TextDatasetMC class with the given texts, labels, tokenizer, and maximum sequence length.

        Args:
            texts (list of str): Text samples.
            labels (list of int): Corresponding labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text preprocessing.
            max_len (int, optional): Maximum sequence length for tokenization. Default is 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and label.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }

class TextDatasetML(Dataset):
    """
    A Dataset class for multi-label text classification.

    Attributes:
        texts (list of str): The list of text samples.
        labels (list of list of int): The list of corresponding multi-labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess the text samples.
        max_len (int): Maximum sequence length for tokenization.
    """

    def __init__(self, texts: list[str], labels: list[list[int]], tokenizer, max_len: int = 128):
        """
        Initializes the TextDatasetML class with the given texts, labels, tokenizer, and maximum sequence length.

        Args:
            texts (list of str): Text samples.
            labels (list of list of int): Corresponding multi-labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text preprocessing.
            max_len (int, optional): Maximum sequence length for tokenization. Default is 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and multi-label.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.float),
        }
