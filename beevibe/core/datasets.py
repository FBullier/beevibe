import torch
from torch.utils.data import Dataset

class BeeTextDataset(Dataset):
    """
    A Dataset class for multi-class text classification.

    Attributes:
        texts (list of str): The list of text samples.
        labels (list of int): The list of corresponding labels.
        hftokenizer (bee.models.HFTokenizer): Tokenizer to preprocess the text samples.
    """

    def __init__(self, texts: list[str], labels: list[int], hftokenizer, multilabel):
        """
        Initializes the TextDatasetMC class with the given texts, labels, hftokenizer, and maximum sequence length.

        Args:
            texts (list of str): Text samples.
            labels (list of int): Corresponding labels.
            hftokenizer (beevive.models.HFTokenizer): Tokenizer for text preprocessing.
            multilabel (Bool): Multi-label classification ? True/False
        """
        self.texts = texts
        self.labels = labels
        self.hftokenizer = hftokenizer
        self.multilabel = multilabel

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

        encoded_batch = self.hftokenizer.encode_plus(text)
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        ret = {
            "text": text,
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten(),
        }

        if self.multilabel:
            ret["label"] = torch.tensor(label, dtype=torch.float)
        else:
            ret["label"] = torch.tensor(label, dtype=torch.long)

        return ret
