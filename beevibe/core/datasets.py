import torch
from torch.utils.data import Dataset
import pandas as pd

class BeeTextDataset(Dataset):
    """
    A Dataset class for multi-class text classification.

    Attributes:
        texts (list of str): The list of text samples.
        labels (list of int): The list of corresponding labels.
        hftokenizer (bee.models.HFTokenizer): Tokenizer to preprocess the text samples.
        ultilabel (Bool): Multi-label classification ? True/False
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


def transform_to_multilabel_by_top_labels(df, text_column, label_columns, nb_top_labels=10):
    """
    Transforms label columns into multi-label binary columns (0/1) for multi-label classification.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the text and label columns.
        text_column (str): The name of the column containing the text data.
        label_columns (list of str): A list of column names containing the labels for each sample.
        nb_top_labels (int, optional): The number of most frequent labels to retain. 
                                       If None, all labels are retained. Default is 10.

    Returns:
        tuple:
            - multilabel_df (pandas.DataFrame): A DataFrame containing the original text column and one binary column 
              for each selected label (0/1 indicating the presence of the label).
            - selected_labels_dict (dict): A dictionary mapping each selected label to its count in the dataset 
              (how many times the label appears in the dataset).

    Example:
        transform_to_multilabel_by_top_labels(df, 'text', ['label1', 'label2', 'label3'], nb_top_labels=5)
    """
    # Step 1: Gather all labels from the label columns and count their frequencies.
    labels_series = df[label_columns].stack().dropna()
    label_counts = labels_series.value_counts()
    
    # Select labels based on nb_top_labels: use all if nb_top_labels is None, else the top nb_top_labels.
    if nb_top_labels is None:
        top_labels = label_counts.index.tolist()
    else:
        top_labels = label_counts.head(nb_top_labels).index.tolist()
    
    # Create a dictionary with the selected labels and their counts.
    selected_labels_dict = {label: int(label_counts[label]) for label in top_labels}
    
    # Step 2: Create a DataFrame for the binary label columns in a vectorized manner.
    # For each selected label, check if it appears in any of the label_columns for each row.
    label_data = {
        label: df[label_columns].isin([label]).any(axis=1).astype(int)
        for label in top_labels
    }
    labels_df = pd.DataFrame(label_data)
    
    # Step 3: Concatenate the text column and the binary labels DataFrame all at once.
    multilabel_df = pd.concat([df[[text_column]].reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1)
    
    return multilabel_df, selected_labels_dict


def transform_to_multilabel_by_label_names(df, text_column, label_columns, labels_to_use=[]):
    """
    Transforms label columns into multi-label binary columns (0/1) for multi-label classification.
    
    If labels_to_use is provided (i.e. a non-empty list), the resulting DataFrame will contain columns
    for each label in that list, in the given order. If a label in the list is not present in any of the 
    specified label columns, its column will be added with all 0s. If labels_to_use is empty, then all 
    labels found in the dataset (across the given label_columns) will be used.
    
    Parameters:
      - df: pandas.DataFrame
            The input DataFrame containing the text and label columns.
      - text_column: str
            The name of the column containing the text data.
      - label_columns: list of str
            A list of column names containing class labels.
      - labels_to_use: list of str, optional (default=[])
            A list of label names to use (in the desired order). If empty, all unique labels from 
            label_columns are used.
    
    Returns:
      - multilabel_df: pandas.DataFrame
            A new DataFrame with the text column and one binary column (0/1) for each selected label.
      - selected_labels_dict: dict
            A dictionary mapping each selected label to its count in the dataset (or 0 if not present).
    """
    # Step 1: Gather all labels from the specified label columns and count their frequencies.
    labels_series = df[label_columns].stack().dropna()
    label_counts = labels_series.value_counts()
    
    # Determine the labels to use:
    if labels_to_use:
        # Use the provided list, preserving order. Even if a label is absent in the data, include it.
        selected_labels = labels_to_use
    else:
        # If no list is provided, use all labels found in the dataset.
        selected_labels = label_counts.index.tolist()
    
    # Build a dictionary mapping each selected label to its count (or 0 if not found).
    selected_labels_dict = {label: int(label_counts[label]) if label in label_counts.index else 0
                            for label in selected_labels}
    
    # Step 2: Build the binary columns in a vectorized way.
    # For each selected label, check across all specified label_columns if it is present in a row.
    label_data = {
        label: df[label_columns].isin([label]).any(axis=1).astype(int)
        for label in selected_labels
    }
    labels_df = pd.DataFrame(label_data)
    
    # Step 3: Concatenate the text column and the binary labels DataFrame.
    multilabel_df = pd.concat(
        [df[[text_column]].reset_index(drop=True), labels_df.reset_index(drop=True)],
        axis=1
    )
    
    return multilabel_df, selected_labels_dict

