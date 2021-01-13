import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from bachelorarbeit.model.utils.analysis import analyse_sequence_length
from bachelorarbeit.model import logger

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


class BuiltPyTorchDataset(Dataset):

    def __init__(self, texts: np.array, scores: np.array, tokenizer: BertTokenizer, max_len: int) -> None:
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item) -> dict:
        text = str(self.texts[item])
        score = self.scores[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {'input_text': text, 'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(), 'scores': torch.tensor(score, dtype=torch.long)}


def split_data(df: pd.DataFrame, random_seed: int, validation_size_ratio: float, logger) -> tuple:
    df_validation, df_test = train_test_split(df,
                                              test_size=1 - validation_size_ratio,
                                              random_state=random_seed,
                                              stratify=df.score
                                              )
    logger.info('Shape of Validation DataFrame: ', df_validation.shape)
    logger.info('Shape of Test DataFrame: ', df_test.shape)
    return df_validation, df_test


def setup_pytorch_dataset(df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, batch_size: int, num_workers: int,
                          shuffle: bool) -> DataLoader:
    ds = BuiltPyTorchDataset(
        texts=df.text.to_numpy(),
        scores=df.score.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def declare_pytorch_loader(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, tokenizer: BertTokenizer,
                           max_len: int, batch_size: int, num_workers: int, shuffle: bool) -> tuple:
    train_data_loader = setup_pytorch_dataset(
        train, tokenizer,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    val_data_loader = setup_pytorch_dataset(
        validation,
        tokenizer,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    test_data_loader = setup_pytorch_dataset(
        test, tokenizer,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return train_data_loader, val_data_loader, test_data_loader


def preprocess(df_train: pd.DataFrame, df_validate: pd.DataFrame, df_test: pd.DataFrame, strategy: str, shuffle: bool, logger,
               batch_size: int = 32, num_workers: int = 4) -> tuple:
    # TODO: get max_len from analyse_sequence .. automatically!
    logger.info(f'{strategy} --> Setting up Data Processing Pipeline for PyTorch with parameters: shuffle={shuffle}, batch_size={batch_size}, num_workers={num_workers}')
    max_len = analyse_sequence_length(
        df_series=df_train.text,
        tokenizer=tokenizer,
        strategy=strategy
    )
    train_data_loader, val_data_loader, test_data_loader = declare_pytorch_loader(
        train=df_train,
        validation=df_validate,
        test=df_test,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return train_data_loader, val_data_loader, test_data_loader
