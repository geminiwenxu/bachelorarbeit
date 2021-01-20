#!/usr/bin/env python
import argparse

import numpy as np
import torch
import yaml
from pkg_resources import resource_filename
from torch import nn

from bachelorarbeit.model import logger
from bachelorarbeit.model.classifier import SentimentClassifier
from bachelorarbeit.model.preprocessor import preprocess
from bachelorarbeit.model.testing import test
from bachelorarbeit.model.training import train
from bachelorarbeit.model.utils.analysis import explore_data
from bachelorarbeit.model.utils.reader import get_training_data, get_test_data


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
np.random.seed(config['random_seed'])
torch.manual_seed(config['random_seed'])

class_names = config['class_names']
strategies = config['strategies']
parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy",
    nargs="+",
    default=['ger_only'],
    choices=strategies,
    help="What is the list of languages?"
)
parser.add_argument(
    "--balanced_training",
    default=False,
    action="store_true",
    help="Using balanced training sets"
)
parser.add_argument(
    "--balanced_testing",
    default=False,
    action="store_true",
    help="Using balanced test sets"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=config['epochs'],
    choices=range(1, 100),
    help="The number of training iterations"
)
parser.add_argument(
    "--shuffle",
    default=config['shuffle'],
    action="store_true",
    help="If turn on the shuffle"
)
args = parser.parse_args()


def setup_model(class_names) -> tuple:
    # README:
    # DATA SHUFFLING
    # https://stackoverflow.com/questions/54354465/impact-of-using-data-shuffling-in-pytorch-dataloader
    # https://pytorch.org/docs/stable/data.html

    # CUDA MULTIPROCESSING
    # https://discuss.pytorch.org/t/difference-between-torch-device-cuda-and-torch-device-cuda-0/46306/20
    # https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch
    logger.info("Setting up Model:")
    if torch.cuda.is_available():
        logger.info("CUDA is available, setting up Tensors to work with CUDA")
        device = torch.device("cuda")
    else:
        logger.info("CUDA is NOT available, setting up Tensors to work with CPU")
        device = torch.device("cpu")

    model = SentimentClassifier(n_classes=len(class_names),
                                dropout_ratio=config['dropout_ratio']
                                )
    if torch.cuda.device_count() > 1:
        logger.info(f"'{torch.cuda.device_count()}' CUDA devices available, trying to utilise DataParallel()")
        model = nn.DataParallel(model)
    else:
        logger.info("CUDA is NOT available, NOT using DataParallel()")
    model = model.to(device)
    return model, device


def main():
    try:
        logger.info(f"Launching bert_runner.py with argparser arguments: {args}")
        for strategy in args.strategy:
            logger.info("-------------------------------")
            logger.info(
                f"Start to work on model: {strategy} with balanced_training: {args.balanced_training}, and balanced_testing: {args.balanced_testing}, shuffle: {args.shuffle}, epochs: {args.epochs}")
            logger.info("-------------------------------")
            if strategy in strategies:
                df = get_training_data(
                    strategy=strategy,
                    balanced=args.balanced_training
                )
                df_validate_ger, df_test_ger = get_test_data(
                    balanced=args.balanced_testing
                )
                explore_data(
                    df=df,
                    df_val=df_validate_ger,
                    df_test=df_test_ger,
                    strategy=strategy
                )
                train_data_loader, val_data_loader, test_data_loader = preprocess(
                    df_train=df,
                    df_validate=df_validate_ger,
                    df_test=df_test_ger,
                    strategy=strategy,
                    shuffle=args.shuffle,
                    batch_size=config['batch_size'],
                    num_workers=config['num_workers'],
                    max_token_lengths=config['max_len'],
                    logger=logger
                )
                model, device = setup_model(class_names)
                train(
                    model=model,
                    device=device,
                    train_data_loader=train_data_loader,
                    val_data_loader=val_data_loader,
                    training_set=df,
                    validation_set=df_validate_ger,
                    epochs=args.epochs,
                    model_name=strategy,
                    correct_bias=config['correct_bias'],
                    learning_rate=config['learning_rate'],
                    num_warmup_steps=config['num_warmup_steps']
                )
                test(
                    df_test=df_test_ger,
                    test_data_loader=test_data_loader,
                    device=device,
                    class_names=class_names,
                    model_name=strategy
                )
            else:
                logger.error(NotImplementedError)
                raise NotImplementedError
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()
