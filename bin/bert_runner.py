#!/usr/bin/env python
import argparse
import torch
import os
import logging
from datetime import datetime
from pkg_resources import resource_filename
from torch import nn
from bachelorarbeit.model.classifier import SentimentClassifier
from bachelorarbeit.model.preprocessor import preprocess
from bachelorarbeit.model.testing import test
from bachelorarbeit.model.training import train
from bachelorarbeit.model.utils.analysis import explore_data
from bachelorarbeit.model.utils.reader import read_cache

# README:
# DATA SHUFFLING
# https://stackoverflow.com/questions/54354465/impact-of-using-data-shuffling-in-pytorch-dataloader
# https://pytorch.org/docs/stable/data.html

# CUDA MULTIPROCESSING
# https://discuss.pytorch.org/t/difference-between-torch-device-cuda-and-torch-device-cuda-0/46306/20
# https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch

class_names = ['negative', 'neutral', 'positive']
strategies = ['multi_noger', 'multi_all', 'ger_only']
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', nargs='+', choices=strategies, help='What is the list of languages?')
parser.add_argument('--balanced_training', default=False, action='store_true', help='Using balanced training sets')
parser.add_argument('--balanced_testing', default=False, action='store_true', help='Using balanced test sets')
parser.add_argument('--epochs', type=int, default=2, choices=range(1, 100), help='The number of training iterations')
parser.add_argument('--shuffle', default=False, action='store_true', help='If turn on the shuffle')
args = parser.parse_args()


log_params = '-'.join(args.strategy)

# Setup file logging
log_datetime = datetime.now().isoformat()
log_path = resource_filename(__name__, '../logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)
file_log_handler = logging.FileHandler(f'{log_path}/{log_datetime}_bert_{log_params}.log')
logger = logging.getLogger(__name__)
logger.addHandler(file_log_handler)
logger.setLevel('INFO')
stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)


# Setup Building Sentiment Classifier starts here ----------------------------------------

def setup_model(class_names) -> tuple:
    logger.info('Setting up Model:')
    if torch.cuda.is_available():
        logger.info('CUDA is available, setting up Tensors to work with CUDA')
        device = torch.device("cuda")
    else:
        logger.info('CUDA is NOT available, setting up Tensors to work with CPU')
        device = torch.device("cpu")

    model = SentimentClassifier(n_classes=len(class_names))
    if torch.cuda.device_count() > 1:
        logger.info(f'"{torch.cuda.device_count()}" CUDA devices available, trying to utilise DataParallel()')
        model = nn.DataParallel(model)
    else:
        logger.info('CUDA is NOT available, NOT using DataParallel()')
    model = model.to(device)
    return model, device


def get_training_data(strategy: str, balanced: bool):
    if strategy == 'ger_only':
        if balanced:
            logger.info('Loading Training Data: "german_sink_train_balanced.csv"')
            df = read_cache(file_path='../../../cache/german_sink_train_balanced.csv')
        else:
            logger.info('Loading Training Data: "german_sink_train.csv"')
            df = read_cache(file_path='../../../cache/german_sink_train.csv')
    elif strategy == 'multi_noger':
        if balanced:
            logger.info('Loading Training Data: "multi_lang_noger_sink_balanced.csv"')
            df = read_cache(file_path='../../../cache/multi_lang_noger_sink_balanced.csv')
        else:
            logger.info('Loading Training Data: "multi_lang_noger_sink.csv"')
            df = read_cache(file_path='../../../cache/multi_lang_noger_sink.csv')
    elif strategy == 'multi_all':
        if balanced:
            logger.info('Loading Training Data: "multi_lang_sink_balanced.csv"')
            df = read_cache(file_path='../../../cache/multi_lang_sink_balanced.csv')
        else:
            logger.info('Loading Training Data: "multi_lang_sink.csv"')
            df = read_cache(file_path='../../../cache/multi_lang_sink.csv')
    else:
        logger.error(NotImplementedError)
        raise NotImplementedError
    return df


def get_test_data(balanced: bool):
    if balanced:
        vali_path = '../../../cache/german_sink_validation_balanced.csv'
        test_path = '../../../cache/german_sink_test_balanced.csv'
        logger.info('Loading Validation and Test Data: "german_sink_test_balanced.csv"')
    else:
        vali_path = '../../../cache/german_sink_validation.csv'
        test_path = '../../../cache/german_sink_test.csv'
        logger.info('Loading Validation and Test Data: "german_sink_test.csv"')
    df_validation = read_cache(file_path=vali_path)
    df_test = read_cache(file_path=test_path)
    logger.info(f'Shape of Validation Set with balanced={balanced}: {df_validation.shape}', df_validation.shape)
    logger.info(f'Shape of Test Set with balanced={balanced}: {df_test.shape}')
    return df_validation, df_test


def main():
    logger.info('Launching bert_runner.py with argparser arguments: ', args)
    for strategy in args.strategy:
        logger.info("\n-------------------------------")
        logger.info(f" Start to work on model: {strategy} with balanced_training: {args.balanced_training}, and balanced_testing: {args.balanced_testing}, shuffle: {args.shuffle}, epochs: {args.epochs}")
        logger.info("-------------------------------\n")
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
                strategy=strategy,
                logger=logger
            )
            train_data_loader, val_data_loader, test_data_loader = preprocess(
                df_train=df,
                df_validate=df_validate_ger,
                df_test=df_test_ger,
                strategy=strategy,
                shuffle=args.shuffle,
                batch_size=32,
                num_workers=4,
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
                logger=logger
            )
            test(
                df_test=df_test_ger,
                test_data_loader=test_data_loader,
                device=device,
                class_names=class_names,
                model_name=strategy,
                logger=logger
            )
        else:
            logger.error(NotImplementedError)
            raise NotImplementedError


if __name__ == "__main__":
    main()
