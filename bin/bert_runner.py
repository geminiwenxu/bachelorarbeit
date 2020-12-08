#!/usr/bin/env python
import argparse
import torch
from torch import nn
from bachelorarbeit.model.classifier import SentimentClassifier
from bachelorarbeit.model.preprocessor import preprocess, split_data
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

class_names = ['negative', 'neutral', 'positive']
strategies = ['multi_noger', 'multi_all', 'ger_only']
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', nargs='+', default=['ger_only'], help='What is the list of languages?')
parser.add_argument('--epochs', type=int, default=2, choices=range(1, 100), help='The number of training iterations')
parser.add_argument('--shuffle', default=False, action='store_true', help='If turn on the shuffle')
args = parser.parse_args()
print('Implementing argparser with arguments: ', args)


# Setup Building Sentiment Classifier starts here ----------------------------------------

def setup_model(class_names) -> tuple:
    print('Setting up Model:')
    if torch.cuda.is_available():
        print('CUDA is available, setting up Tensors to work with CUDA')
        device = torch.device("cuda:0")
    else:
        print('CUDA is NOT available, setting up Tensors to work with CPU')
        device = torch.device("cpu")

    model = SentimentClassifier(n_classes=len(class_names))
    if torch.cuda.device_count() > 1:
        print(f'"{torch.cuda.device_count()}" CUDA devices available, trying to utilise DataParallel()')
        model = nn.DataParallel(model)
    else:
        print('CUDA is NOT available, NOT using DataParallel()')
    model = model.to(device)
    return model, device


def get_training_data(strategy: str):
    if strategy == 'ger_only':
        print('Loading Training Data: "german_sink_train.csv"')
        df = read_cache(file_path='../../../cache/german_sink_train.csv')
    elif strategy == 'multi_noger':
        print('Loading Training Data: "multi_lang_noger_sink.csv"')
        df = read_cache(file_path='../../../cache/multi_lang_noger_sink.csv')
    elif strategy == 'multi_all':
        print('Loading Training Data: "multi_lang_sink.csv"')
        df = read_cache(file_path='../../../cache/multi_lang_sink.csv')
    else:
        raise NotImplementedError
    return df


def get_test_data():
    print('Loading Test Data: "german_sink_test.csv"')
    return read_cache(file_path='../../../cache/german_sink_test.csv')


def main():
    for strategy in args.strategy:
        print("\n-------------------------------")
        print(f" Start working on: {strategy}")
        print("-------------------------------\n")
        if strategy in strategies:
            df = get_training_data(strategy=strategy)
            df_val_test_ger = get_test_data()
            df_validate_ger, df_test_ger = split_data(df_val_test_ger, random_seed=42, validation_size_ratio=0.5)

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
                batch_size=32,
                num_workers=4
            )
            model, device = setup_model(class_names)
            best_model = train(
                model=model,
                device=device,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                training_set=df,
                validation_set=df_validate_ger,
                epochs=args.epochs,
                model_name=strategy
            )
            test(
                df_test=df_test_ger,
                best_model=best_model,
                test_data_loader=test_data_loader,
                device=device,
                class_names=class_names,
                model_name=strategy
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
