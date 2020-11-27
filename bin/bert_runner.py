#!/usr/bin/env python
import argparse

from bachelorarbeit.model.preprocessor import preprocess
from bachelorarbeit.model.testing import test
from bachelorarbeit.model.training import get_model, train
from bachelorarbeit.model.utils.analysis import explore_data
from bachelorarbeit.model.utils.reader import read_cache

# README:
# DATA SHUFFLING
# https://stackoverflow.com/questions/54354465/impact-of-using-data-shuffling-in-pytorch-dataloader
# https://pytorch.org/docs/stable/data.html


class_names = ['negative', 'neutral', 'positive']
df_test_ger = read_cache(file_path='../../../cache/german_sink_test.csv')
strategies = ['multi_noger', 'multi_all', 'ger_only']
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', nargs='+', default=['ger_only'], help='What is the list of languages?')
parser.add_argument('--epochs', type=int, default=2, help='The number of training iterations')
parser.add_argument('--shuffle', type=bool, default=False, help='If turn on the shuffle')
args = parser.parse_args()
print(args)


def main():
    for strategy in args.strategy:
        print("-------------------------------")
        print(f"Training model: {strategy}")
        print("-------------------------------")
        if strategy in strategies:
            if strategy == 'ger_only':
                df = read_cache(file_path='../../../cache/german_sink_train.csv')
            elif strategy == 'multi_noger':
                df = read_cache(file_path='../../../cache/multi_lang_noger_sink.csv')
            elif strategy == 'multi_all':
                df = read_cache(file_path='../../../cache/multi_lang_sink.csv')
            else:
                raise NotImplementedError
            explore_data(
                df=df,
                strategy=strategy
            )
            train_data_loader, val_data_loader, test_data_loader, df_validation, df_test = preprocess(
                df_train=df,
                df_test=df_test_ger,
                strategy=strategy,
                shuffle=args.shuffle,
                random_seed=42,
                batch_size=16,
                validation_size_ratio=0.5,
                num_workers=4
            )
            model = get_model(class_names)
            best_model = train(
                model=model,
                train_data_loader=train_data_loader,
                val_data_loader=val_data_loader,
                training_set=df,
                validation_set=df_validation,
                epochs=args.epochs,
                model_name=strategy
            )
            test(
                df_test=df_test_ger,
                best_model=best_model,
                test_data_loader=test_data_loader,
                class_names=class_names,
                model_name=strategy
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
