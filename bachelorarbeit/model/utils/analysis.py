from collections import defaultdict
import logging
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pkg_resources import resource_filename
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer
from bachelorarbeit.model import logger


def save_value_count(df: pd.DataFrame, strategy: str, set_: str):
    logger.info(f"{strategy} --> {set_}: Value Count of Levels \n{df[['score', 'language', 'source']].value_counts()}")
    df = df[['score', 'language', 'source']].value_counts().reset_index()
    df.to_csv(resource_filename(__name__, f'../../../cache/{strategy}_{set_}_sentiment_count.csv'), sep=';', index_label='index')
    return None


def explore_dataframe(df: pd.DataFrame, strategy: str, set_: str):
    logger.info(f'{strategy} --> {set_}: Shape of the DataFrame: {df.shape}')
    logger.info(f'{strategy} --> {set_}: Memory Consumption of DataFrame in MB: {df.memory_usage(deep=True).sum() / 1000000}')
    save_value_count(df=df, strategy=strategy, set_=set_)
    return None


def plot_countplot(df_series: pd.Series, strategy: str, set_: str, class_names=None, xlabel=None) -> None:
    sns.set_style('ticks')
    sns.set_palette('flare')
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.countplot(x=df_series, ax=ax).set_title(f'Distribution of Sentiment - {strategy} - {set_}')
    if xlabel:
        ax.set(xlabel=xlabel)
    if class_names:
        ax.set_xticklabels(class_names)
    fig.savefig(resource_filename(__name__, f'../../../cache/{strategy}_analyse_sentiment_{set_}.png'))
    return None


def analyse_sentiment(df: pd.DataFrame, strategy: str, set_: str) -> None:
    class_names = ['negative', 'neutral', 'positive']
    plot_countplot(df_series=df.score, class_names=class_names, xlabel=f'{strategy} review sentiment',
                   strategy=strategy, set_=set_)
    logger.info(f'{strategy} --> {set_}: Sentiment Analysis {df.shape}')
    return None


def explore_data(df: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, strategy: str) -> None:
    logger.info('Analysing Training and Test Data:')
    explore_dataframe(df=df, strategy=strategy, set_='train')
    explore_dataframe(df=df_val, strategy=strategy, set_='validation')
    explore_dataframe(df=df_test, strategy=strategy, set_='test')
    analyse_sentiment(df=df, strategy=strategy, set_='train')
    analyse_sentiment(df=df_val, strategy=strategy, set_='validation')
    analyse_sentiment(df=df_test, strategy=strategy, set_='test')
    return None


def plot_distribution(series: list, strategy: str, xlim: list = None, xlabel: str = None) -> None:
    sns.set_style('ticks')
    sns.set_palette('flare')
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.histplot(series, ax=ax).set_title(f'Distribution of Token Length')
    if xlim:
        ax.set_xlim(xlim)
    if xlabel:
        ax.set(xlabel=xlabel)
    fig.savefig(resource_filename(__name__, f'../../../cache/{strategy}_plot_distribution.png'))
    return None


def analyse_sequence_length(df_series: pd.Series, tokenizer: BertTokenizer, strategy: str) -> int:
    # TODO: analyse max_len and use later in create_data_loader
    token_lens = []
    # use df.text here:
    for txt in df_series:
        tokens = tokenizer.encode(txt, max_length=256)
        token_lens.append(len(tokens))
    plot_distribution(token_lens, xlim=[0, 256], xlabel=f'{strategy}Token count', strategy=strategy)
    max_len = 256
    return max_len


def plot_training_results(history: defaultdict, model_name: str):
    acc_df = pd.DataFrame({'epoch': history['epochs'],
                           'training_accuracy': history['train_acc'],
                           'validation_accuracy': history['val_acc']})
    loss_df = pd.DataFrame({'epoch': history['epochs'],
                            'training_loss': history['train_loss'],
                            'validation_loss': history['val_loss']})
    plot_training_accuracy(acc_df, model_name)
    plot_training_loss(loss_df, model_name)
    save_training_results(acc_df, loss_df, model_name=model_name)
    return None


def save_training_results(acc_df: pd.DataFrame, loss_df: pd.DataFrame, model_name: str):
    training_report_df = pd.concat([acc_df, loss_df], axis=1, sort=False)
    training_report_df['training_accuracy'] = training_report_df['training_accuracy'].astype('float')
    training_report_df['validation_accuracy'] = training_report_df['validation_accuracy'].astype('float')
    training_report_df.to_csv(resource_filename(__name__, f'../../../cache/{model_name}_training_report.csv'), sep=';',
                              index_label='index')
    return None


def plot_training_accuracy(df: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    ax.plot(df['training_accuracy'], label='training accuracy')
    ax.plot(df['validation_accuracy'], label='validation accuracy')
    ax.set(xlabel='Epoch', ylabel='Accuracy', title=f'Training Function - {model_name}', ylim=[0, 1])
    fig.legend()
    fig.savefig(resource_filename(__name__, f'../../../cache/{model_name}_accuracy_epoch.png'))
    return None


def plot_training_loss(df: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    ax.plot(df['training_loss'], label='training loss')
    ax.plot(df['validation_loss'], label='validation loss')
    ax.set(xlabel='Epoch', ylabel='Loss', title=f'Loss Function - {model_name}')
    fig.legend()
    fig.savefig(resource_filename(__name__, f'../../../cache/{model_name}_loss_epoch.png'))
    return None


def plot_confusion_matrix(real_values: list, predictions: list, class_names: list, model_name: str):
    cm_df = pd.DataFrame(
        confusion_matrix(real_values, predictions),
        index=class_names,
        columns=class_names
    )
    sns.set_style('ticks')
    sns.set_palette('flare')
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.heatmap(cm_df, annot=True, fmt="d", ax=ax).set_title(f'Confusion Matrix of Test Results - {model_name}')
    ax.set(xlabel='Predicted sentiment', ylabel='Actual sentiment')
    fig.savefig(resource_filename(__name__, f'../../../cache/{model_name}_confusion_matrix.png'))
    return None


def save_test_reports(test_acc: list, test_input: list, predictions: list, prediction_probs: list, actual_values: list,
                      class_names: list, model_name: str):
    logger.info('---------------------------')
    logger.info(f"{model_name} --> The accuracy on the test data: {test_acc}")
    logger.info(f"{model_name} --> Classification Report\n{classification_report(actual_values, predictions, target_names=class_names)}")
    logger.info('---------------------------')
    save_classification_report(
        actual_values=actual_values,
        predictions=predictions,
        class_names=class_names,
        model_name=model_name)
    save_test_report(
        test_input=test_input, actual_values=actual_values,
        predictions=predictions,
        prediction_probs=prediction_probs,
        model_name=model_name)
    return None


def save_classification_report(actual_values: list, predictions: list, class_names: list, model_name: str):
    report = classification_report(actual_values, predictions, target_names=class_names, output_dict=True)
    pd.DataFrame(report).to_csv(
        resource_filename(__name__, f'../../../cache/{model_name}_classification_report.csv'), sep=';',
        index_label='metric')
    return None


def save_test_report(test_input: list, actual_values: list, predictions: list, prediction_probs: list, model_name: str):
    test_report_df = pd.DataFrame(
        {'actual_value': actual_values,
         'prediction': predictions,
         'prediction_probabilities': prediction_probs,
         'test_input': test_input})
    test_report_df.to_csv(resource_filename(__name__, f'../../../cache/{model_name}_test_report.csv'), sep=';',
                          index_label='index')
    return None
