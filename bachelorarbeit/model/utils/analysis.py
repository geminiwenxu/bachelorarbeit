import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from transformers import BertTokenizer

from pkg_resources import resource_filename


def explore_dataframe(df):
    print('Shape of the DataFrame: ', df.shape)
    print('Information about the DataFrame:')
    df.info()


def plot_countplot(df_series: pd.Series, language: str, class_names=None, xlabel=None) -> None:
    ax = sns.countplot(x=df_series)
    if xlabel:
        plt.xlabel(xlabel)
    if class_names:
        ax.set_xticklabels(class_names)
    plt.savefig(resource_filename(__name__, f'../../../cache/{language}_analyse_sentiment.png'))


def analyse_sentiment(df: pd.DataFrame, strategy: str) -> None:
    class_names = ['negative', 'neutral', 'positive']
    plot_countplot(df_series=df.score, class_names=class_names, xlabel=f'{strategy} review sentiment',
                   language=strategy)


def explore_data(df: pd.DataFrame, strategy: str) -> None:
    explore_dataframe(df)
    analyse_sentiment(df, strategy)


def plot_distribution(series: list, strategy: str, xlim: list = None, xlabel: str = None) -> None:
    sns.displot(series)
    if xlim:
        plt.xlim(xlim)
    if xlabel:
        plt.xlabel(xlabel)
    plt.savefig(resource_filename(__name__, f'../../../cache/{strategy}_plot_distribution.png'))


def analyse_sequence_length(df_series: pd.Series, tokenizer: BertTokenizer, strategy: str) -> int:
    # TODO: analyse max_len and use later in create_data_loader
    token_lens = []
    # use df.text here:
    for txt in df_series:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))
    plot_distribution(token_lens, xlim=[0, 256], xlabel=f'{strategy}Token count', strategy=strategy)
    max_len = 256
    return max_len


def accuracy_epoch(history, model_name):
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    return plt.savefig(resource_filename(__name__, f'../../../cache/accuracy_epoch_{model_name}.png'))


def show_confusion_matrix(confusion_matrix, model_name):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig(resource_filename(__name__, f'../../../cache/show_confusion_matrix{model_name}.png'))
