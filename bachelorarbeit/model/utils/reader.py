import pandas as pd
from pkg_resources import resource_filename
from bachelorarbeit.model import logger


def read_cache(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(resource_filename(__name__, file_path), sep=';', names=['score', 'text', 'language', 'source'])
    if len(df) < 2:
        raise IOError
    else:
        df = df[df['text'].str.len() > 2]
    return df


def get_training_data(strategy: str, balanced: bool):
    if strategy == 'ger_only':
        if balanced:
            logger.info('Loading Training Data: "german_sink_train_balanced.csv"')
            df = read_cache(file_path='../../../sink/german_sink_train_balanced.csv')
        else:
            logger.info('Loading Training Data: "german_sink_train.csv"')
            df = read_cache(file_path='../../../sink/german_sink_train.csv')
    elif strategy == 'multi_noger':
        if balanced:
            logger.info('Loading Training Data: "multi_lang_noger_sink_balanced.csv"')
            df = read_cache(file_path='../../../sink/multi_lang_noger_sink_balanced.csv')
        else:
            logger.info('Loading Training Data: "multi_lang_noger_sink.csv"')
            df = read_cache(file_path='../../../sink/multi_lang_noger_sink.csv')
    elif strategy == 'multi_all':
        if balanced:
            logger.info('Loading Training Data: "multi_lang_sink_balanced.csv"')
            df = read_cache(file_path='../../../sink/multi_lang_sink_balanced.csv')
        else:
            logger.info('Loading Training Data: "multi_lang_sink.csv"')
            df = read_cache(file_path='../../../sink/multi_lang_sink.csv')

    elif strategy in ['multi_noenglish', 'multi_noarabic', 'multi_nochinese', 'multi_nopolish', 'multi_nofrench', 'multi_nodutch']:
        key = strategy.split("_")[1]
        logger.info(f'Loading Training Data: "multi_lang_{key}_sink.csv"')
        df = read_cache(file_path=f'../../../sink/multi_lang_{key}_sink.csv')

    elif strategy in ['only_german', 'only_english', 'only_arabic', 'only_chinese', 'only_dutch', 'only_french', 'only_polish']:
        key = strategy.split("_")[1]
        logger.info(f'Loading Training Data: "{key}_sink.csv"')
        df = read_cache(file_path=f'../../../sink/{key}_sink.csv')

    else:
        logger.error(NotImplementedError)
        raise NotImplementedError
    return df


def get_test_data(balanced: bool):
    if balanced:
        vali_path = '../../../sink/german_sink_validation_balanced.csv'
        test_path = '../../../sink/german_sink_test_balanced.csv'
        logger.info('Loading Balanced Validation and Test Data.')
    else:
        vali_path = '../../../sink/german_sink_validation.csv'
        test_path = '../../../sink/german_sink_test.csv'
        logger.info('Loading Validation and Test Data.')
    df_validation = read_cache(file_path=vali_path)
    df_test = read_cache(file_path=test_path)
    logger.info(f'Shape of Validation Set with balanced={balanced}: {df_validation.shape}')
    logger.info(f'Shape of Test Set with balanced={balanced}: {df_test.shape}')
    return df_validation, df_test