import ast
import json
import os

import pandas as pd
from pkg_resources import resource_filename


class DataSource:
    colnames = ['score', 'text']

    def __init__(self, config, logger):
        self.source_path = config['source']['path']
        self.logger = logger

    def get_all_file_paths(self, sub_folder_path):
        base = resource_filename(__name__, self.source_path + sub_folder_path)
        out = []
        for path, subdirs, files in os.walk(base):
            for name in files:
                out.append((name, os.path.join(path, name)))
        return out

    def get_csv(self, file_path: str, file_config: dict) -> pd.DataFrame:
        return pd.read_csv(file_path, **file_config)

    def get_broken_json(self, file_path: str) -> dict:
        with open(file_path, "r") as broken_json:
            contents = broken_json.read()
            dictionary = ast.literal_eval(contents)
        return dictionary

    def get_json(self, file_path: str) -> dict:
        with open(file_path, encoding='utf-8', errors='ignore') as json_data:
            return json.load(json_data, strict=False)

    def get_text(self, file_path: str) -> list:
        with open(file_path, 'r') as file:
            return [line for line in file]

    def stream_large_txt(self, file_path: str) -> pd.DataFrame:
        with open(resource_filename(__name__, self.source_path + file_path), 'r', encoding="utf8",
                  errors='ignore') as infile:
            for line in infile:
                yield line

    def stream_large_csv(self, file_path: str, file_config: dict) -> pd.DataFrame:
        for chunk in pd.read_csv(resource_filename(__name__, self.source_path + file_path), **file_config):
            yield chunk


class SourceLocalDataEnglish(DataSource):
    sub_folder = "english/"

    def semeval_english(self) -> pd.DataFrame:
        source_path = 'SemEvalEnglish/GOLD/'
        substrings = ['twitter', '.txt']
        file_config = {'sep': '\t', 'parse_dates': True, 'header': None}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def amazon_movie_review(self) -> pd.DataFrame:
        file_path = self.sub_folder + 'AmazonMovieReview/movies.txt'
        return self.stream_large_txt(file_path=file_path)

    def webis_tripad(self) -> tuple:
        source_path = 'WebisTripad/'
        substrings = ['.json']
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_broken_json(file_path=file_path)


class SourceLocalDataArabic(DataSource):
    sub_folder = "arabic/"

    def semeval_arabic(self) -> pd.DataFrame:
        source_path = 'SemEvalArabic/GOLD/'
        substrings = ['task4', '.txt']
        file_config = {'sep': '\t', 'parse_dates': True}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)


class SourceLocalDataGerman(DataSource):
    sub_folder = "german/"

    def scare_german(self) -> pd.DataFrame:
        source_path = 'scare/'
        substrings = ['.csv']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': self.colnames, 'header': None, 'usecols': [1, 3]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def potts_german(self) -> pd.DataFrame:
        source_path = 'PotTS/preprocessed-no-noise-cleaner/'
        substrings = ['.tsv']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': self.colnames, 'header': None, 'usecols': [1, 2]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def sb_german(self) -> pd.DataFrame:
        source_path = 'SB10K/preprocessed/'
        substrings = ['.tsv']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': self.colnames, 'header': None, 'usecols': [1, 2]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def germeval(self) -> pd.DataFrame:
        source_path = 'germeval2017/'
        substrings = ['.tsv']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': ['text', 'score'], 'header': 0, 'usecols': [1, 3]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def filmstarts_german(self) -> pd.DataFrame:
        source_path = 'filmstarts/'
        substrings = ['.tsv']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': self.colnames, 'header': None, 'usecols': [1, 2]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def holidaycheck_german(self) -> pd.DataFrame:
        source_path = 'holidaycheck/'
        file_path = self.sub_folder + source_path + 'holidaycheck.clean.filtered.tsv'
        return self.stream_large_csv(file_path=file_path,
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': '\t', 'usecols': [0, 1],
                                                  'names': self.colnames})

    def leipzig_german(self) -> list:
        file_path2 = self.sub_folder + 'leipzig/deu-wikipedia-2016-labeled'
        return [
            self.stream_large_txt(file_path=file_path2)
        ]


class SourceLocalDataPolish(DataSource):
    sub_folder = "polish/"

    def polemo_polish(self) -> pd.DataFrame:
        source_path = 'PolEmo/'
        substrings = ['.txt']
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_text(file_path=file_path)


class SourceLocalDataChinese(DataSource):
    sub_folder = "chinese/"

    def douban_movies(self) -> pd.DataFrame:
        return self.stream_large_csv(file_path=self.sub_folder + 'douban/ratings.csv',
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': ',', 'usecols': [2, 4],
                                                  'names': self.colnames})


class SourceLocalDataFrench(DataSource):
    sub_folder = "french/"

    def kaggle(self) -> pd.DataFrame:
        return self.stream_large_csv(file_path=self.sub_folder + 'french_tweets.csv',
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': ',', 'names': self.colnames})

    def betsentiment_teams(self) -> pd.DataFrame:
        return self.stream_large_csv(file_path=self.sub_folder + 'betsentiment-FR-tweets-sentiment-teams.csv',
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': ',', 'names': ['text', 'score'],
                                                  'usecols': [2, 4], 'encoding': 'latin_1'})

    def betsentiment_worldcup(self) -> pd.DataFrame:
        return self.stream_large_csv(file_path=self.sub_folder + 'betsentiment-FR-tweets-sentiment-worldcup.csv',
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': ',', 'names': ['text', 'score'],
                                                  'usecols': [2, 4], 'encoding': 'latin_1'})


class SourceLocalDataDutch(DataSource):
    sub_folder = "dutch/"

    def social_media_collection(self) -> tuple:
        substrings = ['.json']
        files = self.get_all_file_paths(self.sub_folder + 'DutchSocialMediaCollection/')
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_json(file_path=file_path)

    def book_reviews(self) -> pd.DataFrame:
        folder = 'DutchBookReviewsDataset/'
        file_neg = 'dutch_book_review_neg.csv'
        file_pos = 'dutch_book_review_pos.csv'
        neg = self.get_csv(
            file_path=resource_filename(__name__, self.source_path + self.sub_folder + folder + file_neg),
            file_config={'header': None, 'sep': ';', 'names': ['text'], 'skip_blank_lines': True})
        neg['score'] = 0
        neg = neg[['score', 'text']]
        pos = self.get_csv(
            file_path=resource_filename(__name__, self.source_path + self.sub_folder + folder + file_pos),
            file_config={'header': None, 'sep': ';', 'names': ['text'], 'skip_blank_lines': True})
        pos['score'] = 2
        pos = pos[['score', 'text']]
        return neg.append(pos, ignore_index=True, sort=False)


class SourceLocalSink(DataSource):
    sub_folder = "../sink/"
    colnames = ['score', 'text', 'language', 'source']
    chunksize = 100000
    sep = ';'
    header = 0

    def load(self, file):
        return self.stream_large_csv(file_path=self.sub_folder + file,
                                     file_config={
                                         'header': self.header,
                                         'chunksize': self.chunksize,
                                         'sep': self.sep,
                                         'names': self.colnames
                                     })

    def sink_english(self) -> pd.DataFrame:
        return self.load(file='english_sink.csv')

    def sink_arabic(self) -> pd.DataFrame:
        return self.load(file='arabic_sink.csv')

    def sink_german(self) -> pd.DataFrame:
        return self.load(file='german_sink.csv')

    def sink_german_train(self) -> pd.DataFrame:
        return self.load(file='german_sink_train.csv')

    def sink_german_test(self) -> pd.DataFrame:
        return self.load(file='german_sink_test.csv')

    def sink_german_validation(self) -> pd.DataFrame:
        return self.load(file='german_sink_validation.csv')

    def sink_polish(self) -> pd.DataFrame:
        return self.load(file='polish_sink.csv')

    def sink_chinese(self) -> pd.DataFrame:
        return self.load(file='chinese_sink.csv')

    def sink_french(self) -> pd.DataFrame:
        return self.load(file='french_sink.csv')

    def sink_dutch(self) -> pd.DataFrame:
        return self.load(file='dutch_sink.csv')

    def sink_multi_lang(self) -> pd.DataFrame:
        return self.load(file='multi_lang_sink.csv')

    def sink_multi_lang_noger(self) -> pd.DataFrame:
        return self.load(file='multi_lang_noger_sink.csv')

    def clear_sink(self):
        substrings = ['.csv']
        files = self.get_all_file_paths(self.sub_folder)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                os.remove(file_path)
                self.logger.info(f'Removed: {file_path}')
