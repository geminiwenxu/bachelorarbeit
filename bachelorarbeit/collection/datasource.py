import ast
import json
import os

import pandas as pd
from pkg_resources import resource_filename


class DataSource:
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
        return json.loads(file_path)

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
        colnames = ['score', 'text']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': colnames, 'header': None, 'usecols': [1, 3]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def potts_german(self) -> pd.DataFrame:
        source_path = 'PotTS/preprocessed-no-noise-cleaner/'
        substrings = ['.tsv']
        colnames = ['score', 'text']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': colnames, 'header': None, 'usecols': [1, 2]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def sb_german(self) -> pd.DataFrame:
        source_path = 'SB10K/preprocessed/'
        substrings = ['.tsv']
        colnames = ['score', 'text']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': colnames, 'header': None, 'usecols': [1, 2]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def germeval(self) -> pd.DataFrame:
        source_path = 'germeval2017/'
        substrings = ['.tsv']
        colnames = ['text', 'score']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': colnames, 'header': 0, 'usecols': [3, 1]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def filmstarts_german(self) -> pd.DataFrame:
        source_path = 'filmstarts/'
        substrings = ['.tsv']
        colnames = ['score', 'text']
        file_config = {'sep': '\t', 'parse_dates': True, 'names': colnames, 'header': None, 'usecols': [1, 2]}
        files = self.get_all_file_paths(self.sub_folder + source_path)
        for file_name, file_path in files:
            if all([substring in file_name for substring in substrings]):
                yield file_name, self.get_csv(file_path=file_path, file_config=file_config)

    def holidaycheck_german(self) -> pd.DataFrame:
        source_path = 'holidaycheck/'
        colnames = ['score', 'text']
        file_path = self.sub_folder + source_path + 'holidaycheck.clean.filtered.tsv'
        return self.stream_large_csv(file_path=file_path,
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': '\t', 'usecols': [0, 1],
                                                  'names': colnames})

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
        colnames = ['score', 'text']
        file_path = self.sub_folder + 'douban/ratings.csv'
        return self.stream_large_csv(file_path=file_path,
                                     file_config={'header': 0, 'chunksize': 100000, 'sep': ',', 'usecols': [2, 4],
                                                  'names': colnames})


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
