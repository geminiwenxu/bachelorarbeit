import re

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


class Task:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self):
        pass

    def transform(self):
        pass

    def store(self):
        pass


class ComputeEnglishSemEval(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkLocal = sinkA
        self.source = 'SemEvalEnglish'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def filter_columns(self, file):
        if any(s in file for s in ['BD', 'CE']):
            if len(self.data.columns.tolist()) > 4:
                cols = [0, 1, 4]
            else:
                cols = [0, 1]
        else:
            if len(self.data.columns.tolist()) > 3:
                cols = [0, 3]
            else:
                cols = [0]
        self.data.drop(self.data.columns[cols], axis=1, inplace=True)

    def normalise_score(self):
        def change_score(score):
            if score in ['negative', '-1', '-2', -1, -2]:
                score = 0
            elif score in ['positive', '1', '2', 1, 2]:
                score = 2
            else:
                score = 1
            return score

        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.semeval_english = self.sourceLocalDataEnglish.semeval_english()

    def transform(self):
        for file, data in self.semeval_english:
            self.data = data
            self.filter_columns(file)
            self.data.columns = ['score', 'text']
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeEnglishAmazonMovieReview(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkLocal = sinkA
        self.source = 'AmazonMovieReview'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if float(score) in [1.0, 2.0]:
                score = 0
            elif float(score) in [4.0, 5.0]:
                score = 2
            else:
                score = 1
            return score

        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.amazon_movie_review = self.sourceLocalDataEnglish.amazon_movie_review()

    def transform(self):
        dl = {}
        self.data = []
        counter = 0
        for line in self.amazon_movie_review:
            if line == '\n':
                dl['language'] = self.language
                dl['source'] = self.source
                self.data.append(dl)
                dl = {}
                if counter % 10000 == 0 and counter != 0:
                    self.data = pd.DataFrame(self.data)
                    self.normalise_score()
                    self.store()
                    self.data = []
                counter += 1
            else:
                if line.split(' ')[0] == 'review/score:':
                    dl = {'score': ''.join(line.split(' ')[1:]).replace('\n', '')}
                elif line.split(' ')[0] == 'review/text:':
                    dl.update({'text': ' '.join(line.split(' ')[1:]).replace('\n', '')})
                else:
                    continue
        self.data = pd.DataFrame(self.data)
        self.normalise_score()
        self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeEnglishWebisTripad(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkLocal = sinkA
        self.source = 'WebisTripad'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if score in ['1', '2', 1, 2]:
                score = 0
            elif score in ['4', '5', 4, 5]:
                score = 2
            else:
                score = 1
            return score

        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.webis_tripad = self.sourceLocalDataEnglish.webis_tripad()

    def transform(self):
        for file, data in self.webis_tripad:
            self.data = {}
            self.data['score'] = data['attributes']["de.webis.attribute.Rating"]["score"]
            self.data['text'] = data['text']
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data = pd.DataFrame([self.data])
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeArabicSemEval(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataArabic = sourceA
        self.sinkLocal = sinkA
        self.source = 'SemEvalArabic'
        self.language = 'arabic'
        self.target_name = 'arabic_sink'
        super().__init__()

    def filter_columns(self, file):
        if any(s in file for s in ['BD', 'CE']):
            cols = [0, 1]
        else:
            cols = [0]
        self.data.drop(self.data.columns[cols], axis=1, inplace=True)

    @staticmethod
    def change_score(score):
        if score in ['negative', '-1', -1, -2]:
            score = 0
        elif score in ['positive', '1', 1, 2]:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.semeval_arabic = self.sourceLocalDataArabic.semeval_arabic()

    def transform(self):
        for file, data in self.semeval_arabic:
            self.data = data
            self.filter_columns(file)
            self.data.columns = ['score', 'text']
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanScare(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'SCARE'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['1', '2', 1, 2]:
            score = 0
        elif score in ['4', '5', 4, 5]:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.scare_german = self.sourceLocalDataGerman.scare_german()

    def transform(self):
        for file, data in self.scare_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanPotts(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'PotTS'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['negative', -1]:
            score = 0
        elif score in ['positive', 1]:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.potts_german = self.sourceLocalDataGerman.potts_german()

    def transform(self):
        for file, data in self.potts_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanSB(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'SB10K'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['negative', -1]:
            score = 0
        elif score in ['positive', 1]:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.sb_german = self.sourceLocalDataGerman.sb_german()

    def transform(self):
        for file, data in self.sb_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanEval(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'GermEval2017'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['negative']:
            score = 0
        elif score in ['positive']:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.germeval = self.sourceLocalDataGerman.germeval()

    def transform(self):
        for file, data in self.germeval:
            self.data = data
            self.data = self.data[['score', 'text']]
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            columnsTitles = ['score', 'text', 'language', 'source']
            self.data = self.data.reindex(columns=columnsTitles)
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanFilmStarts(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'Filmstarts'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in [1.0, 2.0, 1, 2]:
            score = 0
        elif score in [4.0, 5.0, 4, 5]:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.filmstarts_german = self.sourceLocalDataGerman.filmstarts_german()

    def transform(self):
        for file, data in self.filmstarts_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanHolidaycheck(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'Holidaycheck'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['1', '2', 1, 2]:
            score = 0
        elif score in ['5', 5, '4', 4, 6, '6']:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.holidaycheck_german = self.sourceLocalDataGerman.holidaycheck_german()

    def transform(self):
        for data in self.holidaycheck_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeGermanLeipzig(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkLocal = sinkA
        self.source = 'Leipzig'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['__label__neutral']:
            score = 1
        else:
            raise NotImplementedError
        return score

    def extract(self):
        self.leipzig_german = self.sourceLocalDataGerman.leipzig_german()

    def transform(self):
        dl = {}
        self.data = []
        counter = 0
        for file in self.leipzig_german:
            for line in file:
                dl = {}
                if line.split(' ')[0] == '__label__neutral':
                    dl = {'score': ''.join(line.split(' ')[0]).replace('\n', '')}
                    dl.update({'text': ' '.join(line.split(' ')[1:]).replace('\n', '')})
                else:
                    continue
                dl['language'] = self.language
                dl['source'] = self.source
                self.data.append(dl)
                if counter % 10000 == 0 and counter != 0:
                    self.data = pd.DataFrame(self.data)
                    self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
                    self.store()
                    self.data = []
                counter += 1
        self.data = pd.DataFrame(self.data)
        self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
        self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputePolishPolEmo(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataPolish = sourceA
        self.sinkLocal = sinkA
        self.source = 'PolEmo'
        self.language = 'polish'
        self.target_name = 'polish_sink'
        super().__init__()

    def normalise_score(self, score):
        if 'minus' in score:
            score = 0
        elif 'plus' in score:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.polemo_polish = self.sourceLocalDataPolish.polemo_polish()

    def transform(self):
        for _, data in self.polemo_polish:
            self.data = []
            for line in data:
                d = line.split('__label__')
                self.data.append({'score': self.normalise_score(d[1]), 'text': d[0], 'language': self.language,
                                  'source': self.source})
            self.data = pd.DataFrame(self.data)
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ClearLocalSink(Task):
    def __init__(self, sourceA):
        self.sourceLocalSink = sourceA
        super().__init__()

    def extract(self):
        self.sourceLocalSink.clear_sink()


class ComputeChineseDouBan(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataChinese = sourceA
        self.sinkLocal = sinkA
        self.source = 'DouBan'
        self.language = 'chinese'
        self.target_name = 'chinese_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if score in ['1', '2', 1, 2]:
                score = 0
            elif score in ['4', '5', 4, 5]:
                score = 2
            else:
                score = 1
            return score

        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.douban = self.sourceLocalDataChinese.douban_movies()

    def transform(self):
        for data in self.douban:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeFrenchKaggle(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataFrench = sourceA
        self.sinkLocal = sinkA
        self.source = 'Kaggle'
        self.language = 'french'
        self.target_name = 'french_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score in ['1', 1]:
            score = 2
        else:
            score = 0
        return score

    def extract(self):
        self.kaggle = self.sourceLocalDataFrench.kaggle()

    def transform(self):
        for data in self.kaggle:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeFrenchBetsentimentTeams(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataFrench = sourceA
        self.sinkLocal = sinkA
        self.source = 'Betsentiment_teams'
        self.language = 'french'
        self.target_name = 'french_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score == 'POSITIVE':
            score = 2
        elif score == 'NEGATIVE':
            score = 0
        else:
            score = 1
        return score

    def extract(self):
        self.betsentiment_teams = self.sourceLocalDataFrench.betsentiment_teams()

    def transform(self):
        for data in self.betsentiment_teams:
            self.data = data
            self.data = self.data[['score', 'text']]
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeFrenchBetsentimentWorldCup(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataFrench = sourceA
        self.sinkLocal = sinkA
        self.source = 'Betsentiment_worldcup'
        self.language = 'french'
        self.target_name = 'french_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score == 'POSITIVE':
            score = 2
        elif score == 'NEGATIVE':
            score = 0
        else:
            score = 1
        return score

    def extract(self):
        self.betsentiment_worldcup = self.sourceLocalDataFrench.betsentiment_worldcup()

    def transform(self):
        for data in self.betsentiment_worldcup:
            self.data = data
            self.data = self.data[['score', 'text']]
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeDutchSocialMedia(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataDutch = sourceA
        self.sinkLocal = sinkA
        self.source = 'SocialMediaKaggle'
        self.language = 'dutch'
        self.target_name = 'dutch_sink'
        super().__init__()

    @staticmethod
    def change_score(score):
        if score < -0.5:
            score = 0
        elif score > 0.5:
            score = 2
        else:
            score = 1
        return score

    def extract(self):
        self.social_media_collection = self.sourceLocalDataDutch.social_media_collection()

    @staticmethod
    def clean_text(text):
        if text:
            text = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", "", text)
            text = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", "", text)
            text = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", "", text)
            # text = re.sub(r"(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", "", text)
            text = re.sub(r"RT : ", "", text)
        return text

    def transform(self):
        for file, data in self.social_media_collection:
            self.data = []
            for ele in data:
                element = dict()
                element['score'] = ele['sentiment_pattern']
                element['text'] = self.clean_text(ele['full_text'])
                element['language'] = self.language
                element['source'] = self.source
                self.data.append(element)
            self.data = pd.DataFrame(self.data)
            self.data['score'] = self.data['score'].apply(lambda x: self.change_score(x))
            self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)


class ComputeDutchBookReviews(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataDutch = sourceA
        self.sinkLocal = sinkA
        self.source = 'BookReviews'
        self.language = 'dutch'
        self.target_name = 'dutch_sink'
        super().__init__()

    def extract(self):
        self.book_reviews = self.sourceLocalDataDutch.book_reviews()

    def transform(self):
        self.book_reviews['language'] = self.language
        self.book_reviews['source'] = self.source
        self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.book_reviews)


class SplitTrainTestGerman(Task):
    def __init__(self, sourceA, sinkA, random_seed: int, train_set_size: int, test_set_size: int):
        self.sourceLocalSink = sourceA
        self.sinkLocal = sinkA
        self.train_name = 'german_sink_train'
        self.test_name = 'german_sink_test'
        self.validation_name = 'german_sink_validation'
        self.random_seed = random_seed
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        super().__init__()

    def extract(self):
        self.sink_german = self.sourceLocalSink.sink_german()

    def transform(self):
        self.data = pd.DataFrame()
        for line in self.sink_german:
            self.data = self.data.append(line, ignore_index=True)
        self.train, self.test = train_test_split(
            self.data,
            random_state=self.random_seed,
            test_size=(1 - self.train_set_size),
            stratify=self.data.score
        )
        self.validation, self.test = train_test_split(
            self.test,
            random_state=self.random_seed,
            test_size=self.test_set_size,
            stratify=self.test.score
        )
        self.store()

    def store(self):
        self.sinkLocal.insert(self.train_name, self.train)
        self.sinkLocal.insert(self.test_name, self.test)
        self.sinkLocal.insert(self.validation_name, self.validation)


class ShuffleLanguages(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalSink = sourceA
        self.sinkLocal = sinkA
        self.target_name = 'multi_lang_sink'
        self.target_name_ng = 'multi_lang_noger_sink'
        self.frac = 1
        super().__init__()

    def extract(self):
        self.sink_english = self.sourceLocalSink.sink_english()
        self.sink_arabic = self.sourceLocalSink.sink_arabic()
        self.sink_german_train = self.sourceLocalSink.sink_german_train()
        self.sink_polish = self.sourceLocalSink.sink_polish()
        self.sink_chinese = self.sourceLocalSink.sink_chinese()
        self.sink_french = self.sourceLocalSink.sink_french()
        self.sink_dutch = self.sourceLocalSink.sink_dutch()

    def transform(self):
        self.data = pd.DataFrame()
        self.data_ng = pd.DataFrame()
        for lang, data in [('english', self.sink_english), ('arabic', self.sink_arabic),
                           ('german', self.sink_german_train), ('polish', self.sink_polish),
                           ('chinese', self.sink_chinese), ('french', self.sink_french), ('dutch', self.sink_dutch)]:
            for line in data:
                self.data = self.data.append(line, ignore_index=True)
        self.data_ng = self.data[self.data['language'] != 'german'].copy()
        self.data = self.data.sample(frac=self.frac).reset_index(drop=True)
        self.data_ng = self.data_ng.sample(frac=self.frac).reset_index(drop=True)
        self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)
        self.sinkLocal.insert(self.target_name_ng, self.data_ng)


class ComputeDownSampleLanguages(Task):
    def __init__(self, sourceA, sinkA, random_seed):
        self.sourceLocalSink = sourceA
        self.sinkLocal = sinkA
        self.random_seed = random_seed
        self.frac = 1
        super().__init__()

    def extract(self):
        self.sink_multi_lang = self.sourceLocalSink.sink_multi_lang()
        self.sink_multi_lang_noger = self.sourceLocalSink.sink_multi_lang_noger()
        self.sink_german_train = self.sourceLocalSink.sink_german_train()
        self.sink_german_test = self.sourceLocalSink.sink_german_test()
        self.sink_german_validation = self.sourceLocalSink.sink_german_validation()

    def transform(self):
        for model, cache in [('german_sink_test_balanced', self.sink_german_test),
                             ('german_sink_train_balanced', self.sink_german_train),
                             ('german_sink_validation_balanced', self.sink_german_validation),
                             ('multi_lang_sink_balanced', self.sink_multi_lang),
                             ('multi_lang_noger_sink_balanced', self.sink_multi_lang_noger)]:
            self.data = pd.DataFrame()
            self.model = model
            for chunk in cache:
                class_sizes = chunk.iloc[:, 0].value_counts()
                min_class_size = min(class_sizes)
                samples = []
                for class_index in class_sizes.index.values:
                    class_sample = chunk[chunk.iloc[:, 0] == class_index]
                    if 0 < min_class_size < class_sample.shape[0]:
                        class_sample = sklearn.utils.resample(class_sample, replace=False, n_samples=min_class_size,
                                                              random_state=self.random_seed)
                    samples.append(class_sample)
                samples = pd.concat(samples)
                samples = samples.sample(frac=self.frac).reset_index(drop=True)
                self.data = self.data.append(samples, ignore_index=True)
            self.store()

    def store(self):
        self.sinkLocal.insert(self.model, self.data)


class FilterLanguagesFromTraining(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalSink = sourceA
        self.sinkLocal = sinkA
        self.frac = 1
        super().__init__()

    def extract(self):
        self.sink_multi_lang = self.sourceLocalSink.sink_multi_lang()

    def transform(self):
        for chunk in self.sink_multi_lang:
            for lang in ['arabic', 'chinese', 'polish', 'english', 'french', 'dutch']:
                self.target_name = f'multi_lang_no{lang}_sink'
                self.data = chunk[~chunk['language'].isin([lang])].copy()
                self.data = self.data.sample(frac=self.frac).reset_index(drop=True)
                self.store()

    def store(self):
        self.sinkLocal.insert(self.target_name, self.data)
