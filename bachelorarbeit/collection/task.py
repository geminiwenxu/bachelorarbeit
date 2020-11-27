import pandas as pd
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
        self.sinkCache = sinkA
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
            if isinstance(score, str):
                if score in ['negative', '-1', '-2']:
                    score = -1
                elif score in ['positive', '1', '2']:
                    score = 1
                else:
                    score = 0
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
        self.sinkCache.insert(self.target_name, self.data)


class ComputeEnglishAmazonMovieReview(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkCache = sinkA
        self.source = 'AmazonMovieReview'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str) or isinstance(score, float):
                if score in ['1', '2', 1.0, 2.0]:
                    score = -1
                elif score in ['4', '5', 4.0, 5.0]:
                    score = 1
                else:
                    score = 0
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
        self.sinkCache.insert(self.target_name, self.data)


class ComputeEnglishKaggleSentiment(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkCache = sinkA
        self.source = 'KaggleSentiment'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str) or isinstance(score, int):
                if score in ['0', 0]:
                    score = -1
                else:
                    pass
            return score
        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.kaggle_sentiment = self.sourceLocalDataEnglish.kaggle_sentiment()

    def transform(self):
        for file, data in self.kaggle_sentiment:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            columns_titles = ['score', 'text', 'language', 'source']
            self.data = self.data.reindex(columns=columns_titles)
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ComputeEnglishWebisTripad(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkCache = sinkA
        self.source = 'WebisTripad'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str) or isinstance(score, int):
                if score in ['1', '2', 1, 2]:
                    score = -1
                elif score in ['4', '5', 4, 5]:
                    score = 1
                else:
                    score = 0
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
        self.sinkCache.insert(self.target_name, self.data)


class ComputeEnglishSentoken(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataEnglish = sourceA
        self.sinkCache = sinkA
        self.source = 'Sentoken'
        self.language = 'english'
        self.target_name = 'english_sink'
        super().__init__()

    def extract(self):
        self.sentoken = self.sourceLocalDataEnglish.sentoken()

    def transform(self):
        for score, data in self.sentoken:
            self.data = {}
            self.data['score'] = score
            self.data['text'] = ' '.join(data).replace('\n', '')
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.data = pd.DataFrame([self.data])
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ComputeArabicSemEval(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataArabic = sourceA
        self.sinkCache = sinkA
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

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str):
                if score in ['negative', '-1']:
                    score = -1
                elif score in ['positive', '1']:
                    score = 1
                else:
                    score = 0
            return score
        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.semeval_arabic = self.sourceLocalDataArabic.semeval_arabic()

    def transform(self):
        for file, data in self.semeval_arabic:
            self.data = data
            self.filter_columns(file)
            self.data.columns = ['score', 'text']
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ComputeGermanScare(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkCache = sinkA
        self.source = 'SCARE'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str) or isinstance(score, int):
                if score in ['1', '2', 1, 2]:
                    score = -1
                elif score in ['4', '5', 4, 5]:
                    score = 1
                else:
                    score = 0
            return score
        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.scare_german = self.sourceLocalDataGerman.scare_german()

    def transform(self):
        for file, data in self.scare_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ComputeGermanPolarityClues(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataGerman = sourceA
        self.sinkCache = sinkA
        self.source = 'PolarityClues'
        self.language = 'german'
        self.target_name = 'german_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str) or isinstance(score, int):
                if score in ['negative']:
                    score = -1
                elif score in ['positive']:
                    score = 1
                else:
                    score = 0
            return score
        self.data['score'] = self.data['score'].apply(lambda x: change_score(x))

    def extract(self):
        self.scare_german = self.sourceLocalDataGerman.scare_german()

    def transform(self):
        for file, data in self.scare_german:
            self.data = data
            self.data['language'] = self.language
            self.data['source'] = self.source
            self.normalise_score()
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ComputePolishPolEmo(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataPolish = sourceA
        self.sinkCache = sinkA
        self.source = 'PolEmo'
        self.language = 'polish'
        self.target_name = 'polish_sink'
        super().__init__()

    def normalise_score(self, score):
        if 'minus' in score:
            score = -1
        elif 'plus' in score:
            score = 1
        else:
            score = 0
        return score

    def extract(self):
        self.polemo_polish = self.sourceLocalDataPolish.polemo_polish()

    def transform(self):
        for _, data in self.polemo_polish:
            self.data = []
            for line in data:
                d = line.split('__label__')
                self.data.append({'score': self.normalise_score(d[1]), 'text': d[0], 'language': self.language, 'source': self.source})
            self.data = pd.DataFrame(self.data)
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ComputeSpanishUnknown(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataSpanish = sourceA
        self.sinkCache = sinkA
        self.source = 'unknown'
        self.language = 'spanish'
        self.target_name = 'spanish_sink'
        super().__init__()

    def normalise_score(self, score):
        if isinstance(score, str) or isinstance(score, int):
            if score in ['-1', '-2', -1, -2]:
                score = -1
            elif score in ['1', '2', 1, 2]:
                score = 1
            else:
                score = 0
        return score

    def extract(self):
        self.unknown_spanish = self.sourceLocalDataSpanish.unknown_spanish()

    def transform(self):
        for file, data in self.unknown_spanish:
            self.data = []
            for review in data['paper']:
                for review_item in review:
                    self.data.append({'score':  self.normalise_score(review_item['evaluation']), 'text': review_item['text'], 'language': self.language,
                                      'source': self.source})
            self.data = pd.DataFrame(self.data)
            self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class FitOutputsToTorch(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalCache = sourceA
        self.sinkCache = sinkA
        super().__init__()

    def extract(self):
        self.cache_english = self.sourceLocalCache.cache_english()
        self.cache_arabic = self.sourceLocalCache.cache_arabic()
        self.cache_german = self.sourceLocalCache.cache_german()
        self.cache_chinese = self.sourceLocalCache.cache_chinese()
        self.cache_polish = self.sourceLocalCache.cache_polish()

    def adjust_score(self, score):
        if isinstance(score, str) or isinstance(score, int):
            if score in ['-1', -1]:
                score = 0
            elif score in ['0', 0]:
                score = 1
            else:
                score = 2
        return score

    def transform(self):
        for self.target_name, language in [('english_cl', self.cache_english), ('arabic_cl', self.cache_arabic), ('german_cl', self.cache_german), ('polish_cl', self.cache_polish), ('chinese_cl', self.cache_chinese)]:
            self.data = []
            for line in language:
                line['score'] = line['score'].apply(self.adjust_score)
                self.data = line
                self.store()
                self.data = []

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)


class ClearCache(Task):
    def __init__(self, sourceA):
        self.sourceLocalCache = sourceA
        super().__init__()

    def extract(self):
        self.sourceLocalCache.clear_cache()


class ManageCache(Task):
    def __init__(self, sourceA):
        self.sourceLocalCache = sourceA
        super().__init__()

    def extract(self):
        self.sourceLocalCache.manage_cache()


class SplitTrainTestGerman(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalCache = sourceA
        self.sinkCache = sinkA
        self.train_name = 'german_sink_train'
        self.test_name = 'german_sink_test'
        super().__init__()

    def extract(self):
        self.cache_german = self.sourceLocalCache.cache_german()

    def transform(self):
        self.data = pd.DataFrame()
        for line in self.cache_german:
            self.data = self.data.append(line, ignore_index=True)
        self.train, self.test = train_test_split(self.data, test_size=0.4)
        self.store()

    def store(self):
        self.sinkCache.insert(self.train_name, self.train)
        self.sinkCache.insert(self.test_name, self.test)


class ShuffleLanguages(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalCache = sourceA
        self.sinkCache = sinkA
        self.target_name = 'multi_lang_sink'
        self.target_name_ng = 'multi_lang_noger_sink'
        super().__init__()

    def extract(self):
        self.cache_english = self.sourceLocalCache.cache_english()
        self.cache_arabic = self.sourceLocalCache.cache_arabic()
        self.cache_german_train = self.sourceLocalCache.cache_german_train()
        self.cache_polish = self.sourceLocalCache.cache_polish()
        self.cache_chinese = self.sourceLocalCache.cache_chinese()

    def transform(self):
        self.data = pd.DataFrame()
        self.data_ng = pd.DataFrame()
        for lang, data in [('english', self.cache_english), ('arabic', self.cache_arabic), ('german', self.cache_german_train), ('polish', self.cache_polish), ('chinese', self.cache_chinese)]:
            for line in data:
                self.data = self.data.append(line, ignore_index=True)
        self.data_ng = self.data[self.data['language'] != 'german'].copy()
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data_ng = self.data_ng.sample(frac=1).reset_index(drop=True)
        self.store()

    def store(self):
        self.sinkCache.insert(self.target_name, self.data)
        self.sinkCache.insert(self.target_name_ng, self.data_ng)


class ComputeChineseDouBan(Task):
    def __init__(self, sourceA, sinkA):
        self.sourceLocalDataChinese = sourceA
        self.sinkCache = sinkA
        self.source = 'DouBan'
        self.language = 'chinese'
        self.target_name = 'chinese_sink'
        super().__init__()

    def normalise_score(self):
        def change_score(score):
            if isinstance(score, str) or isinstance(score, int):
                if score in ['1', '2', 1, 2]:
                    score = -1
                elif score in ['4', '5', 4, 5]:
                    score = 1
                else:
                    score = 0
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
        self.sinkCache.insert(self.target_name, self.data)
