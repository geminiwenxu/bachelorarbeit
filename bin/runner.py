#!/usr/bin/env python
import yaml
from pkg_resources import resource_filename

from bachelorarbeit.collection.datasource import SourceLocalDataEnglish, SourceLocalDataArabic, SourceLocalDataGerman, SourceLocalDataPolish, SourceLocalDataChinese, SourceLocalCache
from bachelorarbeit.collection.sink import SinkCache
from bachelorarbeit.collection.task import ComputeEnglishSemEval, ComputeEnglishAmazonMovieReview, ComputeEnglishKaggleSentiment, ComputeEnglishWebisTripad, ComputeEnglishSentoken, ComputeArabicSemEval, ComputeGermanScare, ComputePolishPolEmo, \
    FitOutputsToTorch, ClearCache, ManageCache, SplitTrainTestGerman, ShuffleLanguages, ComputeGermanPolarityClues, ComputeChineseDouBan, ComputeGermanPotts, ComputeGermanEval, ComputeGermanFilmStarts, ComputeGermanHolidaycheck,ComputeGermanLeipzig, ComputeGermanSB


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/collection_config.yaml')

sinkCache = SinkCache(config)
sourceLocalDataEnglish = SourceLocalDataEnglish(config)
sourceLocalDataArabic = SourceLocalDataArabic(config)
sourceLocalDataGerman = SourceLocalDataGerman(config)
sourceLocalDataPolish = SourceLocalDataPolish(config)
sourceLocalDataChinese = SourceLocalDataChinese(config)
sourceLocalCache = SourceLocalCache(config)

# your pipeline of tasks:
tasks = [
    ClearCache(sourceLocalCache),
    ComputeEnglishSemEval(sourceLocalDataEnglish, sinkCache),
    ComputeEnglishAmazonMovieReview(sourceLocalDataEnglish, sinkCache),
    ComputeEnglishWebisTripad(sourceLocalDataEnglish, sinkCache),
    ComputeArabicSemEval(sourceLocalDataArabic, sinkCache),
    ComputeGermanScare(sourceLocalDataGerman, sinkCache),
    ComputeGermanPotts(sourceLocalDataGerman, sinkCache),
    ComputeGermanSB(sourceLocalDataGerman, sinkCache),
    ComputeGermanHolidaycheck(sourceLocalDataGerman, sinkCache),
    ComputeGermanLeipzig(sourceLocalDataGerman, sinkCache),
    ComputeGermanEval(sourceLocalDataGerman, sinkCache),
    ComputeGermanFilmStarts(sourceLocalDataGerman, sinkCache),
    ComputeArabicSemEval(sourceLocalDataGerman, sinkCache),
    ComputePolishPolEmo(sourceLocalDataPolish, sinkCache),
    ComputeChineseDouBan(sourceLocalDataChinese, sinkCache),
    FitOutputsToTorch(sourceLocalCache, sinkCache),
    ManageCache(sourceLocalCache),
    SplitTrainTestGerman(sourceLocalCache, sinkCache),
    ShuffleLanguages(sourceLocalCache, sinkCache)
]


def main():
    for task in tasks:
        print(task)
        try:
            task.extract()
            task.transform()
        except Exception as e:
            print("Error: ", e)
    print('Task runner complete.')


if __name__ == "__main__":
    main()
