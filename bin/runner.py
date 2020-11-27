#!/usr/bin/env python
import yaml
from pkg_resources import resource_filename

from bachelorarbeit.collection.datasource import SourceLocalDataEnglish, SourceLocalDataArabic, SourceLocalDataGerman, SourceLocalDataPolish, SourceLocalDataChinese, SourceLocalCache
from bachelorarbeit.collection.sink import SinkCache
from bachelorarbeit.collection.task import ComputeEnglishSemEval, ComputeEnglishAmazonMovieReview, ComputeEnglishKaggleSentiment, ComputeEnglishWebisTripad, ComputeEnglishSentoken, ComputeArabicSemEval, ComputeGermanScare, ComputePolishPolEmo, \
    FitOutputsToTorch, ClearCache, ManageCache, SplitTrainTestGerman, ShuffleLanguages, ComputeGermanPolarityClues, ComputeChineseDouBan


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
# sourceLocalDataSpanish = SourceLocalDataSpanish(config)
sourceLocalDataChinese = SourceLocalDataChinese(config)
sourceLocalCache = SourceLocalCache(config)

# your pipeline of tasks:
tasks = [
    ClearCache(sourceLocalCache),
    ComputeEnglishSemEval(sourceLocalDataEnglish, sinkCache),
    # ComputeEnglishAmazonMovieReview(sourceLocalDataEnglish, sinkCache),
    ComputeEnglishKaggleSentiment(sourceLocalDataEnglish, sinkCache),
    ComputeEnglishWebisTripad(sourceLocalDataEnglish, sinkCache),
    ComputeEnglishSentoken(sourceLocalDataEnglish, sinkCache),
    ComputeArabicSemEval(sourceLocalDataArabic, sinkCache),
    ComputeGermanScare(sourceLocalDataGerman, sinkCache),
    ComputeGermanPolarityClues(sourceLocalDataGerman, sinkCache),
    ComputePolishPolEmo(sourceLocalDataPolish, sinkCache),
    ComputeChineseDouBan(sourceLocalDataChinese, sinkCache),
    # ComputeSpanishUnknown(sourceLocalDataSpanish, sinkCache),
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


if __name__ == "__main__":
    main()
