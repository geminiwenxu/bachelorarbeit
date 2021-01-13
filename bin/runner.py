#!/usr/bin/env python
import yaml
import os
import logging
from datetime import datetime
from pkg_resources import resource_filename

from bachelorarbeit.collection.datasource import SourceLocalDataEnglish, SourceLocalDataArabic, SourceLocalDataGerman, \
    SourceLocalDataPolish, SourceLocalDataChinese, SourceLocalSink
from bachelorarbeit.collection.sink import LocalSink
from bachelorarbeit.collection.task import ComputeEnglishSemEval, ComputeEnglishAmazonMovieReview, \
    ComputeEnglishWebisTripad, ComputeArabicSemEval, ComputeGermanScare, ComputePolishPolEmo, \
    ClearLocalSink, SplitTrainTestGerman, ShuffleLanguages, ComputeChineseDouBan, ComputeGermanPotts, \
    ComputeGermanEval, ComputeGermanFilmStarts, ComputeGermanHolidaycheck, ComputeGermanLeipzig, ComputeGermanSB, \
    ComputeDownSampleLanguages


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


# Setup file logging
log_datetime = datetime.now().isoformat()
log_path = resource_filename(__name__, '../logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)
file_log_handler = logging.FileHandler(f"{log_path}/{log_datetime}_collection.log")
logger = logging.getLogger(__name__)
logger.addHandler(file_log_handler)
logger.setLevel('INFO')
stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)


# Setup project structure
sink_path = resource_filename(__name__, '../sink')
cache_path = resource_filename(__name__, '../cache')
data_path = resource_filename(__name__, '../data')
models_path = resource_filename(__name__, '../models')
if not os.path.exists(sink_path):
    os.mkdir(sink_path)
    logger.info("Created Path '/sink'")
if not os.path.exists(cache_path):
    os.mkdir(cache_path)
    logger.info("Created Path '/cache'")
if not os.path.exists(data_path):
    os.mkdir(data_path)
    logger.info("Created Path '/data'")
if not os.path.exists(models_path):
    os.mkdir(models_path)
    logger.info("Created Path '/models'")

config = get_config('/../config/collection_config.yaml')

localSink = LocalSink(config, logger)
sourceLocalDataEnglish = SourceLocalDataEnglish(config, logger)
sourceLocalDataArabic = SourceLocalDataArabic(config, logger)
sourceLocalDataGerman = SourceLocalDataGerman(config, logger)
sourceLocalDataPolish = SourceLocalDataPolish(config, logger)
sourceLocalDataChinese = SourceLocalDataChinese(config, logger)
sourceLocalSink = SourceLocalSink(config, logger)

# Pipeline of tasks:
tasks = [
    ClearLocalSink(sourceLocalSink),
    ComputeEnglishSemEval(sourceLocalDataEnglish, localSink),
    ComputeEnglishAmazonMovieReview(sourceLocalDataEnglish, localSink),
    ComputeEnglishWebisTripad(sourceLocalDataEnglish, localSink),
    ComputeArabicSemEval(sourceLocalDataArabic, localSink),
    ComputeGermanScare(sourceLocalDataGerman, localSink),
    ComputeGermanPotts(sourceLocalDataGerman, localSink),
    ComputeGermanSB(sourceLocalDataGerman, localSink),
    ComputeGermanHolidaycheck(sourceLocalDataGerman, localSink),
    ComputeGermanLeipzig(sourceLocalDataGerman, localSink),
    ComputeGermanEval(sourceLocalDataGerman, localSink),
    ComputeGermanFilmStarts(sourceLocalDataGerman, localSink),
    ComputePolishPolEmo(sourceLocalDataPolish, localSink),
    ComputeChineseDouBan(sourceLocalDataChinese, localSink),
    SplitTrainTestGerman(sourceLocalSink, localSink),
    ShuffleLanguages(sourceLocalSink, localSink),
    ComputeDownSampleLanguages(sourceLocalSink, localSink)
]


def main():
    for idx, task in enumerate(tasks):
        logger.info(f"Performing Collection Task '{idx+1} of {len(tasks)}': {task}")
        try:
            task.extract()
            task.transform()
        except Exception as e:
            logger.error(e)
    logger.info('Task runner complete.')


if __name__ == "__main__":
    main()
