import pandas as pd
from pkg_resources import resource_filename


class Sink:
    def __init__(self, config, logger) -> None:
        self.pg_instructions = {'if_exists': 'replace', 'index': False}
        self.file_instructions = config['files']
        self.logger = logger

    def insert_to_disk(self, file_path: str, file_name: str, data: pd.DataFrame) -> None:
        data.to_csv(resource_filename(__name__, file_path) + file_name + ".csv", **self.file_instructions['csv'])


class SinkCache(Sink):

    def insert(self, file_name, data):
        self.insert_to_disk(file_path=self.file_instructions['path'], file_name=file_name, data=data)
