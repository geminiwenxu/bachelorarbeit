import pandas as pd
from pkg_resources import resource_filename


class Sink:
    def __init__(self, config, logger) -> None:
        self.pg_instructions = {'if_exists': 'replace', 'index': False}
        self.sink_config = config['sink']
        self.logger = logger

    def insert_to_disk(self, file_path: str, file_name: str, data: pd.DataFrame) -> None:
        data.to_csv(resource_filename(__name__, file_path) + file_name + self.sink_config['suffix'], **self.sink_config['csv'])


class LocalSink(Sink):

    def insert(self, file_name, data):
        self.insert_to_disk(file_path=self.sink_config['path'], file_name=file_name, data=data)
