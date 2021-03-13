import glob

import pandas as pd


class Misclassification:

    def __init__(self, path, out_name: str='misclassification_error_report.csv'):
        self.path = path
        self.file_name = ''
        self.frame = pd.DataFrame()
        self.out_name = out_name

    @staticmethod
    def __load(file_name) -> pd.DataFrame:
        df = pd.read_csv(f'{file_name}',
                         header=0, sep=';',
                         error_bad_lines=False)
        return df

    def __get_files(self) -> pd.DataFrame:
        files = glob.glob(self.path + "*.csv")
        return files

    def extract(self) -> None:
        files = self.__get_files()
        for file in files:
            self.file_name = file
            df = self.__load(file)
            df = self.__select(df=df)
            self.__combine(df)
        self.__save()

    def __select(self, df) -> pd.DataFrame:
        df = df[df['actual_value'] != df['prediction']].copy()
        df['source'] = self.file_name.split('/')[-1]
        return df

    def __combine(self, df: pd.DataFrame) -> None:
        self.frame = self.frame.append(df, ignore_index=True, sort=False)

    def __save(self) -> None:
        path = self.path + self.out_name
        self.frame.to_csv(path, sep=';')


if __name__ == "__main__":
    path = '/Users/geminiwenxu/PycharmProjects/bachelorarbeit/text_report/'
    mist = Misclassification(path=path)
    mist.extract()
