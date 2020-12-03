from bachelorarbeit.model.utils.analysis import *
from bachelorarbeit.model.utils.reader import *

if __name__ == '__main__':
    class_names = ['negative', 'neutral', 'positive']
    df = read_cache(file_path='../../../cache/german_sink_test.csv')
    explore_dataframe(df)
    # explore_data(df, strategy='ger-only')
    print(df.score.value_counts())