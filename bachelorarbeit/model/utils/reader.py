import pandas as pd
from pkg_resources import resource_filename


def read_cache(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(resource_filename(__name__, file_path), sep=';', names=['score', 'text', 'language', 'source'])
    if len(df) < 2:
        raise IOError
    else:
        df = df[df['text'].str.len() > 2]
    return df
