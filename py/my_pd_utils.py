import pandas as pd
from collections.abc import Iterable

def strip_all_strings(df):
    df_obj = df.select_dtypes(include=['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())


def concat_dfs(df_list):
    df = pd.DataFrame()
    for df_curr in df_list:
        df = pd.concat([df, df_curr])
    return df

def exclude_fields(df, exclude_fields):
    fields_to_stay = [col for col in df.columns if col not in exclude_fields]
    return df[fields_to_stay] 

def make_list_if_single(kinds_ext):
    if isinstance(kinds_ext, Iterable) and not isinstance(kinds_ext, str):
        return kinds_ext
    else:
        return [kinds_ext]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False