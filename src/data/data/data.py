import streamlit as st
import pandas as pd

__all__ = [
    'get_cleaned_data',
    'process_data',
    'get_interquartile',
]

# * Data import
def split_martial(xdf):
    sex = list()
    martial = list()
    field = 'personal_status'
    c = xdf[field]

    for i in c:
        [s, m] = i.split(' ')
        sex.append(s)
        martial.append(m)
    
    xdf['sex'] = sex
    xdf['martial'] = martial

    # xdf.drop(field, axis = 1, inplace = True)

    return xdf.drop(field, axis = 1)

def classify_creditclass(x_df):
    map_credit = dict(good = 1, bad = 0)
    field = 'class'
    
    x_df[field] = x_df[field].map(map_credit)

    return x_df

def objects_to_categorical(x_df):
    cs = x_df.select_dtypes('object').columns

    for c in cs:
        x_df[c] = x_df[c].astype('category')
    
    return x_df

@st.cache_data
def get_cleaned_data(path: str = './credit_customers.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df = (
        df
            .pipe(split_martial)
            .pipe(classify_creditclass)
            .pipe(objects_to_categorical)
    )
    
    return df

def trim_df(x_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = x_df.select_dtypes(include=['int64', 'float64'])
    Q1 = numeric_cols.quantile(1/4)
    Q3 = numeric_cols.quantile(3/4)

    IQR = Q3 - Q1

    # * set bounds
    lower_b = Q1 - IQR * 1.5
    upper_b = Q3 + IQR * 1.5

    df = x_df[
        ~(
            (x_df[numeric_cols.columns] < lower_b)
            |
            (x_df[numeric_cols.columns] > upper_b)
        ).any(axis=1)
    ]

    return df

def feature_dti(x_df: pd.DataFrame) -> pd.DataFrame:
    x_df['dti'] = x_df['credit_amount'] / x_df['installment_commitment']

    return x_df

def feature_age(x_df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 20, 30, 40, 50, 60, 120]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']
    x_df['age_agg'] = pd.cut(
        x_df['age'],
        bins = bins,
        labels=labels,
        include_lowest=True,
    )

    return x_df.drop('age', axis=1)

def feature_credit_util(x_df:pd.DataFrame)->pd.DataFrame:
    x_df['credit_util'] = x_df['credit_amount'] / x_df['existing_credits']

    return x_df

def drop_num_dependents(x_df: pd.DataFrame)->pd.DataFrame:
    return x_df.drop('num_dependents', axis=1)

def process_data(x_df: pd.DataFrame) -> pd.DataFrame:
    df = (
        x_df
            .pipe(trim_df)
            .pipe(feature_dti)
            .pipe(feature_age)
            .pipe(feature_credit_util)
            .pipe(drop_num_dependents)
    )

    return df

def get_interquartile(xdf: pd.DataFrame) -> pd.DataFrame:
    num = xdf.select_dtypes(include=['float64', 'int64'])
    
    Q1 = num.quantile(1/4)
    Q3 = num.quantile(3/4)

    IQR: pd.DataFrame = Q3 - Q1
    IQR.columns = ['IQR']

    return IQR