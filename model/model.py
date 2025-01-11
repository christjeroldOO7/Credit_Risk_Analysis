from typing import List, Dict, Union
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
from xgboost import XGBClassifier
import streamlit as st

__all__ = [
    "process_df",
    "xgbclassifier_model",
    "transform_for_pred",
    "LABEL_MAPPING",
    "SCALER_MAPPING",
]

BEST_PARAMS = dict(learning_rate=0.1, max_depth=5, n_estimators=500)
RANDOM_STATE = 38
LABEL_MAPPING: Dict[str, LabelEncoder] = dict()
SCALER_MAPPING: Dict[str, MinMaxScaler] = dict()


def normalize_df(xdf: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = xdf.select_dtypes(include=["float64", "int64"]).columns

    for c in numeric_cols:
        scaler = MinMaxScaler()
        scaler.fit(xdf[c].to_numpy().reshape(-1, 1))
        SCALER_MAPPING[c] = scaler
        xdf[c] = scaler.transform(xdf[c].to_numpy().reshape(-1, 1))

    return xdf


def labels_encode(xdf: pd.DataFrame) -> pd.DataFrame:
    category_cols = xdf.select_dtypes("category").columns

    for c in category_cols:
        le = LabelEncoder()
        le.fit(xdf[c])
        LABEL_MAPPING[c] = dict(zip(le.classes_, le.transform(le.classes_)))
        xdf[c] = le.transform(xdf[c])

    return xdf


def apply_smote(xdf: pd.DataFrame) -> pd.DataFrame:
    smote = SMOTE(random_state=RANDOM_STATE)
    y = xdf["class"]
    X = xdf.drop("class", axis=1)

    X, y = smote.fit_resample(X, y)

    return pd.concat([X, y], axis=1)


def split_ml_df(
    xdf: pd.DataFrame, test_size: float, random_state=RANDOM_STATE
) -> List[pd.DataFrame]:
    return train_test_split(
        xdf.drop("class", axis=1),
        xdf["class"],
        test_size=test_size,
        random_state=random_state,
    )


def process_df(xdf: pd.DataFrame) -> Dict["str", pd.DataFrame]:
    xdf = xdf.pipe(normalize_df).pipe(labels_encode).pipe(apply_smote)

    splitted = split_ml_df(xdf, 0.2)
    keys = ["X_train", "X_test", "y_train", "y_test"]

    return {k: v for [k, v] in zip(keys, splitted)}


@st.cache_resource
def xgbclassifier_model(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(**{**BEST_PARAMS, "random_state": RANDOM_STATE})

    model.fit(X_train, y_train)

    return model


def get_label_encoding(coll: Dict[str, Dict[str, int]], label: str, key: str) -> int:
    return coll[label][key]


def scale_for_pred(
    coll: Dict[str, MinMaxScaler], label: str, val: Union[int, float]
) -> float:
    scaler: MinMaxScaler = coll[label]
    val = [[val]]

    return scaler.transform(val)[0][0]


def transform_for_pred(
    coll_num: Dict[str, MinMaxScaler],
    coll_cat: Dict[str, Dict[str, int]],
    form: Dict[str, Union[str, int, float]],
) -> pd.DataFrame:
    for [k, v] in form.items():
        try:
            temp = coll_cat[k]
            form[k] = get_label_encoding(coll_cat, k, v)
        except KeyError:
            form[k] = scale_for_pred(coll_num, k, v)

    return pd.DataFrame(data = form.values(), index= form.keys()).T
