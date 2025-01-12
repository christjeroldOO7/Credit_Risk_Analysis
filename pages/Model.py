from typing import List, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
from xgboost import XGBClassifier
import streamlit as st

_all_ = [
    "process_df",
    "xgbclassifier_model",
    "transform_for_pred",
    "LABEL_MAPPING",
    "SCALER_MAPPING",
]

BEST_PARAMS = dict(learning_rate=0.1, max_depth=5, n_estimators=500)
RANDOM_STATE = 38
LABEL_MAPPING: Dict[str, Dict[str, int]] = dict()
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
    category_cols = xdf.select_dtypes(include=["object", "category"]).columns
    for c in category_cols:
        le = LabelEncoder()
        xdf[c] = xdf[c].astype(str)  # Ensure all data is string before encoding
        le.fit(xdf[c])
        LABEL_MAPPING[c] = dict(zip(le.classes_, le.transform(le.classes_)))
        xdf[c] = le.transform(xdf[c])
    return xdf


def apply_smote(xdf: pd.DataFrame) -> pd.DataFrame:
    smote = SMOTE(random_state=RANDOM_STATE)
    y = xdf["class"]
    X = xdf.drop("class", axis=1)

    X_res, y_res = smote.fit_resample(X, y)
    X_res["class"] = y_res  # Add back the class column after resampling
    return X_res


def split_ml_df(
    xdf: pd.DataFrame, test_size: float, random_state=RANDOM_STATE
) -> List[pd.DataFrame]:
    return train_test_split(
        xdf.drop("class", axis=1),
        xdf["class"],
        test_size=test_size,
        random_state=random_state,
    )


def process_df(xdf: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    st.write("Initial Data Preview:")
    st.dataframe(xdf)  # Display initial data in Streamlit

    xdf = normalize_df(xdf)
    st.write("After Normalization:")
    st.dataframe(xdf)

    xdf = labels_encode(xdf)
    st.write("After Label Encoding:")
    st.dataframe(xdf)

    xdf = apply_smote(xdf)
    st.write("After SMOTE Resampling:")
    st.dataframe(xdf)

    splitted = split_ml_df(xdf, 0.2)
    keys = ["X_train", "X_test", "y_train", "y_test"]
    return {k: v for [k, v] in zip(keys, splitted)}


@st.cache_resource
def xgbclassifier_model(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(**BEST_PARAMS, random_state=RANDOM_STATE)

    try:
        model.fit(X_train, y_train)
        st.success("Model training completed successfully!")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None

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
    for k, v in form.items():
        if k in coll_cat:  # Categorical feature
            form[k] = get_label_encoding(coll_cat, k, v)
        elif k in coll_num:  # Numerical feature
            form[k] = scale_for_pred(coll_num, k, v)

    return pd.DataFrame(data=[form])


# Streamlit App
st.title("XGBoost Classifier with Preprocessing")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if "class" not in data.columns:
        st.error("The dataset must contain a 'class' column.")
    else:
        processed_data = process_df(data)
        X_train, X_test, y_train, y_test = (
            processed_data["X_train"],
            processed_data["X_test"],
            processed_data["y_train"],
            processed_data["y_test"],
        )

        if st.button("Train Model"):
            model = xgbclassifier_model(X_train, y_train)
            if model:
                accuracy = model.score(X_test, y_test)
                st.write(f"Model Accuracy: {accuracy:.2f}")
else:
    st.info("Please upload a dataset to begin.")
