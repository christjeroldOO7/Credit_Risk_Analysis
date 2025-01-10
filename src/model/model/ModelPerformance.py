from xgboost import XGBClassifier
from typing import Union
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


class ModelPerformance:
    def __init__(
        self,
        model: XGBClassifier,
        y_pred: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> None:
        self.model = model
        self.c_report = classification_report(y_test, y_pred, output_dict=True)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @property
    def accuracy(self) -> float:
        return self.c_report["accuracy"]

    @property
    def recall(self) -> float:
        return self.c_report["macro avg"]["recall"]

    @property
    def f1_score(self) -> float:
        return self.c_report["macro avg"]["f1-score"]

    @property
    def train_auc(self) -> float:
        train_pred_proba = self.model.predict_proba(self.X_train)[:, 1]
        return roc_auc_score(self.y_train, train_pred_proba)

    @property
    def test_auc(self) -> float:
        test_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        return roc_auc_score(self.y_test, test_pred_proba)

    @classmethod
    def fmt_float_to_str(cls, val: Union[float, int]) -> str:
        return f"{val: .1%}"
