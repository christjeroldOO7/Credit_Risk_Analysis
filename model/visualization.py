from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np

__all__ = [
    'plot_roc',
    'plot_feature_importance'
]

def plot_roc(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    roc = go.Scatter(
        x = fpr,
        y = tpr,
        name = 'XGBoost',
    )

    rand_guess = go.Scatter(
        x = [0, 1],
        y = [0, 1],
        line= go.scatter.Line(
            color='purple',
            dash='dash'
        ),
        name = 'Random Guess',
    )

    fig = make_subplots()
    fig.add_trace(roc)
    fig.add_trace(rand_guess)
    # fig.update_layout(
    #     title_text = 'ROC Curve'
    # )

    return fig

def plot_feature_importance(model: XGBClassifier, X_train: pd.DataFrame) -> go.Figure:
    feat_imp = model.feature_importances_
    idxs = np.argsort(feat_imp)
    lbl =  [X_train.columns[i] for i in idxs]
    fig = px.bar(
        x = np.sort(feat_imp),
        y = lbl,
        orientation='h',
        labels= dict(
            x = 'weight',
            y = 'feat',
        ),
    )
    fig.update_traces(
        marker_color = 'green'
    )

    return fig