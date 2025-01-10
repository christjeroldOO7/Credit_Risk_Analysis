import streamlit as st
from PIL import Image
import sys

# Ensure the directory containing 'data.py' is in the sys.path if not in the same folder
sys.path.append('../src/data')  # Update with the actual path to the directory containing 'data.py'

import data
import model

# Image import
CATEGORY_COUNTPLOT = Image.open('../assets/category_countplot.png')
CATEGORY_PIE = Image.open('../assets/category_pie.png')
CORR_NUMERIC = Image.open('../assets/corr_numeric.png')
CORR_NUMERIC_PROCESSED = Image.open('../assets/corr_numeric_processed.png')
NUMERIC_COUNTPLOT = Image.open('../assets/numeric_countplot.png')
NUMERIC_KDE = Image.open('../assets/numeric_kde.png')

st.set_page_config(page_title="Data Insights", layout="wide")

# * data init
df_clean = data.get_cleaned_data("../credit_customers.csv")
df = data.process_data(df_clean)

st.title("Data Insights")
st.header("Descriptive Statistics")

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Overview")
    st.write(
        df.select_dtypes(include=['float64', 'int64']).describe()
    )

with col2:
    st.subheader("Inter-quartile Range")
    st.write(
        data.get_interquartile(df_clean)
    )

st.divider()

st.header("Visual Description")
st.subheader("Category")

col1, col2 = st.columns(2)

with col1:
    st.image(
        CATEGORY_COUNTPLOT,
        caption='Countplot'
    )

with col2:
    st.image(
        CATEGORY_PIE,
        caption='Pie Plot'
    )

st.subheader("Numeric")

col1, col2 = st.columns(2)

with col1:
    st.image(
        NUMERIC_COUNTPLOT,
        caption='Countplot'
    )

with col2:
    st.image(
        NUMERIC_KDE,
        caption='KDE'
    )

st.subheader("Correlations")

col1, col2 = st.columns(2)

with col1:
    st.image(
        CORR_NUMERIC,
        caption='Correlation prior-processing'
    )

with col2:
    st.image(
        CORR_NUMERIC_PROCESSED,
        caption='Correlation past-processing'
    )
