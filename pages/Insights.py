import os
import sys
import logging
import streamlit as st
from PIL import Image

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Ensure the directory containing 'data.py' is in the sys.path
sys.path.append('./data')  # Update with the actual path to 'data.py'

# Import local modules
import data
import model

# Dynamically construct paths
current_dir = os.path.dirname(_file_)
assets_dir = os.path.join(current_dir, 'assets')

# Verify and log paths
logging.debug(f"Assets Directory: {assets_dir}")

# Image import with dynamically constructed paths
CATEGORY_COUNTPLOT = Image.open(os.path.join(assets_dir, 'category_countplot.png'))
CATEGORY_PIE = Image.open(os.path.join(assets_dir, 'category_pie.png'))
CORR_NUMERIC = Image.open(os.path.join(assets_dir, 'corr_numeric.png'))
CORR_NUMERIC_PROCESSED = Image.open(os.path.join(assets_dir, 'corr_numeric_processed.png'))
NUMERIC_COUNTPLOT = Image.open(os.path.join(assets_dir, 'numeric_countplot.png'))
NUMERIC_KDE = Image.open(os.path.join(assets_dir, 'numeric_kde.png'))

# Streamlit page configuration
st.set_page_config(page_title="Data Insights", layout="wide")

# Data initialization
df_clean = data.get_cleaned_data(os.path.join(current_dir, "credit_customers.csv"))
df = data.process_data(df_clean)

# Page title
st.title("Data Insights")

# Descriptive Statistics Section
st.header("Descriptive Statistics")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Overview")
    st.write(df.select_dtypes(include=['float64', 'int64']).describe())

with col2:
    st.subheader("Inter-quartile Range")
    st.write(data.get_interquartile(df_clean))

st.divider()

# Visual Description Section
st.header("Visual Description")

# Category Section
st.subheader("Category")

col1, col2 = st.columns(2)

with col1:
    st.image(CATEGORY_COUNTPLOT, caption='Countplot')

with col2:
    st.image(CATEGORY_PIE, caption='Pie Plot')

# Numeric Section
st.subheader("Numeric")

col1, col2 = st.columns(2)

with col1:
    st.image(NUMERIC_COUNTPLOT, caption='Countplot')

with col2:
    st.image(NUMERIC_KDE, caption='KDE')

# Correlations Section
st.subheader("Correlations")

col1, col2 = st.columns(2)

with col1:
    st.image(CORR_NUMERIC, caption='Correlation prior-processing')

with col2:
    st.image(CORR_NUMERIC_PROCESSED, caption='Correlation post-processing')
