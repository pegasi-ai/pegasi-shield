import streamlit as st
from enum import Enum
from helper.add_logo import add_logo
from PIL import Image
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from helper.agstyler import PINLEFT, PRECISION_ONE, PRECISION_TWO, draw_grid, highlight

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Guardrail ML App",
    page_icon=im,
    layout="wide"
)

add_logo()

st.title("LLM Model Evaluation")
st.write(
        'Average evaluation and task values displayed in the range from 0 (worst) to 1 (best).'
)

# Add custom CSS styles
st.markdown(
    """
    <style>
    .css-15ftl5c {
        color: white !important;
    }
    .css-t4mh3n {
        color: white !important;
    }
    table.dataframe {
        width: 100%;
        height: 100%;
        padding-top: -20px;
        border-collapse: collapse;
    }
    div[data-testid="stTable"] table {
        width: 100%;
        height: 100%;
    }
    table.dataframe th, table.dataframe td {
        padding: 0px;
        text-align: center;
    }
    table.dataframe th {
        background-color: #555555;
        color: white;
        font-weight: bold;
    }
    table.dataframe td {
        background-color: white;
        font-weight: normal;
        border: none;
    }
    .table-content {
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.sidebar.success("Select a page above.")

models = ["dolly-v2-12b", "alpaca-lora-7b", "open_llama_13b", "mosaic-mpt-7b", "vicuna-13b", "gpt4"]

# Create a sample dataframe
data = {
    "Base Model": np.random.choice(models, size=30),
    "Version": np.random.uniform(1.0, 3.0, size=30),
    "Relevance": np.random.uniform(0, 1, size=30),
    "Toxicity": np.random.uniform(0, 1, size=30),
    "Bias": np.random.uniform(0, 1, size=30),
    "Quality": np.random.uniform(0, 1, size=30),
    "Security": np.random.uniform(0, 1, size=30),
    "Sentiment": np.random.uniform(0, 1, size=30),
    "TruthfulQA (MC) (0-s)": np.random.uniform(0, 1, size=30)
}

class Color(Enum):
    RED_LIGHT = "#fcccbb"
    GREEN_LIGHT = "#abf7b1"

condition_one_value = "params.value < 0.5"
condition_other_values = \
    "params.value > 0.8"

formatter = {
    'Base Model': ('Base Model', {'width': 80}),
    'Version': ('Version', {**PRECISION_ONE, 'width': 40}),
    'Relevance': ('Relevance', {**PRECISION_TWO, 'width': 50, 'cellStyle': highlight(
             Color.GREEN_LIGHT.value, condition_other_values
         )}),
    'Toxicity': ('Toxicity', {**PRECISION_TWO, 'width': 50, 'cellStyle': highlight(
             Color.GREEN_LIGHT.value, condition_other_values
         )}),
    'Bias': ('Bias', {**PRECISION_TWO, 'width': 50,
         'cellStyle': highlight(
             Color.RED_LIGHT.value, condition_one_value
         )}),
    'Quality': ('Quality', {**PRECISION_TWO, 'width': 50, 'cellStyle': highlight(
             Color.GREEN_LIGHT.value, condition_other_values
         )}),
    'Security': ('Security', {**PRECISION_TWO, 'width': 50, 'cellStyle': highlight(
             Color.RED_LIGHT.value, condition_one_value
         )}),
    'Sentiment': ('Sentiment', {**PRECISION_TWO, 'width': 50,'cellStyle': highlight(
             Color.GREEN_LIGHT.value, condition_other_values
         )}),
    'TruthfulQA (MC) (0-s)': ('TruthfulQA (MC) (0-s)', {**PRECISION_TWO, 'width': 50, 'cellStyle': highlight(
             Color.GREEN_LIGHT.value, condition_other_values
         )}),
}

# row_number = st.number_input('Number of rows', min_value=0, value=30)

df = pd.DataFrame(data)
data = draw_grid(
    df,
    formatter=formatter,
    fit_columns=True,
    selection='multiple',  # or 'single', or None
    use_checkbox='True',  # or False by default
    max_height=3000
)
