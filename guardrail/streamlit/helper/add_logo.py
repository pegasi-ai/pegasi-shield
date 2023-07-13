import base64

import pkg_resources
import streamlit as st

def add_logo():
    logo = open(
        pkg_resources.resource_filename('helper', 'guardrail_logo.svg'),
        "rb"
    ).read()
    logo_encoded = base64.b64encode(logo).decode()

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/svg+xml;base64,{logo_encoded}');
                background-size: 100% auto;
                padding-top: 60px;
                background-position: 0px 0px;
                background-repeat: no-repeat;
            }}
            [data-testid="grSidebarNav"]::before {{
                margin-left: 0px;
                margin-top: 0px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )