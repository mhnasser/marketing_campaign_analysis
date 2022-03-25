import sys
import os
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO, StringIO
from optimizer import optimized_customer_selection
from etl import clustering_customers, predict_acceptance_proba, data_prep


## Configs

# Style
STYLE = """
<style>
img {
    max-width: 100%;
}
<style>
"""


def main():

    header = st.container()
    st.image(r"..\01_images\app\logo.png", use_column_width=True)

    with header:
        st.title("Marketing Campaign Optimizer")
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader(
        "Upload Customer Data", type=["csv"], accept_multiple_files=False
    )
    show_file = st.empty()

    if not file:
        show_file.info("Upload file")
        return

    content = file.getvalue()
    data = pd.read_csv(file)
    st.dataframe(data.head(5))
    file.close()

    budget_input = st.number_input("Budget available for campaign", min_value=0)
    contact_cost = st.number_input("Contact Cost per customer", min_value=0)
    profit = st.number_input("Profit if customer accepts the campaign", min_value=0)


main()
