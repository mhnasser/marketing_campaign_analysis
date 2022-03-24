import sys
import os
import streamlit as st
import pandas as pd
import pickle
from io import BytesIO, StringIO


## Configs

# Style
STYLE = """
<style>
img {
    max-width: 100%;
}
<style>
"""

# Paths
cluster_model_path = r"..\05_models\customer_cluster_model.pkl"
customer_acceptance_model_path = r"..\05_models\customer_customer_acceptance.pkl"

# Models
with open(cluster_model_path, "rb") as m:
    clustering_model = pickle.load(m)

with open(customer_acceptance_model_path, "rb") as m:
    customer_acceptance_model = pickle.load(m)


def main():

    header = st.container()
    st.image(r"..\01_images\app\logo.png", use_column_width=True)
    with header:
        st.title("Marketing Campaign Optimizer")

    # st.info(__doc__)
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


main()
