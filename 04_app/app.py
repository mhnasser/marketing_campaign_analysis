from cProfile import label
import streamlit as st
import pandas as pd
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

    # Receving the data
    content = file.getvalue()
    data = pd.read_csv(file)
    st.dataframe(data.head(5))
    file.close()

    # Preparing DataFrame
    data = data_prep(data)
    data = clustering_customers(data)
    data = predict_acceptance_proba(data)

    # Setting campaing variables
    with st.form(key="form1"):
        budget_input = st.number_input("Budget available for campaign", min_value=None)
        contact_cost = st.number_input("Contact Cost per customer", min_value=None)
        sucess_profit = st.number_input(
            "Profit if customer accepts the campaign", min_value=None
        )

        submit = st.form_submit_button(label="Submit")

    if submit:
        data, expected_profit, amount_invested = optimized_customer_selection(
            data, contact_cost, sucess_profit, budget_input
        )

        col_1, col_2 = st.columns(2)

        with col_1:
            st.metric(label="Amount to be invested", value="$ %.2f" % amount_invested)

        with col_2:
            st.metric(label="Liquid Profit expected", value="$ %.2f" % expected_profit)

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df(data)

        st.download_button(
            "Download Customers of interest file",
            csv,
            "best_customers.csv",
            "text/csv",
            key="download-csv",
        )


main()
