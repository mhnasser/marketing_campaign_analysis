import pandas as pd
from etl import clustering_customers, predict_acceptance_proba, data_prep

df = pd.read_csv(
    r"C:\Users\m3830101\Documents\git\marketing_campaign_analysis\02_data\raw\ml_project1_data.csv"
)

df = data_prep(df)
df = clustering_customers(df)
df = predict_acceptance_proba(df)
