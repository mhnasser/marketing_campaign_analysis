import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


## Configs
# Paths
cluster_model_path = r"..\05_models\customer_cluster_model.pkl"
customer_acceptance_model_path = r"..\05_models\customer_customer_acceptance.pkl"

# Columns datatypes
categorical_columns = [
    "Response",
    "Complain",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Education",
    "Marital_Status",
    "accepted_cmp_before",
]

# Models
with open(cluster_model_path, "rb") as m:
    clustering_model = pickle.load(m)

with open(customer_acceptance_model_path, "rb") as m:
    customer_acceptance_model = pickle.load(m)


## Functions

# Calculated fields
def accepted_campaign(data):
    if (
        data["AcceptedCmp1"]
        == 1 | data["AcceptedCmp2"]
        == 1 | data["AcceptedCmp3"]
        == 1 | data["AcceptedCmp4"]
        == 1 | data["AcceptedCmp5"]
        == 1
    ):

        return 1

    else:

        return 0


def num_cmp_accepted(data):
    return (
        data["AcceptedCmp1"]
        + data["AcceptedCmp2"]
        + data["AcceptedCmp3"]
        + data["AcceptedCmp4"]
        + data["AcceptedCmp5"]
    )


# Etl
def data_prep(data):
    """[summary]

    Args:
        data ([type]): [description]
    """
    data.Income.fillna(0, inplace=True)
    data.Dt_Customer = pd.to_datetime(data.Dt_Customer)
    data["YearOfEnrollment"] = data.Dt_Customer.apply(lambda x: int(x.year))
    data["Age"] = data["YearOfEnrollment"] - data["Year_Birth"]
    data["YearsOfEnrollment"] = 2014 - data["YearOfEnrollment"]
    data["accepted_cmp_before"] = data.apply(accepted_campaign, axis=1)
    data["qtd_cmp_accepted"] = data.apply(num_cmp_accepted, axis=1)

    return data


# Predicting
def clustering_customers(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    df_aux = deepcopy(
        data[
            [
                "Income",
                "NumCatalogPurchases",
                "NumWebPurchases",
                "MntFruits",
                "NumStorePurchases",
                "MntMeatProducts",
                "MntSweetProducts",
                "MntWines",
                "MntFishProducts",
                "Age",
                "MntGoldProds",
                "Recency",
            ]
        ]
    )

    for col in categorical_columns:

        if col in df_aux.columns:
            df_aux[col] = df_aux[col].astype("category")
            df_aux[col] = df_aux[col].cat.codes

    X = df_aux.values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    data["Cluster"] = list(clustering_model.predict(X))

    return data


def predict_acceptance_proba(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    df_aux = deepcopy(
        data[
            [
                "MntFishProducts",
                "Cluster",
                "MntGoldProds",
                "MntSweetProducts",
                "Age",
                "Income",
                "NumWebVisitsMonth",
                "NumStorePurchases",
                "MntWines",
                "Marital_Status",
                "AcceptedCmp3",
                "NumCatalogPurchases",
                "YearsOfEnrollment",
                "AcceptedCmp5",
                "MntMeatProducts",
                "qtd_cmp_accepted",
                "Recency",
            ]
        ]
    )

    for col in categorical_columns:

        if col in df_aux.columns:
            df_aux[col] = df_aux[col].astype("category")
            df_aux[col] = df_aux[col].cat.codes

    X = df_aux.values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    probs = np.array(customer_acceptance_model.predict_proba(X))
    acceptance_proba_list = [x[1] for x in probs]

    data["acceptance_prob"] = acceptance_proba_list

    return data
