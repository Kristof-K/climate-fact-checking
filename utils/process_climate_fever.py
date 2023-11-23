import pandas as pd
import os

from constants import DATA_PATH


def get_claims_and_labels():
    # go through statements and derive whether they are true or not
    data_df = pd.read_json(os.path.join(DATA_PATH, 'climate-fever.jsonl'), lines=True)
    n_rows = data_df.shape[0]

    claims = [data_df.loc[i, 'claim'] for i in range(n_rows)]
    labels = [data_df.loc[i, 'claim_label'] for i in range(n_rows)]

    return claims, labels
