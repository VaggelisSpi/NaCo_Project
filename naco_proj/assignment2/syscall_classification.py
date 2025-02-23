# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
snd_cert_path = './data/syscalls/snd-cert'
snd_unm_path = './data/syscalls/snd-unm'

# %%
snd_cert_1_data = pd.DataFrame()
snd_cert_1_data["data"] = pd.read_csv(snd_cert_path + "/snd-cert.1.test")
snd_cert_1_data["label"] = pd.read_csv(snd_cert_path + "/snd-cert.1.labels")

snd_cert_2_data = pd.DataFrame()
snd_cert_2_data["data"] = pd.read_csv(snd_cert_path + "/snd-cert.2.test")
snd_cert_2_data["label"] = pd.read_csv(snd_cert_path + "/snd-cert.2.labels")

snd_cert_3_data = pd.DataFrame()
snd_cert_3_data["data"] = pd.read_csv(snd_cert_path + "/snd-cert.3.test")
snd_cert_3_data["label"] = pd.read_csv(snd_cert_path + "/snd-cert.3.labels")

snd_cert_train_data = pd.DataFrame()
snd_cert_train_data["data"] = pd.read_csv(snd_cert_path + "/snd-cert.train")

# %%
snd_unm_1_data = pd.DataFrame()
snd_unm_1_data["data"] = pd.read_csv(snd_unm_path + "/snd-unm.1.test")
snd_unm_1_data["label"] = pd.read_csv(snd_unm_path + "/snd-unm.1.labels")

snd_unm_2_data = pd.DataFrame()
snd_unm_2_data["data"] = pd.read_csv(snd_unm_path + "/snd-unm.2.test")
snd_unm_2_data["label"] = pd.read_csv(snd_unm_path + "/snd-unm.2.labels")


snd_unm_3_data = pd.DataFrame()
snd_unm_3_data["data"] = pd.read_csv(snd_unm_path + "/snd-unm.3.test")
snd_unm_3_data["label"] = pd.read_csv(snd_unm_path + "/snd-unm.3.labels")

snd_unm_train_data = pd.DataFrame()
snd_unm_train_data["data"] = pd.read_csv(snd_unm_path + "/snd-unm.train")


# %%
def preprocess_data_frames(df: pd.DataFrame) -> pd.DataFrame:
    # assign a unique id to each data element so when we split them in substrings, we can then get back to the original
    # and coclude about the final class
    df['id'] = range(len(df))

    df["length"] = df["data"].str.len()

    return df

def extract_substrings(df: pd.DataFrame, substr_len: int = 7) -> pd.DataFrame:
    """
    Create a new dataframe with all the subsrings from each row
    We will try to get non overlapping substrings, but if the text is not divisible by the desired length, then for the
    last substring we will get the last substr_len elements of the string

    This method returns a dataframe with the substrings and the label of the original string
    """
    substr_df = {"data": [], "label": []}
    for index, row in df.iterrows():
        text = row["data"]
        label = row["label"]
        substrings = []
        start = 0
        length = row["length"]
        while start < length:
            end = start + substr_len
            if end < length:
                substrings.append(text[start:end])
            else:
                substrings.append(text[-length:])
        substr_df["data"].extend(substrings)
        # substr_df["label"].extend(label for i in range(len(substrings)))

    print(substr_df["data"])
    # return pd.DataFrame(substr_df)
    return pd.DataFram


# %%
def analyse_df(df: pd.DataFrame, name: str) -> None:
    print("Counts for df " + name)
    display(df.groupby("length").count())


# %%
snd_cert_1_data = preprocess_data_frames(snd_cert_1_data)
snd_cert_2_data = preprocess_data_frames(snd_cert_2_data)
snd_cert_3_data = preprocess_data_frames(snd_cert_3_data)
snd_cert_train_data = preprocess_data_frames(snd_cert_train_data)

snd_unm_1_data = preprocess_data_frames(snd_unm_1_data)
snd_unm_2_data = preprocess_data_frames(snd_unm_2_data)
snd_unm_3_data = preprocess_data_frames(snd_unm_3_data)
snd_unm_train_data = preprocess_data_frames(snd_unm_train_data)

# analyse_df(snd_cert_1_data, "snd_cert_1_data")
# analyse_df(snd_cert_2_data, "snd_cert_2_data")
# analyse_df(snd_cert_3_data, "snd_cert_3_data")
# analyse_df(snd_cert_train_data, "snd_cert_train_data")

# analyse_df(snd_unm_1_data, "snd_unm_1_data")
# analyse_df(snd_unm_2_data, "snd_unm_2_data")
# analyse_df(snd_unm_3_data, "snd_unm_3_data")
# analyse_df(snd_unm_train_data, "snd_unm_train_data")

# %%
# snd_cert_1_substrs_data = extract_substrings(snd_cert_1_data)
# snd_cert_1_substrs_data.head(20)
# this is too slow currently

# %%

snd_cert_1_data.to_csv(snd_cert_path + "/snd_cert.1.csv")
snd_cert_2_data.to_csv(snd_cert_path + "/snd_cert.2.csv")
snd_cert_3_data.to_csv(snd_cert_path + "/snd_cert.3.csv")

snd_unm_1_data.to_csv(snd_unm_path + "/snd_unm.1.csv")
snd_unm_2_data.to_csv(snd_unm_path + "/snd_unm.2.csv")
snd_unm_3_data.to_csv(snd_unm_path + "/snd_unm.3.csv")
