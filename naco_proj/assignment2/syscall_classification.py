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

# %% [markdown]
# # Imports and setup

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
snd_cert_path = './data/syscalls/snd-cert'
snd_unm_path = './data/syscalls/snd-unm'

# %% [markdown]
# ## Load the data

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


# %% [markdown]
# # Preprocessing

# %%
def preprocess_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform any preprocessing steps in a single data frame

    We assign a new id column and a new column with the length of the data
    """
    # assign a unique id to each data element so when we split them in substrings, we can then get back to the original
    # and conclude about the final class
    df['id'] = range(len(df))

    df["length"] = df["data"].str.len()

    return df

def extract_substrings(df: pd.DataFrame, substr_len: int = 7) -> pd.DataFrame:
    """
    Create a new dataframe with all the substrings from each row
    We will try to get non overlapping substrings, but if the text is not divisible by the desired length, then for the
    last substring we will get the last substr_len elements of the string

    This method returns a dataframe with the substrings and the label of the original string
    """
    # note for improvement, check the apply method of data frame paired with explode
    substr_df = {"data": [], "label": [], "id": []}
    for index, row in df.iterrows():
        text = row["data"]
        label = row["label"]
        id = row["id"]
        substrings = []
        length = row["length"]
        start = 0
        while start < length:
            end = start + substr_len
            if end < length:
                substrings.append(text[start:end])
            else:
                substrings.append(text[-substr_len:])
            start = end

        substr_df["data"].extend(substrings)
        substr_df["label"].extend(label for i in range(len(substrings)))
        substr_df["id"].extend(id for i in range(len(substrings)))

    return pd.DataFrame(substr_df)

def preprocess(df: pd.DataFrame, name: str, data_path: str, is_train: bool = False) -> None:
    """
    Apply all the preprocessing steps in a dataframe and save all the resulting dataframes
    """
    df = preprocess_data_frame(df)
    df.to_csv(data_path + "/" + name + ".csv")

    # extract substrings and save the new dataframe
    df_substr = extract_substrings(df)
    df_substr.to_csv(data_path + "/" + name + "_substr.csv")

    # save the data in a file to be used in the negative selection algorithm
    df_substr["data"].to_csv(data_path + "/" + name + "_substr" + ".train" if is_train else ".test", header=False, index=False)



# %%
def analyse_df(df: pd.DataFrame, name: str) -> None:
    print("Counts for df " + name)
    display(df.groupby("length").count())


# %% [markdown]
# Preprocess and save all the data

# %%
preprocess(snd_cert_1_data, "snd_cert_1", snd_cert_path)
preprocess(snd_cert_2_data, "snd_cert_2", snd_cert_path)
preprocess(snd_cert_3_data, "snd_cert_3", snd_cert_path)
preprocess(snd_cert_train_data, "snd_cert_train", snd_cert_path, True)

preprocess(snd_unm_1_data, "snd_unm_1", snd_unm_path)
preprocess(snd_unm_2_data, "snd_unm_2", snd_unm_path)
preprocess(snd_unm_3_data, "snd_unm_3", snd_unm_path)
preprocess(snd_unm_train_data, "snd_unm_train", snd_unm_path, True)
