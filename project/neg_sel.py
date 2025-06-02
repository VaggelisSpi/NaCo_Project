import subprocess
import pandas as pd
import numpy as np


class NegativeSelection:
    def __init__(
        self,
        alphabet_path: str,
        self_path: str,
        n: int = 6,
        r_start: int = 1,
        r_stop: int = 2,
    ) -> None:
        self.alphabet_path = alphabet_path
        self.self_path = self_path
        self.n = n
        self.r_start = r_start
        self.r_stop = r_stop

    def run(self, data_path: str) -> None:
        """
        Run the negative selection algorithm for the data in the data path and save the results in a file

        :param data_path: The path to the file with the test data
        """
        for r in range(self.r_start, self.r_stop):
            cmd = (
                "java -jar negsel2.jar -alphabet " + self.alphabet_path +
                " -self " + self.self_path + " -n " + str(self.n) + " -l -c -r "
                + str(r) + " <" + data_path +
                "> ./data/results/" + str(data_path.split("/")[-1][:-4]) + "_r" + str(r) + ".txt"
            )
            subprocess.run(cmd, capture_output=True, shell=True)


def load_data(data_path: str, r: int, anomalous: int = 0) -> pd.DataFrame:
    """
    Load the strings and their scores in a dataframe along with the label

    There are two files, one that the data_path parameter points to that contains the strings, and the one under the 
    results directory with the scores for the data in data_path. For each different r value we trained the negative 
    selection algorithm, there will be a different score file.
    Finally the dataframe will contain a column called anomalous, that will indicate the class of the data. This is 
    determined from the parameter of the same name
    """
    data = pd.DataFrame()
    data["input"] = pd.read_csv(data_path, header=None)  # input sentence
    data["score"] = pd.read_csv("./data/results/" + str(data_path.split("/")[-1][:-4]) + "_r" + str(r) + ".txt", header=None).astype(np.float32)  # anomaly score
    data["anomalous"] = anomalous

    return data