# Project

## Introduction
In this project we experimented with language and peptide data. We have two notebooks which contain the code of our experiments, `language_discrimination.ipynb` and `calculations.ipynb` respectively.


For this project we created 3 different datasets to train a negative selection algorithm. The code for each is under the following files:

- `create_random_dataset.py`: Creates the random dataset
- `evolutionary_algorithm.py`: Code to run the evolutionary algorithm and create the respective dataset
- `greedy_algorithm.py`: Code to run a greedy algorithm and create the respective dataset

The file `neg_sel.py` contains the code to run the negative selection algorithm, and it uses `negsel2.jar` to achieve this

## How to run

### Language Discrimination
In order to run the language discrimination experiments you need to run the `language_discrimination.ipynb`. The notebook creates the 3 datasets, runs the negative selection algorithm and create plots to evaluate the performance. 

The code under the section Dataset Creation generates the datasets we used for these experiments. The files are stored under `data/languages/`. All the files for the english data are under `data/languages/english/` while the rest are under `data/languages/other/`. We then create the three self sets.

#### Greedy Algorithm
First there is the code for the greedy algorithm. A run of the code blocks in this section will generate all the motifs of length 6 that can be created from the characters `_abcdefghijklmnopqrstuvwxyz`, keep 1% of these motifs and then instantiate an object of the class `GreedyAlgorithm`, which is defined in `greedy_algorithm.py`, and run it. The resulting dataset is then saved in a file. The parameters for the greedy algorithm are the ones we used in all our runs. 

#### Random dataset
Next, we generate the random dataset, with the `create_random_dataset` which is defined in `create_random_dataset.py` and save it in a file. The parameters for the random dataset are the ones we used in all our runs. 

#### Evolutionary algorithm
Lastly we have the code for the evolutionary algorithm. In the first code block in this section, we create all the methods we need to calculate all the features we used in the fitness function. The fitness is calculated in the `composite_fitness` method. You can change the factors in the return statement to test the different weights we used in our experiments. Lastly we will run the evolutionary algorithm. The parameters for it, are the ones we used across all our experiments. We only modified the weights for the different features in the fitness function. The results are then saved in a file. We used the format of `english_6_ea_XYZ.txt`, where `XYZ`are the different weights of the features in the fitness function.


#### Negative Selection
Next, we run the negative selection algorithm. In order to do this, we first create an object of the class `NegativeSelection`. The first parameter in the constructor is the path to the alphabet, the second is the path to the self set, the `r_start` and `r_stop` values are the range of r values we want to examine, and they correspond to the r parameter of `negsel2.jar`. Lastly, we pass n, the length of the strings. 

In order to run the negative selection algorithm for a given dataset we need to call the method `run` of the `NegativeSelection` class. This method takes 3 parameters. The path to the data we want to test for, the path to the directory of the results, and a postfix. This is added to the end of the file name that holds the results.

If we run an experiment with the self set as `data/languages/english/english_6_ea_010.txt`, for modern english which are saved under `data/languages/english/english_6_test.txt`, and we define the result path as `data/languages/results/` and the postfix as `ea_010`, then all the resulting files will be saved under `data/languages/results/`. The names will have the form `english_6_test_rX_ea_010.txt`, where `X` is the value of r. If we use `r_start=1` and `r_stop=6`, then we will have 6 files for `english_6_test.txt`. In our code we run this and then loop over all the other languages.

Finally, we generate the plots for the results. We once again loop over all languages, and then for each r value we read the results for the test english and the given language and generate the roc plot, as well as calculate the accuracy, error rate, precision, recall and F1 scores. In the final experiments we only looked at the ROC curves, so the code bellow can be ignored.

### Foreign Peptide Detection
The same process is followed in the `calculations.ipynb` file for the peptides problem. The paths to the data are slightly different, as the files for the self and foreign sets are stored under `/data/peptides/`, while the self datasets  created by our algorithms are stored under `/data/sampled/`, and the results of the negative selection algorithm are stored under `/data/results/`