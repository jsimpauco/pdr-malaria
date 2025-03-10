# Predicting Drug Resistance in Malaria Using Language Models

The goal of this project is to see if using a language model can be used to predict drug resistance within a malaria genome. This is done through the use of a custom BERT model, that checks if a strand of malaria has mutated. We can then speculate that the mutation is enough for the malaria strand to survive the vaccine.

## Table of Contents
- [Getting the required files](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#getting-the-required-files)
- [Creating environment](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#creating-environment)
- [Building the project stages using script.py](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#building-the-project-stages-using-scriptpy)
    - [Data](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#data)
    - [Train](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#train)
    - [Evaluate](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#evaluate)
    - [Compare](https://github.com/jsimpauco/pdr-malaria?tab=readme-ov-file#compare)

## Getting the required files
To get the required files, [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) is required to be downloaded. Once downloaded and installed, initialize `Git LFS` by running
```
git lfs install
```
Once initialized, the repository can be cloned by running
```
git clone https://github.com/jsimpauco/pdr-malaria.git
```
Finally, once cloned, run the following command from the root directory of the project:
```
git lfs pull
```
> [!NOTE]
> `git lfs pull` may not be needed, but should be ran just in case.

The required files for the project should now be successfully acquired.

## Creating environment
This project uses a `conda` environment for all dependencies.

Instructions to recreate project:
- First, to install the dependencies, run the following command from the root directory of the project:
```
conda env create -f environment.yml
```
- To activate the environment, run the following command:
```
conda activate pdr
```

## Building the project stages using script.py
This project has **four** total stages to build the whole project: `data`, `train`, `evaluate`, and `compare`
- To run the whole project (i.e., all four stages in order), from the project root, run
```
python script.py all
```
> [!NOTE]
> This will run the whole script with one argument. Keep reading below if you would like the build the script at a certain step.

### Data
To get the data, from the project root, run
```
python script.py data
```
- This fetches the data and creates many new files, stored in the data folder. The most important file is the `paired_data.txt` file, which will be used to train the model.
- The parameter `chunk_size` can be changed within the `config/data.json` file.
    - `chunk_size` controls the chunk size for the data that is saved

### Train
To train the model, from the project root, run
```
python script.py train
```
- This trains the model based on the classes and functions within `model.py`.
- The parameters `seed`, `epochs`, `training_size`, and `model_name` can be changed within the `config/train.json` file.
    - `seed` controls the random seed used to shuffle the paired data
    - `epochs` determine the amount of epochs for the data to train on
    - `training_size` is a percentage of the dataset to train the data on
    - `model_name` is the name of the output model that is saved in `models` folder
> [!WARNING]
> This step will take a long time to run depending on the given `training_size`. We recommend a GPU to speed up this process. The current training size is set to only run on 5 data points for testing purposes.

### Evaluate
To evaluate the model, from the project root, run
```
python script.py evaluate
```
- This evaluates the model through the calculation of accuracy score on validation set.
- The parameters `validation_size` and `model_name` can be changed within the `config/evaluate.json` file.
    - `validation_size` is a percentage of the dataset to validate the model on
    - `model_name` is the name of the model that will be evaluated
> [!WARNING]
> This step will take a long time to run depending on the given `validation_size`. We recommend a GPU to speed up this process. The current validation size is set to only run on 5 data points for testing purposes.

### Compare
To compare model accuracies, from the project root, run
```
python script.py compare
```
- This compares the model's accuracy scores on the validation set.
- The parameter `add_model` can be changed within the `config/compare.json` file.
    - `add_model` is a boolean that determines whether or not to compare the model within the `config/evaluate.json` file
> [!IMPORTANT]
> `compare` uses the `config/evaluate.json` for the model that will be added into the comparison with the 3 other base models and the validation size for the set of data. This step will also take a long time depending on the given `validation_size`.