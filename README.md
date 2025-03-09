# Predicting Drug Resistance in Malaria Using Language Models

The goal of this project is to see if using a language model can be used to predict drug resistance within a malaria genome. This is done through the use of a custom BERT model, that checks if a strand of malaria has mutated. We can then speculate that the mutation is enough for the malaria strand to survive the vaccine.

## Getting the required files
- git clone
- git lfs?

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

## Building the project stages using script.py.
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
- The parameter `chunk_size` can be changed within the `config/data.json` file
    - This controls the chunk size for the data that is saved.

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
    - `model_name` is the name of the output model that is saved in `models/` folder
> [!WARNING]
> This step will take a long time to run depending on the given `training_size`. We recommend a GPU to speed up this process. The current training size is set to only run on 5 data points for testing purposes.