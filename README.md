# Predicting Drug Resistance in Malaria Using Language Models

The goal of this project is to see if using a language model can be used to predict drug resistance within a malaria genome. This is done through the use of a custom BERT model, that checks if a strand of malaria has mutated. We can then speculate that the mutation is enough for the malaria strand to survive the vaccine.

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
This project has CHANGE total stages to build the whole project: `data`
- To run the whole project (i.e., all five stages in order), from the project root, run
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
- This fetches the data and creates many new files. The most important file is the `paired_data.txt` file, which will be used to train the model.