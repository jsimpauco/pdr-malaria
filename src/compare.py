#!/usr/bin/env python

"""
compare.py compares models accuracy against one another. Use this to compare
a user created model to the 3 base models that were included
"""

# Imports #
import warnings # Prevents popups of any possible warnings #
warnings.filterwarnings('ignore')
import json
import src.evaluate as evaluate

def compare(add_model):
    """
    Function is used to compare a model to the 3 base models that were created

    add_model (boolean): If True, adds the model within evaluate.json to the
    list of models to be compared. If False, only compares the 3 base models
    """
    
    # Getting config #
    eval_config = json.load(open('config/evaluate.json'))
    val_size, user_model = eval_config.values()

    # Adding user model to list of models if add_model is True #
    models = [
        'smallMalariaModelEpoch10',
        'largeMalariaModelEpoch1',
        'largeMalariaModelEpoch2'
    ]
    if add_model:
        models.insert(0, user_model)

    # Looping through models to evaluate all accuracies #
    accuracies = {}
    for model in models:

        # Calculating accuracy for current model #
        acc = evaluate.calc_acc(val_size, model)

        # Appending to dictionary #
        accuracies[model] = acc

    # Printing accuracies #
    print('\n\nAccuracies for each model:')
    for key, value in accuracies.items():
        print(f'{key}: {value:.5f}')