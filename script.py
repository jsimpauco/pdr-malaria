#!/usr/bin/env python

# Imports #
import warnings # Prevents popups of any possible warnings #
warnings.filterwarnings('ignore')
import sys
import json
import src.data as data
import src.train as train

# Checking if script.py is being run as a script in command line #
if __name__ == '__main__':

    # Getting args #
    args = sys.argv[1:]

    # all argument #
    if 'all' in args:

        print("\n'all' argument given. Running whole script...")

        # Setting args to all available arguments #
        args = [
            'data',
            'train'
        ]
    # Other arguments given #
    else:
        print('\nArguments given: ' + ', '.join(args))
        print('\nRunning script based on given arguments...')

    # data argument #
    if 'data' in args:

        print('\nCurrently running: data.py')

        config = json.load(open('config/data.json'))

        # Creating helper data #
        data.create_helperdata(**config)

        # Creating paired data #
        data.create_pairedData()

    # train argument #
    if 'train' in args:

        print('\nCurrently running: train.py')

        config = json.load(open('config/train.json'))

        # Training model #
        train.train_model(**config)

    print('\nScript successfully ran!')