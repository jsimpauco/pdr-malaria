#!/usr/bin/env python

"""
evaluate.py calculates the accuracy for a given model using validation data
"""

# Imports #
import warnings # Prevents popups of any possible warnings #
warnings.filterwarnings('ignore')
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import src.model as model

def calc_acc(validation_size, model_name):
    """
    Calculates the accuracy of the given model

    validation_size (float): Percentage of data to validate on
    model_name (str): Name of model to be evaluated
    """

    print(f'\nCurrently evaluating: {model_name}')

    # Getting validation data #
    with open('data/paired_data.txt', 'r') as f:
        pairs = json.loads(f.read())
    random.Random(4).shuffle(pairs)
    t = model.tokenizer(
        vocab=['A', 'T', 'G', 'C'], 
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )
    validation_resized = 100 - validation_size
    validation_data = model.BERTDataset(
        pairs[int(len(pairs) * (validation_resized / 100)):],
        seq_len=1024,
        tokenizer=t,
        is_train=False
    )

    # Setting variables #
    bert_model = model.BERT(len(t.vocab))
    check_model = model.BERTLM(bert_model, len(t.vocab))
    check_model.load_state_dict(torch.load(
        'models/'+model_name,
        map_location=torch.device('cpu')
        )
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For loop to calculate accuracy #
    accuracies = []
    ind = 0
    total_points = len(validation_data)
    for ind, data in enumerate(tqdm(
            validation_data,
            desc='Evaluating',
            total=total_points,
            unit='point'
        )
    ):

        in_bert = data["bert_input"].to(device)
        
        # Turns tensorized and tokenized input back into readable nucleotides #
        seq = t.convert_ids_to_tokens(in_bert.tolist())

        # Turns tensorized and tokenized #
        # output back into readable nucleotides #
        output = check_model(in_bert.reshape(1,-1))

        # .max grabs the option with the largest weight #
        tensor_pred = torch.max(output[:512], axis=-1)[1]
        pred = tensor_pred.tolist()
        pred = sum(pred, [])
        pred = t.convert_ids_to_tokens(pred)
        
        # Calcuating accuracy #
        num_correct = 0
        for i in range(len(seq)):
            if seq[i] == pred[i]:
                num_correct = num_correct + 1
        acc = num_correct / len(seq)
        accuracies.append(acc)
        ind = ind + 1

    accuracy = np.mean(accuracies)
    print(f'\nAccuracy of {model_name}: {accuracy}')
    return accuracy