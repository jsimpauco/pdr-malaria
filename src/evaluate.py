#!/usr/bin/env python

"""
evaluate.py TO ADD 
"""

# Imports #
import warnings # Prevents popups of any possible warnings #
warnings.filterwarnings('ignore')
import json
import torch
import numpy as np
import src.model as model

def calc_acc(model_name):

    # Setting variables #
    with open('data/paired_data.txt', 'r') as f:
        pairs = json.loads(f.read())

    t = model.tokenizer(
        vocab=['A', 'T', 'G', 'C'], 
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

    validation_data = model.BERTDataset(pairs[:int(len(pairs) * (0.010977430403091244 / 100))], seq_len=1024, tokenizer=t, is_train=False)
    bert_model = model.BERT(len(t.vocab))
    check_model = model.BERTLM(bert_model, len(t.vocab))
    check_model.load_state_dict(torch.load("models/"+model_name, map_location=torch.device('cpu')))

    accuracies = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ind = 0
    for data in validation_data:
        if ind % 1000 == 0:
            print(ind)
        in_bert = data["bert_input"].to(device)
        
        # Turns tensorized and tokenized input back into readable nucleotides #
        seq = t.convert_ids_to_tokens(in_bert.tolist())

        # Turns tensorized and tokenized output back into readable nucleotides #
        output = check_model(in_bert.reshape(1,-1))

        # .max grabs the option with the largest weight #
        tensor_pred = torch.max(output[:512], axis=-1)[1]
        pred = tensor_pred.tolist()
        pred = sum(pred, [])
        pred = t.convert_ids_to_tokens(pred)
        
        num_correct = 0
        for i in range(len(seq)):
            if seq[i] == pred[i]:
                num_correct = num_correct + 1
        acc = num_correct / len(seq)
        accuracies.append(acc)
        ind = ind + 1

    print(np.mean(accuracies))