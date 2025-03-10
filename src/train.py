#!/usr/bin/env python

"""
train.py trains the model with the classes and functions within model.py
"""

# Imports #
import warnings # Prevents popups of any possible warnings #
warnings.filterwarnings('ignore')
import os
import json
import tqdm
import torch
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import src.model as model

class ScheduledOptim():
    """
    Optimizer; A simple wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer, d_model, n_warmup_steps):

        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        """
        Step with the inner optimizer
        """

        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """
        Zero out the gradients by the inner optimizer
        """

        self._optimizer.zero_grad()

    def _get_lr_scale(self):

        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """
        Learning rate scheduling per step
        """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class BERTTrainer:
    """
    Trainer; creates models that are around 300 mb in size
    """

    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader=None,
        lr= 1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device='cuda'
        ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param #
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # Using Negative Log Likelihood Loss function for predicting the masked_token #
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print('\nTotal Parameters:', sum([p.nelement() for p in self.model.parameters()]), '\n')

    def train(self, epoch):

        self.iteration(epoch, self.train_data)

    def test(self, epoch):

        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        # Progress bar #
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu) #
            data = {key: value.to(self.device) for key, value in data.items()}

            in_bert = data["bert_input"]
            segmentlabel_bert = data["segment_label"]
            is_next_bert = data["is_next"]
            label_bert = data["bert_label"]
            
            # 1. forward the next_sentence_prediction and masked_lm model #
            next_sent_output, mask_lm_output = self.model.forward(in_bert, segmentlabel_bert, is_train=True)

            # 2-1. NLL(negative log likelihood) loss of is_next #
            # classification result #
            next_loss = self.criterion(next_sent_output, is_next_bert)

            # 2-2. NLLLoss of predicting masked token word #
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len) #
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), #
            # data["bert_label"].view(-1)) #
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), label_bert)

            # 2-3. Adding next_loss and mask_loss: 3.4 Pre-training Procedure # 
            loss = mask_loss

            # 3. backward and optimization only in train #
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # Next sentence prediction accuracy #
            correct = next_sent_output.argmax(dim=-1).eq(is_next_bert).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += is_next_bert.nelement()
        
        # Printing epoch #
        print(
            f'EP{epoch}, {mode}: '
            f'avg_loss={avg_loss / len(data_iter)}, '
            f'total_acc={total_correct * 100.0 / total_element:}\n'
        )

def train_model(seed, epochs, training_size, model_name):
    """
    Saves a model locally to be recalled again later

    seed (int): The random seed for pair shuffling
    epochs (int): 
    training_size (float): Percentage of data to train on
    model_name (str): Name for output model to be saved
    """

    # Loading paired data #
    with open('data/paired_data.txt', 'r') as f:
        pairs = json.loads(f.read())

    # Shuffling pairs #
    random.Random(seed).shuffle(pairs)

    # Getting pairs to train on based on training_size #
    train_pairs = pairs[:int(len(pairs) * (training_size / 100))]

    # Setting variables #
    t = model.tokenizer(
        vocab=['A', 'T', 'G', 'C'], 
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )
    train_data = model.BERTDataset(
        train_pairs,
        seq_len=1024,
        tokenizer=t
    )
    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        pin_memory=True
    )
    bert_model = model.BERT(len(t.vocab))
    bert_lm = model.BERTLM(bert_model, len(t.vocab))
    bert_trainer = BERTTrainer(bert_lm, train_loader, device='cuda')

    # Training data #
    for epoch in range(epochs):
        bert_trainer.train(epoch)
        if epoch % 1 == 0 and epoch != 0:
            filename = f'models/{model_name}'
            torch.save(bert_trainer.model.state_dict(), filename)
            print(
                'Model successfully trained! Model saved to filepath:',
                filename
            )