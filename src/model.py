#!/usr/bin/env python

"""
model.py stores all the model classes and functions to create the model 
"""

# Imports #
import warnings # Prevents popups of any possible warnings #
warnings.filterwarnings('ignore')
import math
import torch
import random
import itertools
import torch.nn.functional as F
from torch.utils.data import Dataset

class tokenizer():
    """
    Turns input nucleotides into appropriate integer label
    """

    def __init__(self, vocab, special_tokens):
        """
        vocab (list): given as a list of all words, but will turn into a
        dictionary with words as keys and integers as values
        special_tokens (list): extra symbols that aren't standard words,
        but rather used to delimit or do something within the text
        """

        self.vocab = special_tokens + vocab
        self.tokens = {}
        d = {}
        for i in range(len(self.vocab)):
            d[self.vocab[i]] = i
            self.tokens[i] = self.vocab[i]
        self.vocab = d

    def convert_ids_to_tokens(self, ids):
        """
        This function will convert each integer into a word

        ids (list): list of integers that represent the encoded words, ex,
        ids = [0, 3, 4, 1]
        dict={0: 'word1', 1: 'word2', 2:'word3', 3:'word4', 4:'word5'}
        """

        self.output_ids = []
        for oneid in ids:
            if oneid not in self.tokens.keys():
                self.input_ids.append(self.tokens['[UNK]'])
                continue
            self.output_ids.append(self.tokens[oneid])
        return self.output_ids

    def __call__(self, sequence):
        """
        Use tokenizer on a sequence of nucleotides to output the resulting
        integers that are mapped to each nucleotide

        sequence (list): list of nucleotides
        """

        self.input_ids = {'input_ids':[]}
        if not isinstance(sequence, list):
            sequence = [sequence]
        for nucleotide in sequence:
            if nucleotide not in self.vocab.keys():
                self.input_ids['input_ids'].append(self.vocab['[UNK]'])
                continue
            self.input_ids['input_ids'].append(self.vocab[nucleotide])
        return self.input_ids

class BERTDataset(Dataset):
    """
    Takes in paired nucleotide data and tokenizes and tensorizes it to prepare
    for training or prediction. Will mask portions of the nucleotide sequence
    when training
    """

    def __init__(self, data_pair, tokenizer, seq_len=1024, is_train=False):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
        self.is_train = is_train

    def __len__(self):
        
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive #
        # (saved as is_next_label) #
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words #
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
            

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences #
        # Adding PAD token for labels #
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input #
        # adding PAD tokens to make the sentence same length as seq_len #
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):

        tokens = list(sentence)
        output_label = []
        output = []

        # 15% of the tokens would be replaced #
        for i, token in enumerate(tokens):

            prob = random.random()

            # 15% chance of altering token #
            if prob < 0.15 and self.is_train:
                prob /= 0.15

                # 80% chance change token to mask token #
                if prob < 0.8:
                    output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token #
                elif prob < 0.9:
                    output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token #
                else:
                    output.append(self.tokenizer(token)["input_ids"])

                output_label.append(self.tokenizer(token)["input_ids"])

            else:
                output.append(self.tokenizer(token)["input_ids"])
                output_label.append(self.tokenizer(token)["input_ids"])

        # Flattening #
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self, index):
        """
        Return random sentence pair
        """

        t1, t2 = self.get_corpus_line(index)

        # Negative or positive pair, for next sentence prediction #
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        """
        Return sentence pair
        """

        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        """
        Return random single sentence       
        """
        
        return self.lines[random.randrange(len(self.lines))][1]
    
# Creates embeddings/ associates values with the value, position, and #
# segment of each nucleotide in a sequence #

class PositionalEmbedding(torch.nn.Module):
    """
    Gives mathematical values to where and what each nucleotide in a
    sequence so that they can be mathematically computed
    """

    def __init__(self, d_model, max_len=1024):

        super().__init__()

        # Compute the positional encodings once in log space #
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pe = torch.zeros(max_len, d_model).float().to(self.device)
        pe.require_grad = False

        for pos in range(max_len):
            # For each dimension of the each position #
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # Include the batch size #
        self.pe = pe.unsqueeze(0)

    def forward(self, x):

        return self.pe
    
class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=1024, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size) #
        # padding_idx is not updated during training, remains as fixed pad (0) #
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0, device=self.device)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0, device=self.device)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label=None, is_train=False):
        x = self.token(sequence) + self.position(sequence)
        if segment_label is not None:
            x += self.segment(segment_label)

        if is_train == False:
            return x
        return self.dropout(x)
    
# Attention layers #

class MultiHeadedAttention(torch.nn.Module):
    """
    Creates a matrix where each row represents the word being queried and the
    column represents the word whose relationship value is being calculated so
    that words which are highly related have a high value, and words that are
    not related have a low value
    """

    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads == 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model, device=self.device)
        self.key = torch.nn.Linear(d_model, d_model, device=self.device)
        self.value = torch.nn.Linear(d_model, d_model, device=self.device)
        self.output_linear = torch.nn.Linear(d_model, d_model, device=self.device)

    def forward(self, query, key, value, mask, is_train=False):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model) #
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k) #
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len) #
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight #
        # (batch_size, h, max_len, max_len) #
        mask = mask.float()
        mask = (1 - mask) * -1e9 
        scores = scores + mask

        # (batch_size, h, max_len, max_len) #
        # softmax to put attention weight for all non-pad tokens #
        # max_len X max_len matrix of attention #
        weights = F.softmax(scores, dim=-1)
        if is_train == True:
            weights = self.dropout(weights) 

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k) #
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model) #
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model) #
        return self.output_linear(context)
    
class FeedForward(torch.nn.Module):
    """
    Implements FFN equation
    """

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):

        super(FeedForward, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fc1 = torch.nn.Linear(d_model, middle_dim, device=self.device)
        self.fc2 = torch.nn.Linear(middle_dim, d_model, device=self.device)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x, is_train=False):

        out = self.activation(self.fc1(x))
        if is_train == True:
            out = self.fc2(self.dropout(out))
        else:
            out = self.fc2(out)
        return out
    
class EncoderLayer(torch.nn.Module):
    """
    Uses multihead attention to calculate the relationship of each word in the
    input sequence with each other word
    """

    def __init__(
        self,
        d_model=768,
        heads=12,
        feed_forward_hidden=768 * 4,
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layernorm = torch.nn.LayerNorm(d_model, device = self.device)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask, is_train=False):
        """
        embeddings: (batch_size, max_len, d_model)
        encoder mask: (batch_size, 1, 1, max_len) 

        result: (batch_size, max_len, d_model)
        """
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # Residual layer #
        interacted = self.layernorm(interacted + embeddings)
        # Bottleneck #
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        
        if is_train == True:
            interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
            interacted = self.layernorm(interacted + embeddings)
            feed_forward_out = self.dropout(self.feed_forward(interacted))
        else: 
            interacted = self.self_multihead(embeddings, embeddings, embeddings, mask)
            interacted = self.layernorm(interacted + embeddings)
            feed_forward_out = self.feed_forward(interacted)
    
        encoded = self.layernorm(feed_forward_out + interacted)
        
        return encoded
    
# Combines each layer above to create predictions based on the values #
# calculated by each layer #
class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # Paper noted they used 4*hidden_size for ff_network_hidden_size #
        self.feed_forward_hidden = d_model * 4

        # Embedding for BERT, sum of positional, segment, token embeddings #
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model)

        # Multi-layers transformer blocks, deep network #
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info=None, is_train=False):

        # Attention masking for padded token #
        # (batch_size, 1, seq_len, seq_len) #
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # Embedding the indexed sequence to sequence of vectors #
        x = self.embedding(x, segment_info)

        # Running over multiple transformer blocks #
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
class NextSentencePrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.linear = torch.nn.Linear(hidden, 2, device=self.device)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, is_train=False):

        # Use only the first token which is the [CLS] #
        return self.softmax(self.linear(x[:, 0]))
    
class MaskedLanguageModel(torch.nn.Module):
    """
    Predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.linear = torch.nn.Linear(hidden, vocab_size, device=self.device)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, is_train=False):

        return self.softmax(self.linear(x))
    
class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label=None, is_train=False):
        
        x = self.bert(x, segment_label)
        if segment_label is not None:
            return self.next_sentence(x), self.mask_lm(x)
        else:
            return self.mask_lm(x)