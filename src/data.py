#!/usr/bin/env python

"""
data.py cleans the data required for this project
"""

import os
import pandas as pd
from Bio import SeqIO

# Together, these two functions take a large fasta file and create strings of the nucleotide data contained #
# within into seperate files for each speciemen of malaria included in the file #

# RUNNING THESE WILL CREATE FILES #

def _helper():
    """
    Helper function for create_helperdata
    """
    
    filename = 'data/Plasmodium_falciparum_3D7_Genome.fasta'

    # Parses file with genome into dictionary format #
    record_dict = SeqIO.to_dict(SeqIO.parse(filename, 'fasta'))
    for key in record_dict.keys():
        yield record_dict[key].seq, key

def create_helperdata(chunk_size):

    for sequence, name in iter(_helper()):
        with open(f'data/{name}.txt', 'w') as f:
            chunk = len(sequence) // chunk_size
            for i in range(chunk):
                indx = i*chunk_size
                chunk = sequence[indx:indx+chunk_size]
                f.write(f'{chunk}\n')