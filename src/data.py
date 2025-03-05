#!/usr/bin/env python

"""
data.py creates the data files required for this project
"""

# Imports #
import os
import json
from Bio import SeqIO

# Together, these two functions take a large fasta file and create #
# strings of the nucleotide data contained within into seperate files for #
# each speciemen of malaria included in the file #

# RUNNING THESE WILL CREATE FILES #

def create_helperdata(chunk_size):
    """
    TO ADD
    """

    def _helper():
        """
        Helper function for create_helperdata
        """
        
        filename = 'data/Plasmodium_falciparum_3D7_Genome.fasta'

        # Parses file with genome into dictionary format #
        record_dict = SeqIO.to_dict(SeqIO.parse(filename, 'fasta'))
        for key in record_dict.keys():
            yield record_dict[key].seq, key

    for sequence, name in iter(_helper()):
        with open(f'data/{name}.txt', 'w') as f:
            chunk = len(sequence) // chunk_size
            for i in range(chunk):
                indx = i*chunk_size
                chunk = sequence[indx:indx+chunk_size]
                f.write(f'{chunk}\n')

# Each file contains many instances of strings of #
# size=chunk_size (default 512), these functions pairs each chunk #
# with it's neighbors #

def create_pairedData():
    """
    TO ADD
    """

    def create_onepairs(filename):
        """
        Helper function for create_pairedData
        """

        with open(filename) as f:
            lines = [line.rstrip('\n') for line in f]
        lines = list(zip(lines[:-1], lines[1:]))
        return lines

    # Getting all filenames #
    filenames = []
    for filename in os.listdir('data'):
        if '.txt' in filename:
            # Paired data file already created #
            if filename == 'paired_data.txt':
                continue
            f = 'data/' + filename
            filenames.append(f)

    # Creating pairs #
    pairs = []
    for temp_file in filenames:
        pair = create_onepairs(temp_file)
        pairs = pairs + pair
    
    # Writing file #
    with open('data/paired_data.txt', 'w') as f:
        f.write(json.dumps(pairs))

    with open('data/paired_data.txt', 'r') as f:
        a = json.loads(f.read())