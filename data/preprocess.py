import random
from Bio import SeqIO

# set seed
random.seed(42)

# get raw seqs
with open("fluB.fasta") as f:
    fluB = list(SeqIO.parse(f, "fasta"))

with open("fluA.fasta") as f:
    fluA = list(SeqIO.parse(f, "fasta"))

# annotate, merge, shuffle, reprocess
fluB = [(str(seq.seq), 'B') for seq in fluB]
fluA = [(str(seq.seq), 'A') for seq in fluA]

all_seqs = fluB + fluA
random.shuffle(all_seqs)

# write to file
with open("flu.txt", "w") as f:
    for seq in all_seqs:
        _ = f.write('\t'.join(seq) + '\n')
