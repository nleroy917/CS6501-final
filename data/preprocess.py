import random
from Bio import SeqIO

# set seed
random.seed(42)

# get raw seqs
with open("flu.fasta") as f:
    flu = list(SeqIO.parse(f, "fasta"))

with open("covid.fasta") as f:
    covid = list(SeqIO.parse(f, "fasta"))

# annotate, merge, shuffle, reprocess
flu = [(str(seq.seq), 'flu') for seq in flu]
covid = [(str(seq.seq), 'covid') for seq in covid]

all_seqs = flu + covid
random.shuffle(all_seqs)

# write to file
with open("virus.txt", "w") as f:
    for seq in all_seqs:
        _ = f.write('\t'.join(seq) + '\n')
