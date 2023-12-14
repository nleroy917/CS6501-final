# CS6501 Final Project
This is the final project for UVA's graduate level NLP course Fall 2023. The project is called "NLP for social good." The goal is to apply NLP techniques to a problem that has a positive social impact. Our group descided to build a DNA/Protein sequence classifier that can predict the viral source given a protein sequence. The classifier is trained on a dataset of DNA/protein sequences and their corresponding viral origins. The dataset derived [from the NCBI virus data portal](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/).

## Get the data
We've provided a script to download the data. Run the following command to download the data to the `data` directory:
```console
sh get_data.sh
```
The data will be in two separate files: `covid.fasta` and `flu.fasta`. Each file contains a list of protein sequences in [FASTA format](https://en.wikipedia.org/wiki/FASTA_format). The data is formatted as follows.

We've provided a preprocess script that will convert the data into a format that can be used by the model. Run the following command to preprocess the data:
```console
cd data
python preprocess.py
```

## Setup
To begin, install the dependencies:
```console
pip install -r requirements.txt
```

## Train a model
Included in the repository is a notebook that will enable you to train a model on the data. The notebook is called `train.ipynb`. You can run the notebook in Google Colab or on your local machine.

## Load pretrained model
Models can be exported using the `model.export()` method. The exported model can be loaded again by using the `from_pretrained` method. For example:
```python
from dna_classification.models import DNASequenceClassifier

model = DNASequenceClassifier("nleroy917/viral-sequence-prediction")

virus = model.predict("MGYINVFAFPFTIYSLLLCRMNFRNYIAQVDVVNFNLT")

print(virus) # covid
```

