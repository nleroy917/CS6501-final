# CS6501 Final Project
This is the final project for UVA's graduate level NLP course Fall 2023. The project is called "NLP for social good." The goal is to apply NLP techniques to a problem that has a positive social impact. Our group descided to build a DNA sequence classifier that can predict the function of a DNA sequence. The classifier is trained on a dataset of DNA sequences and their corresponding proteomic functions. The dataset derived [from kaggle](https://www.kaggle.com/code/singhakash/dna-sequencing-with-machine-learning).

## Get the data
We've provided a script to download the data. Run the following command to download the data to the `data` directory:
```console
sh get_data.sh
```
The data will be in three separate files: `human.txt`, `chimpanzee.txt`, and `dog.txt`. Each file contains a list of DNA sequences with their corresponding proteomic functions as an integer label. The data is formatted as follows:
```
ATGTGTACAGTCGA 1
ATGTGTACAGTCGA 2
ATGCTAGTGTGACA 1
...
```

## Load a pretrained model
We've uploaded our pre-trained models to huggingface for easy use. To load a model, use the following code:
```python
from models.dna_classifier import DNASequenceClassifier

model = DNASequenceClassifier("nleroy917/cs6501-final-project")

seq = "ATGTGTACAGTCGA"
label = model.predict(seq)
```