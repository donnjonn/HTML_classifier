from torchtext import data
from torchtext.vocab import Vectors, GloVe
import spacy
from torch.autograd import Variable
import torch
import random 
from config import *

def build_vector():
    TEXT = data.Field(sequential=True,tokenize='spacy',tokenizer_language="en_core_web_sm",batch_first=True,include_lengths=True)
    LABEL = data.LabelField(dtype = torch.float,batch_first=True, sequential=False, use_vocab=False)
    fields = [('label', LABEL), ('text',TEXT), (None, None)]
    training_data=data.TabularDataset(path = TSV_READ_DATA,format = 'tsv',fields = fields, skip_header = False)
    train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))
    TEXT.build_vocab(train_data,min_freq=1,vectors=GloVe(name='6B', dim=300))  

def main():
    build_vector()

if __name__ == "__main__":
    main()