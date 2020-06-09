import random
import time
import pandas as pd
import numpy as np
import torch
import sys
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch.onnx
import argparse
from config import *
import config as cfg
#from test_ui import *
tqdm.pandas(desc='Progress')
#constants
progress = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN_Text(nn.Module):
    def __init__(self, le, embedding_matrix):
        super(CNN_Text, self).__init__()
        filter_sizes = FILTER_SIZES
        num_filters = NUM_FILTERS
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(MAX_FEATURES, EMBED_SIZE)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, EMBED_SIZE)) for K in filter_sizes])
        self.dropout = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, n_classes)
        
    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x) 
        return logit
       
       
class BiLSTM(nn.Module):
    def __init__(self, le, embedding_matrix):
        super(BiLSTM, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        drp = DROPOUT
        n_classes = len(le.classes_)
        self.embedding = nn.Embedding(MAX_FEATURES, EMBED_SIZE)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(EMBED_SIZE, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


def load_glove(word_index):
    EMBEDDING_FILE = 'glove.6B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="utf8"))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]
    nb_words = min(MAX_FEATURES, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def plot_graph(epochs, train_loss, valid_loss):
    
    fig = plt.figure(figsize=(12,12))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation loss')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.show()


def train_nn(n_epochs, model, train_X, train_y, test_X, test_y, le, action):
    import config as cfg
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    model.to(device)
    # Load train and test in CUDA Memory
    x_train = torch.tensor(train_X, dtype=torch.long).to(device)
    y_train = torch.tensor(train_y, dtype=torch.long).to(device)
    x_cv = torch.tensor(test_X, dtype=torch.long).to(device)
    y_cv = torch.tensor(test_y, dtype=torch.long).to(device)
    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)
    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=cfg.BATCH_SIZE, shuffle=False)
    train_loss = []
    valid_loss = []
    valid_acc = []
    for epoch in range(n_epochs):
        
        start_time = time.time()
        # Set model to train configuration
        model.train()
        avg_loss = 0.  
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Predict/Forward Pass
            #print(x_batch.shape)
           # print(x_batch[0].shape)
            #print(x_batch[0][0])
            y_pred = model(x_batch)
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        # Set model to validation configuration -Doesn't get trained here
        model.eval()        
        avg_val_loss = 0.
        val_preds = np.zeros((len(x_cv),len(le.classes_)))
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            # keep/store predictions
            val_preds[i * BATCH_SIZE:(i+1) * BATCH_SIZE] =F.softmax(y_pred).cpu().numpy()
        # Check Accuracy
        val_accuracy = sum(val_preds.argmax(axis=1)==test_y)/len(test_y)
        train_loss.append(avg_loss)
        valid_loss.append(avg_val_loss)
        valid_acc.append(val_accuracy)
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(
                    epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy, elapsed_time))  
        global progress
        progress += 1
        
    model.eval()
    if action.action=='cnn' or not len(sys.argv)>1:
        torch.save(model,SAVE_MODEL_CNN)
        torch.save(model.state_dict(), SAVE_DICT_CNN)
    elif action.action=='lstm':
        torch.save(model,SAVE_MODEL_LSTM)
        torch.save(model.state_dict(), SAVE_DICT_LSTM)
    plot_graph(n_epochs, train_loss, valid_loss)


def prep_tokenizer(train_X):
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_X))
    return tokenizer

def prep_data():
    
    import config as cfg
    
    
    print(cfg.TSV_READ_DATA)
    data = pd.read_csv(cfg.TSV_READ_DATA, delimiter="\t")
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    data = pd.concat([train,test])[['type','element','nummer']]
    print(data)
    data['len'] = data['element'].apply(lambda s : len(s))
    #data['len'].plot.hist(bins=100)
    #plt.show()
    print(data.len.quantile(0.9))
    count_df = data[['type','element']].groupby('type').aggregate({'element':'count'}).reset_index().sort_values('element',ascending=False)
    print(count_df)
    target_conditions = count_df[count_df['element']>200]['type'].values
    def condition_parser(x):
        if x in target_conditions:
            return x
        else:
            return "OTHER"     
    data['type'] = data['type'].apply(lambda x: condition_parser(x))
    #bar = px.bar(count_df[count_df['element']>200],x='nummer',y='element')
    #bar.show() 
    train_X, test_X, train_y, test_y = train_test_split(data['element'], data['type'], stratify=data['type'], test_size=0.25)                                          
    print("Train shape : ",train_X.shape)
    print("Test shape : ",test_X.shape)
    tokenizer = prep_tokenizer(train_X)
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    train_X = pad_sequences(train_X, maxlen=cfg.MAX_LEN)
    test_X = pad_sequences(test_X, maxlen=cfg.MAX_LEN)
    le = LabelEncoder()
    train_y = le.fit_transform(train_y.values)
    test_y = le.transform(test_y.values)
    print(le.classes_)
    embedding_matrix = load_glove(tokenizer.word_index)
    return tokenizer, le, train_X, test_X, train_y, test_y, embedding_matrix
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', dest='action')
    tokenizer, le, train_X, test_X, train_y, test_y, embedding_matrix = prep_data()
    action = parser.parse_args()
    if action.action=='cnn' or not len(sys.argv)>1:
        print('CNN trainen\n')
        model = CNN_Text(le, embedding_matrix)
    elif action.action=='lstm':
        print('BiLSTM trainen\n')
        model = BiLSTM(le, embedding_matrix)
    else:
        print('Fout argument, gebruik "cnn" of "lstm"')
        sys.exit()
    train_nn(N, model, train_X, train_y, test_X, test_y, le, action)
    

if __name__ == "__main__":
    main()