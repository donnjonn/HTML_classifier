#General params
SEED = 2020 #Seed for consistent randomization
WEBSITE_NAME = '127.0.0.1'
WEB_ADDRESS = "http://127.0.0.1:8000" #Website to train/test network for

#Augmentation params
CSV_READ_AUGMENT = 'data.csv'
CSV_WRITE = 'data_augment_nieuw.tsv'
AUG_AMOUNT = 200

#Get attributes
TSV_NEW = "attributes_new.tsv" #Where to save attributes

#Word2vec params
EMBED_SIZE = 300 # how big is each word vector
MAX_FEATURES = 120000 # how many unique words to use (i.e num rows in embedding vector)
MAX_LEN = 750 # max number of words in a question to use

#Training params
TSV_READ_DATA = 'data_augment.tsv'
LR = 0.001 #Learning Rate
BATCH_SIZE = 256 # how many samples to process at once
N = 5 # how many times to iterate over all samples
SAVE_MODEL_CNN = 'textcnn_model_new.pt'
SAVE_DICT_CNN = 'textcnn_dict_new.pt'
SAVE_MODEL_LSTM = 'bilstm_model_new.pt'
SAVE_DICT_LSTM = 'bilstm_dict_new.pt'

#Testing params
LOAD_MODEL_NAME = "textcnn_model.pt" #Model to be tested
TEST_XPATH = "//a[text()='Geendress']"
SEARCH_ELEMENT = "Laptoplink"

#CNN params
FILTER_SIZES = [1,2,3,5]
NUM_FILTERS = 36

#BiLSTM params
DROPOUT = 0.1
HIDDEN_SIZE = 64

# N_SPLITS = 5 # Number of K-fold Splits
# debug = 0
