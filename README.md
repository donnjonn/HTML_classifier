# HTML_classifier
Classify html elements using neural networks.

## Prerequisites
### 1) Install Conda & setup virtual environment
#### Windows
Follow instructions here:<br/>
https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html<br/>
Open a conda terminal (Use this as your main terminal window) and create a fresh virtual environment (python 3.6).

#### Mac OS
Follow instructions here:<br/>
https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html<br/>
Open a conda terminal (Use this as your main terminal window) and create a fresh virtual environment (python 3.6).

### 2) Install pip
Follow instructions here:<br/>
https://www.liquidweb.com/kb/install-pip-windows/<br/>
(Idem voor Mac)

####

### 3) Install pytorch
Inside conda terminal install pytorch using the conda install command found here:<br/>
https://pytorch.org/get-started/locally/<br/>
For most modern systems following command will work:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
### 4) Install other dependencies
Inside this project folder use following command:
```
pip install -r requirements.txt
```

## Usage
### Configuration
Most parameters can be changed in config.py

### Setup site
Link to github-repo: https://github.com/amitkadivar1/OnlineShop.git<br/>
This was the site I used to train and test  the neural network.<br/>
The included dataset and networks are based on this website.<br/><br/>
Navigate to the 'Onlineshop' folder. <br/>
Install dependencies:
```
pip install requirements.txt
```

Make migrations:
```
python manage.py migrate
```

Start the server:
```
python manage.py runserver
```
The site should now be accessible at 127.0.0.1:8000


### Build vocabulary
First:
```
python -m spacy download en_core_web_sm
```

Run
```
python build_vector.py
```
This will build a vocabulary based on the attribute data of a certain website (This doesn't have to be the site you want to test).<br/>
Or you can download a pretrained vocabulary here:

### Get attributes of new site

If you want to extract attributes from your own site. Run the following command:
```
python getattrs_any.py
```
(Make sure to use the correct URL in config.py).<br/>
Note: this doesn't work perfectly yet. <br/>
If you want a better dataset use the site included under the folder 'Onlineshop'.<br/>
data_augment.tsv is a dataset generated using this site. (See 'Setup site')

Afterwards to make a usable dataset out of this data, run:
```
python augment_data.py
```
This will generate a tsv file. with augmented data, which can be used to train your network.

### Train the neural network

Here there are 2 options:

#### CNN

run
```
python train.py -n cnn
```
This will start training for a convolutional neural network for n epochs (ths number can be chosen in config.py)

#### BiLSTM

run
```
python train.py -n lstm
```
This will start training for a Bidirectional LSTM network for n epochs (ths number can be chosen in config.py)
Keep in mind that training of an LSTM is slower so it might be necessary to raise the amount of epochs.

### Test a neural network

First make sure to select the neural network you want to test in config.py. Make sure to change the elements you want to test according to the website the network was trained for. (also in config.py)
Then run:
```
python test_nn.py
```
This will let the neural network predict the type of an element you selected in config.py. <br/>
Then this test will try to let the neural network pick an element which has the highest probability of being of the type you chose in config.py.


