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
pip install -r requirements.txt
```

Make migrations:
```
python manage.py makemigrations --merge
python manage.py migrate
```

Start the server:
```
python manage.py runserver
```
The site should now be accessible at 127.0.0.1:8000

### Use GUI
Run
```
python test_ui.py
```
Her you will be able to use the full set of functionalities included in this repository. 

### Use commandline
#### Get attributes of new site

If you want to extract attributes from your own site. Run the following command:
```
python getattrs_any.py
```
(Make sure to use the correct URL in config.py).<br/>
Note: this doesn't work perfectly yet. <br/>
If you want a better dataset use the site included under the folder 'Onlineshop'.<br/>
data_augment.tsv is a dataset generated using this site. (See 'Setup site', this dataset is already augmented)

Afterwards to make a usable dataset out of this data, run:
```
python augment_data.py
```
This will generate a tsv file with augmented data, which can be used to train your network.

#### Build vocabulary
First:
```
python -m spacy download en_core_web_sm
```

Run
```
python build_vector.py
```
This will build a vocabulary based on the attribute data of a certain website (This doesn't have to be the site you want to test, dataset can be chosen in config.py).<br/>
This might take a while! <br/>
When this is done cut and paste 'glove.6B.300d.txt' from .vector_cache to the main folder.

Or you can download a pretrained vocabulary here: https://www.dropbox.com/s/5c17s95bsyg02zy/vocabulary.zip?dl=0

#### Train the neural network

Here there are 2 options:

##### CNN

run
```
python train.py -n cnn
```
This will start training for a convolutional neural network for n epochs (ths number can be chosen in config.py)

##### BiLSTM

run
```
python train.py -n lstm
```
This will start training for a Bidirectional LSTM network for n epochs (ths number can be chosen in config.py)
Keep in mind that training of an LSTM is slower so it might be necessary to raise the amount of epochs.

#### Test a neural network

First make sure to select the neural network you want to test in config.py. Make sure to change the elements you want to test according to the website the network was trained for. (also in config.py)
Then run:
```
python test_nn.py
```
This will let the neural network predict the type of an element you selected in config.py. <br/>
Then this test will try to let the neural network pick an element which has the highest probability of being of the type you chose in config.py.

### How to use for testing purposes (pipeline)
This software can be used in testing when following the following steps: <br/>
1. Extract the elements of the site you want to run (using the included crawler or write a better one for the specific site). This shuold be done regularly to avoid elements not being recognized by the neural network.
2. Write working selenium test(s) for the site to be tested.
3. While running test --> Feed each element  to the neural network --> save the ID returned by the NN for each element
4. When a change has been applied to the website build, this may cause the test to return an error.
5. When this happens, loop over page where the test gets stuck, feed each element to the NN and let it return the probability an element is the element which is needed for the test to continue (using the saved ID). If there's one or more elements with a probability higher than a certain threshold, let the tester choose which element to continue with. If an element is not recognzed by the NN (probability of any element is below certain threshold), add it to the data, augment it, and retrain the network. When no element has a high enough probability of being the required element, debugging manually will be necessary. 
