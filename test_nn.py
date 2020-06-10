from train import *
from train import prep_data
import torch
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from config import *
import config as cfg
import importlib
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_attributes(xpath):
    import config as cfg
    driver = webdriver.Firefox()
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.get(cfg.WEB_ADDRESS)
    el = driver.find_element_by_xpath(xpath)
    attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', el)
    attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', el)
    attrs = attrs + attrs2
    driver.quit()
    return attrs

def predict_single(x, model, tokenizer, le):
    import config as cfg
    # lower the text
    x = x.lower()
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=cfg.MAX_LEN)
    # create dataset
    x = torch.tensor(x, dtype=torch.long).to(device)
    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()
    pred_chance = pred.max()
    pred_classes = pred.argmax(axis=1)

    pred_class = le.classes_[pred_classes]
    return pred_chance, pred[0], pred_class[0]
    
def search_element(type, tokenizer, le, model):
    import config as cfg
    max_prob = 0.0
    driver = webdriver.Firefox()
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.get(cfg.WEB_ADDRESS)
    elements1 = driver.find_elements_by_xpath("//a")
    elements2 = driver.find_elements_by_xpath("//input")
    elements = elements1 + elements2
    for e in elements:
        attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', e)
        attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', e)
        attrstext = attrs + attrs2
        attrs = tokenizer.texts_to_sequences([attrstext])
        attrs = pad_sequences(attrs, maxlen=cfg.MAX_LEN)
        attrs = torch.tensor(attrs, dtype=torch.long).to(device)
        index = np.where(np.array(le.classes_) == type)
        
        pred = model(attrs).detach()
        pred = F.softmax(pred).cpu().numpy()
        print(pred)
        print(le.classes_)
        print(type)
        print(index)
        print(pred[0][index])
        #print(float(pred[0][index][0]))
        if float(pred[0][index][0]) > max_prob:
            print(pred[0][index][0], '>', max_prob)
            max_prob = pred[0][index][0]
            best_el = e
        print('kans op', type, 'is:', pred[0][index], 'met attrs:\n', attrstext, '\n')
    def apply_style(s):
        driver.execute_script("arguments[0].setAttribute('style', arguments[1]);", best_el, s)
    print(best_el)
    original_style = best_el.get_attribute('style')
    apply_style("background: red; border: 10px solid red;")

def test():
    import config as cfg
    importlib.reload(cfg)
    # model = CNN_Text()
    # model.load_state_dict(torch.load('textcnn_dict.pt'))
    model = torch.load(cfg.LOAD_MODEL_NAME)
    model.eval()
    model.to(device)
    tokenizer, le, train_X, test_X, train_y, test_y, embedding_matrix = prep_data()
    #dummy_input = torch.randn(512, 750)
    #torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")
    attrs = load_attributes(cfg.TEST_XPATH)
    #attrs = data['element'].values[20]
    #x = data['element'].values[20]
    print(attrs)
    print(predict_single(attrs, model, tokenizer, le))
    search_element(cfg.SEARCH_ELEMENT, tokenizer, le, model)
    
def main():
    test()

if __name__ == "__main__":
    main()