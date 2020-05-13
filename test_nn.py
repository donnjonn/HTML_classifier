from train import *
import torch
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def load_attributes(xpath):
    driver = webdriver.Firefox()
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.get("http://127.0.0.1:8000")
    el = driver.find_element_by_xpath(xpath)
    attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', el)
    attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', el)
    attrs = attrs + attrs2
    driver.quit()
    return attrs

def predict_single(x, model, tokenizer, le):    
    # lower the text
    x = x.lower()
    x = tokenizer.texts_to_sequences([x])
    # pad
    x = pad_sequences(x, maxlen=maxlen)
    # create dataset
    x = torch.tensor(x, dtype=torch.long).cuda()
    pred = model(x).detach()
    pred = F.softmax(pred).cpu().numpy()
    pred_chance = pred.max()
    pred_classes = pred.argmax(axis=1)

    pred_class = le.classes_[pred_classes]
    return pred_chance, pred[0], pred_class[0]
    
def search_element(type, tokenizer, le, model):
    max_prob = 0.0
    driver = webdriver.Firefox()
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.get("http://127.0.0.1:8000")
    elements1 = driver.find_elements_by_xpath("//a")
    elements2 = driver.find_elements_by_xpath("//input")
    elements = elements1 + elements2
    for e in elements:
        attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', e)
        attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', e)
        attrstext = attrs + attrs2
        attrs = tokenizer.texts_to_sequences([attrstext])
        attrs = pad_sequences(attrs, maxlen=maxlen)
        attrs = torch.tensor(attrs, dtype=torch.long).cuda()
        index = np.where(le.classes_ == type)
        pred = model(attrs).detach()
        pred = F.softmax(pred).cpu().numpy()
        print(float(pred[0][index][0]))
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
    # model = CNN_Text()
    # model.load_state_dict(torch.load('textcnn_dict.pt'))
    model = torch.load('textcnn_model.pt')
    model.eval()
    model.cuda()
    tokenizer, le, train_X, test_X, train_y, test_y, embedding_matrix = prep_data()
    #dummy_input = torch.randn(512, 750)
    #torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")
    attrs = load_attributes("//a[text()='Geendress']")
    #attrs = data['element'].values[20]
    #x = data['element'].values[20]
    print(attrs)
    print(predict_single(attrs, model, tokenizer, le))
    search_element('Laptoplink', tokenizer, le, model)
    
def main():
    test()

if __name__ == "__main__":
    main()