from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv 
from config import *
driver = webdriver.Firefox()
driver.implicitly_wait(5)
driver.maximize_window()
name = WEBSITE_NAME
driver.get(WEB_ADDRESS)
#driver.manage().timeouts().pageLoadTimeout(10, TimeUnit.SECONDS);
elements1 = driver.find_elements_by_xpath("//a")
elements2 = driver.find_elements_by_xpath("//input")
elements = elements1 + elements2
print(elements)
attrs_list = []
counter = 0
attr_dict = {}
def attrs_func(elements, driver, counter):
    old_url = driver.current_url
    for e in elements:
        print(e.get_attribute('outerHTML'))
        o = e.get_attribute('outerHTML')
        tag = o.split('>',1)[0]
        print(tag)
        attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', e)
        attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', e)
        attrs = attrs + attrs2
        if attrs not in attrs_list:
            attr_dict[counter] = attrs
            counter  += 1
            attrs_list.append(attrs)
            #print(attrs)
            print(e.get_attribute("type"))
            if '<a' in tag:
                print('click')
                e.click()
                element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//a")))
            if name in driver.current_url:
                elements1 = driver.find_elements_by_xpath("//a")
                elements2 = driver.find_elements_by_xpath("//input")
                els_new = elements1 + elements2
                try:
                    attrs_func(els_new, driver, counter)
                except:
                    break
            else:
                driver.back()

    
attrs_func(elements, driver, counter)
csv_columns = ['element', 'attributes']
csv_file = TSV_NEW
try:
    with open(csv_file, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
        for key, value in attr_dict.items():
            writer.writerow([key, value])
except IOError:
    print("I/O error\n\n\n")