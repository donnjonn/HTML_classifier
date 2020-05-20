from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *
import csv 
from config import *
driver = webdriver.Firefox()
driver.implicitly_wait(5)
driver.maximize_window()
name = WEBSITE_NAME
driver.get(WEB_ADDRESS)
#driver.manage().timeouts().pageLoadTimeout(10, TimeUnit.SECONDS);
# elements1 = driver.find_elements_by_xpath("//a")
# elements2 = driver.find_elements_by_xpath("//input")
# elements = elements1 + elements2
# print(elements)
counter = 0
attr_dict = {}
visited = []
def get_attrs(driver):
    attrs_list = []
    old_url = driver.current_url
    elements1_len = len(WebDriverWait(driver, 1).until(EC.presence_of_all_elements_located((By.XPATH, "//a"))))
    
    elements2 = driver.find_elements_by_xpath("//input")
    #elements = elements1 + elements2
    visited.append(driver.current_url)
    for i in range(elements1_len):
        print(old_url)
        print(driver.current_url)
        if driver.current_url != old_url:
            driver.get(old_url)
        print('max:',elements1_len)
        print(WebDriverWait(driver, 1).until(EC.presence_of_all_elements_located((By.XPATH, "//a"))))
        print(i)
        
        e = WebDriverWait(driver, 1).until(EC.presence_of_all_elements_located((By.XPATH, "//a")))[i]
        #e = driver.find_elements_by_xpath("//a")[i]
        print(e.get_attribute('outerHTML'))
        attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', e)
        attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', e)
        attrs = attrs + attrs2
        if attrs not in attrs_list:
            print('nieuw')
            attrs_list.append(attrs)
            xpath = driver.execute_script("gPt=function(c){if(c.id!==''){return'id(\"'+c.id+'\")'}if(c===document.body){return c.tagName}var a=0;var e=c.parentNode.childNodes;for(var b=0;b<e.length;b++){var d=e[b];if(d===c){return gPt(c.parentNode)+'/'+c.tagName+'['+(a+1)+']'}if(d.nodeType===1&&d.tagName===c.tagName){a++}}};return gPt(arguments[0]).toLowerCase();", e);
            attr_dict[driver.current_url+xpath] = attrs
            try:
                e.click()
                print(driver.current_url)
                print(name)
                if name in driver.current_url and driver.current_url not in visited:
                    print('JAJAJAJAJAJAJJAJAJAJAJJAJAJAJA')
                    get_attrs(driver)
                else:
                    print('back')
                    driver.back()
            except ElementNotInteractableException:
                print('not interactable')
                continue
    
                

# def attrs_func(elements, driver, counter):
    # old_url = driver.current_url
    # for e in elements:
        # print(e.get_attribute('outerHTML'))
        # o = e.get_attribute('outerHTML')
        # tag = o.split('>',1)[0]
        # print(tag)
        # attrs = driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', e)
        # attrs2 = driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', e)
        # attrs = attrs + attrs2
        # if attrs not in attrs_list:
            # attr_dict[counter] = attrs
            # counter  += 1
            # attrs_list.append(attrs)
            # #print(attrs)
            # print(e.get_attribute("type"))
            # if '<a' in tag:
                # print('click')
                # e.click()
                # element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//a")))
            # if name in driver.current_url:
                # elements1 = driver.find_elements_by_xpath("//a")
                # elements2 = driver.find_elements_by_xpath("//input")
                # els_new = elements1 + elements2
                # try:
                    # attrs_func(els_new, driver, counter)
                # except:
                    # break
            # else:
                # driver.back()

get_attrs(driver)
csv_columns = ['element', 'attributes']
csv_file = TSV_NEW
try:
    with open(csv_file, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
        for key, value in attr_dict.items():
            writer.writerow([key, value])
except IOError:
    print("I/O error\n\n\n")