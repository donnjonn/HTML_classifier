from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import *
import csv 
from config import *


name = WEBSITE_NAME
#visited = []
linkAlreadyVisited = []
attr_dict = {}
attr_list = []
class RecursiveLinkTest:
    #list to save visited links

    def __init__(self, driver, name, attr_list, attr_dict):
        self.driver = driver
        self.name = name
        self.attr_list  = attr_list
        self.attr_dict = attr_dict
        
    def linkTest(self, counter):
        #loop over all the a elements in the page
        if self.name in self.driver.current_url:
            print(name)
            old_url = self.driver.current_url
            print(self.driver.current_url)
            for i in range(len(self.driver.find_elements_by_tag_name("a"))):
                print(old_url)
                print(self.driver.current_url)
                if old_url != self.driver.current_url:
                    self.driver.get(old_url)
                #print(self.driver.find_elements_by_tag_name("a"))
                #print(i)
                link = self.driver.find_elements_by_tag_name("a")[i]
                try:
                    attrs = self.driver.execute_script('var items = ""; for (index = 0; index < arguments[0].attributes.length; ++index) {items+= arguments[0].attributes[index].value; items+=" ";}; return items;', link)
                    attrs2 = self.driver.execute_script('var items = ""; var o = getComputedStyle(arguments[0]); for (index = 0; index < o.length; index++) {items+=o.getPropertyValue(o[index]); items+=" ";}; return items;', link)
                    attrs = attrs + attrs2
                
                except StaleElementReferenceException:
                    continue
                if attrs not in self.attr_list:
                    self.attr_list.append(attrs)
                    print('length:',len(self.attr_list))
                # Check if link is displayed and not previously visited
                if link.is_displayed():
                    # add link to list of links already visited
                    
                    print(link.text)
                    txt = link.text
                    link.click()
                    if txt not in linkAlreadyVisited:
                        linkAlreadyVisited.append(txt)
                        rcl = RecursiveLinkTest(self.driver, self.name, self.attr_list, self.attr_dict)
                        self.attr_list = rcl.linkTest(counter)
                    self.driver.back()
        return self.attr_list
            

def extract_el():
    import config as cfg
    importlib.reload(cfg)
    counter = 0
    attr_dict = {}
    visited = []
    forms_filledin = ''
    driver = webdriver.Firefox()
    driver.implicitly_wait(5)
    driver.get(cfg.WEB_ADDRESS)
    rcl = RecursiveLinkTest(driver,name, [], {})
    attr_list = rcl.linkTest(0)
    print(attr_dict)
    driver.close()
    print('length:',len(attr_list))
    for i in  range(len(attr_list)):
        attr_dict[i] = attr_list[i]
        

    csv_columns = ['element', 'attributes']
    csv_file = cfg.TSV_NEW
    try:
        with open(csv_file, 'w', newline="") as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter='\t')
            writer.writerow(['type', 'element', 'nummer'])
            for key, value in attr_dict.items():
                writer.writerow([key, value])
    except IOError:
        print("I/O error\n\n\n")


def main():
    extract_el()

if __name__ == "__main__":
    main()






