import kivy
 
from kivy.app import App
from kivy.config import Config
Config.set('graphics', 'width', '1820')
Config.set('graphics', 'height', '980')
Config.write()
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.app import App
from config import *
import config as cfg
from train import *
from getattrs_any import *
from kivy.properties import StringProperty, ObjectProperty, NumericProperty
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from kivy.core.text import Label as CoreLabel
from kivy.uix.scrollview import ScrollView
from multiprocessing import Process
from test_nn import *
from augment_data import *
from kivy.graphics import *
import os
 
popup_open = False 
class MyBrowser(Popup):
 
    def __init__(self, headertext, labeltext, train, **kwargs):
        super(MyBrowser, self).__init__(**kwargs)
        self.title=headertext
        container = BoxLayout(orientation='vertical')
        l = Label(text=labeltext, size_hint=(1, .1),font_size='20sp')
        filechooser = FileChooserListView()
        filechooser.bind(on_selection=lambda x: self.selected(filechooser.selection))
        open_btn = Button(text='Select', size_hint=(1, .2))
        open_btn.bind(on_release=lambda x: self.openfile(filechooser.path, filechooser.selection, train))
        
    
        container.add_widget(l)
        container.add_widget(filechooser)
        container.add_widget(open_btn)
        
        self.add_widget(container)
 
    def openfile(self, path, filename, type):
        if type=="train":
            print(filename)
            if len(filename) > 0:
                cfg.TSV_READ_DATA = filename[0]
                print(cfg.TSV_READ_DATA)
                # with open(os.path.join(path, filename[0])) as f:
                    # print(f.read()) 
            self.dismiss()
            popup_open = False
        elif type=="test":
            print(filename)
            if len(filename) > 0:
                cfg.LOAD_MODEL_NAME = filename[0]
                print(cfg.LOAD_MODEL_NAME)
                # with open(os.path.join(path, filename[0])) as f:
                    # print(f.read()) 
            self.dismiss()
            popup_open = False
        
        elif type=="extract":
            print(filename)
            if len(filename) > 0:
                cfg.TSV_NEW = filename[0]
                print(cfg.TSV_NEW)
                # with open(os.path.join(path, filename[0])) as f:
                    # print(f.read()) 
            self.dismiss()
            popup_open = False
        
        elif type=="augload":
            print(filename)
            if len(filename) > 0:
                cfg.TSV_NEW = filename[0]
                print(cfg.TSV_NEW)
                # with open(os.path.join(path, filename[0])) as f:
                    # print(f.read()) 
            self.dismiss()
            popup_open = False
        
        elif type=="augwrite":
            print(filename)
            if len(filename) > 0:
                cfg.CSV_WRITE = filename[0]
                print(cfg.TSV_NEW)
                # with open(os.path.join(path, filename[0])) as f:
                    # print(f.read()) 
            self.dismiss()
            popup_open = False
        
    def selected(self, filename):
        print("selected: %s" % filename[0])
        
class network():
    def __init__(self, type):
        self.action = type


 
class MyConfig(Popup):
    def __init__(self, **kwargs):
        super(MyConfig, self).__init__(**kwargs)
        container = BoxLayout(orientation='vertical')
        self.dict = {}
        self.counter = 0
        self.key_list = []
        with open("config.py", "r") as a_file:
            
            for line in a_file:
                stripped_line = line.strip()
                if not (stripped_line.startswith('#') or stripped_line == ""):
                    print('stripped_line:',stripped_line)
                    list = stripped_line.split(" = ")
                    value = list[1].split("#")[0]
                    key = list[0]
                    self.key_list.append(key)
                    print(len(self.key_list))
                    print(self.counter)
                    
                    self.dict[key] = value
                    subcont = BoxLayout(orientation='horizontal')
                    l = Label(text=key, size_hint=(.5, 1))
                    ti = TextInput(text=value, size_hint=(.5,1))
                    ti.bind(text=self.on_text)
                    self.counter += 1
                    subcont.add_widget(l)
                    subcont.add_widget(ti)
                    container.add_widget(subcont)
        save_button = Button(text='save')
        save_button.bind(on_release=self.save_config)
        container.add_widget(save_button)
        self.add_widget(container)
                # key, value = stripped_line.split("=")
                # print('key:', key)
                # print('value:', value)
    def on_text(self, instance, value):
        key = instance.parent.children[1].text
        #print(self.dict_ti[key])
        self.dict[key] = value
        print(self.dict[key])
    
    def save_config(self, instance): 
        print(self.dict)
        with open("config.py", "r") as a_file:
            lines = a_file.readlines()
        with open("config.py", "w") as a_file:
            for line in lines:
                stripped_line = line.strip()
                list = stripped_line.split(" = ")
                if not (stripped_line.startswith('#') or stripped_line == ""):
                    key = list[0]
                    value = self.dict[key]
                    if len(list[1].split("#"))>1:
                        comment = '#'+list[1].split("#")[1]
                    else:
                        comment = ""
                    line_new = key + ' = ' + value + comment + '\n'
                    a_file.write(line_new)
                else:
                    a_file.write(line)
        self.dismiss()
                    
                
        
 
class MyWidget(BoxLayout):
    filepath = StringProperty()
    def __init__(self, **kwargs):
        
        self.filepath = str(TSV_READ_DATA)
        self.netpath = str(LOAD_MODEL_NAME)
        self.url = str(WEB_ADDRESS)
        super(MyWidget, self).__init__(**kwargs)
        # with self.canvas:
            # Color(1., 1, 1)
            # Rectangle(pos=(0, 0), size=(1820, 1080))
        root = BoxLayout(orientation='horizontal')
       
        container = BoxLayout(orientation='vertical')
        h = Label(text='TRAINING', font_size='30sp')
        container.add_widget(h)
        l = Label(text='Select data to use for training')
        subcont1 = BoxLayout(orientation = 'horizontal')
        self.textinput = TextInput(text = self.filepath, hint_text ='Select data file (tsv)', size_hint=(.9, 1))
        browse_btn = Button(text='...', size_hint=(.1, 1))
        browse_btn.bind(on_release=lambda x: self.show_train())
        subcont1.add_widget(self.textinput)
        subcont1.add_widget(browse_btn)
        container.add_widget(l)
        container.add_widget(subcont1)
        l2 = Label(text='Train CNN or BiLSTM')
        train_cnn_btn = Button(text='Train cnn', size_hint=(.5, 1))
        train_cnn_btn.bind(on_release=lambda x: self.train_cnn())
        train_BiLSTM_btn = Button(text='Train BiLSTM', size_hint=(.5, 1))
        train_BiLSTM_btn.bind(on_release=lambda x: self.train_lstm())
        subcont2 = BoxLayout(orientation = 'horizontal')
        subcont2.add_widget(train_cnn_btn)
        subcont2.add_widget(train_BiLSTM_btn)
        #container.add_widget(l2)
        container.add_widget(subcont2)
        #container.add_widget(self.progress_bar)
        h = Label(text='TESTING', font_size='30sp')
        container.add_widget(h)
        l2 = Label(text='Select model to use for testing')
        container.add_widget(l2)
        subcont3 = BoxLayout(orientation = 'horizontal')
        self.textinput2 = TextInput(text = self.netpath, hint_text ='Select pt file', size_hint=(.9, 1))
        browse_btn2 = Button(text='...', size_hint=(.1, 1))
        browse_btn2.bind(on_release=lambda x: self.show_test())
        subcont3.add_widget(self.textinput2)
        subcont3.add_widget(browse_btn2)
        container.add_widget(subcont3)
        test_btn = Button(text="test selected model")
        test_btn.bind(on_release=lambda x: self.test_net())
        container.add_widget(test_btn)
        h = Label(text='OTHER OPTIONS', font_size='30sp')
        container.add_widget(h)
        cfg_btn = Button(text='Change config.py')
        cfg_btn.bind(on_release=lambda x: self.set_config())
        container.add_widget(cfg_btn)
        
        container2 = BoxLayout(orientation='vertical')
        h = Label(text='EXTRACT DATA', font_size='30sp')
        container2.add_widget(h)
        l = Label(text='Type Web address to extract data from')
        self.textinput3 = TextInput(text=WEB_ADDRESS)
        container2.add_widget(l)
        container2.add_widget(self.textinput3)
        l = Label(text='Select or type filename to save extracted data to')
        container2.add_widget(l)
        subcont4 = BoxLayout(orientation='horizontal')
        self.textinput4 = TextInput(text=TSV_NEW, size_hint=(.9,1))
        browse_btn3 = Button(text='...', size_hint=(.1, 1))
        browse_btn3.bind(on_release=lambda x: self.show_ext())
        subcont4.add_widget(self.textinput4)
        subcont4.add_widget(browse_btn3)
        container2.add_widget(subcont4)
        ex_btn = Button(text='Extract elements')
        ex_btn.bind(on_release=lambda x: self.extract())
        container2.add_widget(ex_btn)
        h = Label(text='AUGMENT DATA', font_size='30sp')
        container2.add_widget(h)
        l = Label(text='Select data to augment and a file to save augmentation to')
        container2.add_widget(l)
        subcont5 = BoxLayout(orientation='horizontal')
        self.textinput5 = TextInput(text=TSV_NEW, size_hint=(.9,1))
        browse_btn4 = Button(text='...', size_hint=(.1, 1))
        browse_btn4.bind(on_release=lambda x: self.show_augload())
        subcont5.add_widget(self.textinput5)
        subcont5.add_widget(browse_btn4)
        subcont6 = BoxLayout(orientation='horizontal')
        self.textinput6 = TextInput(text=CSV_WRITE, size_hint=(.9,1))
        browse_btn5 = Button(text='...', size_hint=(.1, 1))
        browse_btn5.bind(on_release=lambda x: self.show_augwrite())
        subcont6.add_widget(self.textinput6)
        subcont6.add_widget(browse_btn5)
        container2.add_widget(subcont5)
        container2.add_widget(subcont6)
        aug_btn = Button(text='augment data')
        aug_btn.bind(on_release=lambda x: self.augment())
        container2.add_widget(aug_btn)
        root.add_widget(container2)
        root.add_widget(container)
        self.add_widget(root)
    
    def augment(self):
        import config as cfg
        cfg.TSV_NEW = self.textinput5.text
        cfg.CSV_WRITE = self.textinput6.text
        augment_data(cfg.TSV_NEW, cfg.CSV_WRITE, cfg.AUG_AMOUNT)
    
    def extract(self):
        cfg.WEB_ADDRESS = self.textinput3.text
        cfg.TSV_NEW = self.textinput4.text
        extract_el()
    
    def test_net(self):
        self.update_test()
        test()
     
    def update_train(self):
        self.textinput.text = cfg.TSV_READ_DATA
    
    def update_test(self):
        self.textinput2.text = cfg.LOAD_MODEL_NAME
    
    def update_ext(self):
        self.textinput4.text = cfg.TSV_NEW
    
    def update_augload(self):
        self.textinput5.text = cfg.TSV_NEW
    
    def update_augwrite(self):
        self.textinput6.text = cfg.CSV_WRITE
    
    def set_config(self):
        popupWindow = MyConfig()
        popup_open = True
        popupWindow.open()
    
    def show_train(self):
        popupWindow = MyBrowser('Select data to use for training', 'Select a file', "train") 
    # Create the popup window
        popup_open = True
        popupWindow.open() # show the popup
        popupWindow.bind(on_dismiss=lambda x:self.update_train())
        
    def show_test(self):
        popupWindow = MyBrowser('Select pt (network) file to test', 'Select a file', "test") 
    # Create the popup window
        popup_open = True
        popupWindow.open() # show the popup
        popupWindow.bind(on_dismiss=lambda x:self.update_test())    
    
    def show_ext(self):
        popupWindow = MyBrowser('Select tsv file to save extraction to', 'Select a file', "extract") 
    # Create the popup window
        popup_open = True
        popupWindow.open() # show the popup
        popupWindow.bind(on_dismiss=lambda x:self.update_ext())    
    
    def show_augload(self):
        popupWindow = MyBrowser('Select tsv file to augment', 'Select a file', "augload") 
    # Create the popup window
        popup_open = True
        popupWindow.open() # show the popup
        popupWindow.bind(on_dismiss=lambda x:self.update_augload()) 
    
    def show_augwrite(self):
        popupWindow = MyBrowser('Select tsv file to save augmentation to', 'Select a file', "augwrite") 
    # Create the popup window
        popup_open = True
        popupWindow.open() # show the popup
        popupWindow.bind(on_dismiss=lambda x:self.update_augwrite()) 
    
    def train_cnn(self):
        self.update_train()
        tokenizer, le, train_X, test_X, train_y, test_y, embedding_matrix = prep_data()
        model = CNN_Text(le, embedding_matrix)
        action = network('cnn')
        p1 = Process(target = train_nn, args=(cfg.N, model, train_X, train_y, test_X, test_y, le, action))
        p1.start()
        # p2 = Process(target = self.update_progress)
        # p2.start()
        
        #train_nn(N, model, train_X, train_y, test_X, test_y, le, action)
        # os.system("python train.py -n cnn")
        
    def train_lstm(self):
        self.update_train()
        tokenizer, le, train_X, test_X, train_y, test_y, embedding_matrix = prep_data()
        model = BiLSTM(le, embedding_matrix)
        action = network('lstm')
        p1 = Process(target = train_nn, args=(N, model, train_X, train_y, test_X, test_y, le, action))
        p1.start()
    
    def update_progress(self):
        while True:
            global progress
            rel_prog = progress/N
            self.progress_bar.value = self.progress_bar.max*rel_prog
            sleep(0.5)
    

class MyApp(App):
    def build(self):
        return MyWidget()
 
 
if __name__ == '__main__':
    MyApp().run()