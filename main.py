import os
import json
import pickle
import winsound
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from CNN import Modded_CNN
import FullReportMaker as FR  
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Sort_o_and_I import Is_O_or_I_old,Is_O_or_I
from sklearn.model_selection import train_test_split
from Opener import getData

def prepare_data():
    attribute_dict = {}
    attribute_dict["ALL"] = {'X': [], 'y': [],"Letters":[]}

    for i, image in tqdm(enumerate(train_x)):
        O, I, H_l = Is_O_or_I_old(image,[17,4,17,4],0.9)
        key = f"O_{O}_I_{I}_Hl_{H_l}"        
        if key not in attribute_dict:
            attribute_dict[key] = {'X': [], 'y': [],"Letters":[]}       
        
        if np.argwhere(train_y[i] == 1)[0][0] not in attribute_dict[key]['Letters']:
            attribute_dict[key]['Letters'].append(np.argwhere(train_y[i] == 1)[0][0])        
        attribute_dict["ALL"]['X'].append(image)
        attribute_dict["ALL"]['y'].append(train_y[i])


    for i, image in tqdm(enumerate(train_x)):
        for key in attribute_dict.keys():
        
                if key !="ALL" and np.argwhere(train_y[i] == 1)[0][0] in attribute_dict[key]['Letters']:
                    attribute_dict[key]['X'].append(image)
                    attribute_dict[key]['y'].append(train_y[i])
                    
    ListToRemove = []
    for key in attribute_dict.keys():
        if len(attribute_dict["ALL"]['X']) == len(attribute_dict[key]['X']):
            if key != "ALL":
                ListToRemove.append(key)
            
    for key0 in ListToRemove:
        ReturnedValue = attribute_dict.pop(key0)
    
    return attribute_dict

def remap_labels(y_train_int, y_test_int):
    
    unique_labels = np.unique(y_train_int)
    label_mapping = {}
    for new_label, old_label in tqdm(enumerate(unique_labels), desc="Remapping Labels"):
        label_mapping[old_label] = new_label
        
    y_train_int_list = []  
    for label in tqdm(y_train_int, desc="Processing y_train"):
        y_train_int_list.append(label_mapping[label])  

    y_train_int = np.array(y_train_int_list)
    y_test_int_list = [] 
    for label in tqdm(y_test_int, desc="Processing y_test_int"):
        y_test_int_list.append(label_mapping[label]) 

    y_test_int = np.array(y_test_int_list)
    
    return y_train_int, y_test_int, label_mapping

def save_dict_to_file(attribute_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(attribute_dict, f)
        
def load_dict_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def train_and_save_models(attribute_dict,ShouldTrain):
    
    model_paths = {}
    os.makedirs("models", exist_ok=True)
    valDict = {} 
    for key, data in tqdm(attribute_dict.items()):
        
        X_train, X_test, y_train, y_test = train_test_split(np.array(data['X']), np.array(data['y']), test_size=0.3, random_state=42)
     
        
        y_train_int = np.argmax(y_train, axis=1)
        num_classes = np.unique(y_train_int).size
        y_test_int = np.argmax(y_test, axis=1)
        model_path = f"models/{key}.keras"
        unique_labels = np.unique(y_train_int)
        y_train_int, y_test_int, label_mapping  = remap_labels(y_train_int, y_test_int)
        unique_labels = np.unique(y_train_int)
        y_test_int0 = np.unique(y_test_int)

        if key not in valDict:
            valDict[key] = {'X': [], 'y': [],"map":label_mapping}
            
        valDict[key]['X'].append(X_test)
        valDict[key]['y'].append(y_test_int)
        if ShouldTrain:
            model = Modded_CNN(X_train, X_test, y_train_int, y_test_int, num_classes, 2, "models",key)
        
        model_paths[key] = model_path        
    with open("models/model_index.pkl", "wb") as f:
        pickle.dump(model_paths, f)

    return valDict

def FullRun(image):
    with open("models/model_index.pkl", "rb") as f:
        model_paths = pickle.load(f)
    
    O, I, H_l = Is_O_or_I(image)
    key = f"O_{O}_I_{I}_Hl_{H_l}"
    
    if key in model_paths:
        model = load_model(model_paths[key])
        image = image.reshape(1, 32, 32, 1)
        prediction = model.predict(image)
        return np.argmax(prediction)
    else:
        return "No relevant model found."

def ReportMaker(valDict):
    report = "System Performance Report\n\n"
    

    with open("models/model_index.pkl", "rb") as f:
        model_paths = pickle.load(f)
    
    for key, path in model_paths.items():
        models_folder = os.path.join(os.getcwd(), 'models')  
        path2 = os.path.join(models_folder, key + ".keras")  
        
        model = load_model(path2)
        loss, accuracy = model.evaluate(valDict[key]["X"], valDict[key]["y"], verbose=0)
        
        report += f"Model: {key}\nAccuracy: {accuracy * 100:.2f}%\n\n"
    
    with open("models/system_reportTry4.txt", "w") as f:
        f.write(report)
    
    print("Report saved as system_reportTry4.txt")



def CombinedReportMaker(ListForSort, attribute_dict, test_x, test_y, valDict):

    Totalcount = 0 
    
    Sorted_data = FR.SortData(test_x, test_y, ListForSort, attribute_dict)

    indexCounter = 0
    
    
    with open("models/model_index.pkl", "rb") as f:
        model_paths = pickle.load(f)

    for key in Sorted_data:
        
        models_folder = os.path.join(os.getcwd(), 'models') 
        model_path = os.path.join(models_folder, f"{key}.keras")
        TempDict = valDict[key]["map"]
        model = load_model(model_path)
        for Local_index, image in enumerate(Sorted_data[key]["data"]):
            indexCounter += 1
            imageModded = image.reshape(1, 28, 28, 1) 
            prediction = model.predict(imageModded)
            
            
            Test_y_modded = np.int64(np.argwhere(test_y[Sorted_data[key]["indexs"][Local_index]] == 1)[0][0])

            if key == "ALL":
                if np.argmax(prediction) == TempDict[Test_y_modded]:
                    Totalcount += 1
            else:
                if np.argmax(prediction) == Test_y_modded:
                    Totalcount += 1
                    break  

    print(Totalcount, "Totalcount") 
    print((Totalcount / 18799) * 100, "Totalcount percent")

    report = ""
    report += f"Totalcount: {Totalcount}\n"
    report += f"Accuracy: {(Totalcount / 18799) * 100:.2f}%\n"

    with open("models/Full_System_Report.txt", "w") as f:
        f.write(report)




train_x, train_y, test_x, test_y,Full_x,Full_y = getData()
ModelPath = 'models/TestFasterAttributeDict7.pkl'

attribute_dict = load_dict_from_file(ModelPath)
print(attribute_dict.keys(),"attribute_dict.keys()")

ShouldTrain = False
valDict = train_and_save_models(attribute_dict,ShouldTrain)
ListForSort= [[17,4,17,4],0.9]
a = CombinedReportMaker(ListForSort,attribute_dict, test_x, test_y,valDict)
