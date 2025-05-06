import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Sort_o_and_I import Is_O_or_I_old
from tqdm import tqdm

def SortData(test_x,test_y,ListForSort,attribute_dict):
    
    Sorted_data = {}
    Sorted_List = [] 
    for i, image in tqdm(enumerate(test_x)):
        
        O, I, H_l = Is_O_or_I_old(image,ListForSort[0],ListForSort[1])
        key = f"O_{O}_I_{I}_Hl_{H_l}"
        Flag = True 
        for keyTest in attribute_dict.keys():
            if keyTest == key:
                Flag = False
        if Flag:
            key = "ALL"
            
        if key not in Sorted_List:
            Sorted_data[key]  = {"data":[],"indexs":[]}
            Sorted_List.append(key)
        Sorted_data[key]["data"].append(image)
        Sorted_data[key]["indexs"].append(i)
        
    return Sorted_data
            
        
    
    

def CombinedReportMaker(ListForSort,attribute_dict,test_x, test_y):
    
    
    with open("models/model_index.pkl", "rb") as f:
        model_paths = pickle.load(f)
    for i, image in enumerate(test_x):
       
        if i == 1:
            break
    
    return 0;
            